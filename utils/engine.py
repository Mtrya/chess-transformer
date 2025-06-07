"""An engine class to provide a universal way to interact with both chessformer and stockfish"""
import torch
import chess
import math
import chess.engine
import multiprocessing
from dataclasses import dataclass, field
from functools import partial
import time
import os

try:
    from .mapping import UCI_MOVE_TO_IDX, IDX_TO_UCI_MOVE
except ImportError:
    from mapping import UCI_MOVE_TO_IDX, IDX_TO_UCI_MOVE
from torch.distributions import Categorical
from typing import Optional, Tuple, List, Union

@dataclass
class ChessformerConfig:
    chessformer: torch.nn.Module=None
    device: Optional[torch.device]=None
    temperature: float=0.5
    depth: int=2
    top_k: int=8
    decay_rate: float=0.6
    max_batch_size: int=896

@dataclass
class StockfishConfig:
    engine_path: str="/usr/games/stockfish"
    depth: int=16


def _stockfish_worker(board_fen: str, engine_path: str, depth: int) -> Optional[Tuple[str, float]]:
    """
    Analyzes a single board FEN using a temporary Stockfish engine instance.
    Designed for use with multiprocessing.
    Returns the best move UCI and the normalized score [-1,1].
    Does not handle draw claims explicitly as FEN lacks history.
    Caller should check board.is_game_over() on the main board object.
    """
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        # initialize board from FEN - history is lost here
        board = chess.Board(board_fen)

        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        
        score_obj = info.get("score")
        pv = info.get("pv")

        if score_obj is None or pv is None or not pv:
            # Analysis failed
            print(f"Warning: Stockfish analysis failed for FEN: {board_fen}")
            return None
        
        best_move_uci = pv[0].uci()
        pov_score = score_obj.pov(board.turn)
        cp = None

        if pov_score.is_mate():
            mate_score = pov_score.mate()
            cp = 10000.0 if mate_score > 0 else -10000.0
        elif pov_score.cp is not None:
            cp = float(pov_score.cp)
        else:
            print(f"Warning: Stockfish score object lacks cp/mate for FEN: {board_fen}")
            return None # analysis is unclear

        normalized_cp = 2 / (1 + math.exp(-0.004*cp)) - 1        

        return best_move_uci, normalized_cp
    
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, FileNotFoundError, ValueError) as e:
        print(f"Stockfish worker error for FEN {board_fen}: {e}")
        return None
    finally:
        if engine:
            engine.quit()

def _compute_repetition_single(board: chess.Board) -> int:
    """Compute repetition count. Used in _chessformer_move and _batch_chessformer_move"""
    
    transposition_key = board._transposition_key()
    count = 0
    if board.move_stack:
        if board._transposition_key() == transposition_key:
            count = 1
    else:
        count = 1
    try:
        # Iterate back through history
        while board.move_stack:
            move = board.pop() # note that history is lost here
            if board.is_irreversible(move):
                break
            if board._transposition_key() == transposition_key:
                count += 1
    except Exception as e:
        print(f"Error occurred during repetition count for board {board.fen()}: {e}")
        return 1 # fallback to 1
    return max(1, count)
    
# Engine class, designed to be used in the Evaluator class and app.py
class Engine:
    def __init__(self,
                 type: str,
                 chessformer_config: Optional[ChessformerConfig]=None,
                 stockfish_config: Optional[StockfishConfig]=None):
        self.type = type
        if type == "chessformer":
            if chessformer_config is None:
                raise ValueError("ChessformerConfig must be provided for chessformer engine.")
            
            self.config = chessformer_config
            if self.config.chessformer is None:
                raise ValueError("ChessFormer model must be provided in config.")
            
            if self.config.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(self.config.device, str):
                self.device = torch.device(self.config.device)
            else:
                self.device = self.config.device

            self.model = self.config.chessformer
            self.model.to(self.device)
            self.model.eval()

            if not (self.config.temperature > 0):
                raise ValueError("Temperature must be greater than 0.")
            if not (self.config.top_k > 0):
                raise ValueError("Top-k must be greater than 0.")
            if not (self.config.depth >= 0):
                raise ValueError("Depth must be greater than or equal to 0.")
            if not (0.0 < self.config.decay_rate <= 1.0):
                raise ValueError("Decay rate must be in range (0.0,1.0].")
            if not (self.config.max_batch_size > 0):
                raise ValueError("Max batch size must be an integer greater than 0.")
                
            self.temperature = self.config.temperature
            self.top_k = self.config.top_k
            self.initial_k = self.top_k
            self.depth = self.config.depth
            self.decay_rate = self.config.decay_rate
            self.max_batch_size = self.config.max_batch_size
        elif type == "stockfish":
            if stockfish_config is None:
                raise ValueError("StockfishConfig must be provided for stockfish engine.")
            
            self.config = stockfish_config
            self.engine_path = self.config.engine_path
            self.depth = self.config.depth
            if self.config.engine_path is None:
                raise ValueError("Engine path must be provided in config.")
            try:
                with chess.engine.SimpleEngine.popen_uci(self.config.engine_path) as test:
                    pass
            except (FileNotFoundError, chess.engine.EngineError) as e:
                raise ValueError(f"Invalid engine path or engine not found: {e}")
        else:
            raise ValueError("Invalid engine type. Choose 'chessformer' or 'stockfish'.")

    def get_invalid_mask(self, boards: List[chess.Board]) -> torch.Tensor:
        bs = len(boards)
        possible_moves = len(UCI_MOVE_TO_IDX)
        invalid_mask = torch.full((bs,possible_moves), -torch.inf, dtype=torch.float32, device=self.device)
        for idx,board in enumerate(boards):
            if board.is_game_over(claim_draw=True):
                continue # leave all as -inf
            legal_moves = list(board.legal_moves)
            legal_move_ids = [UCI_MOVE_TO_IDX[move.uci()] for move in legal_moves]
            if legal_move_ids:
                invalid_mask[idx,legal_move_ids] = 0
            if board.can_claim_draw():
                invalid_mask[idx,0] = 0
        
        return invalid_mask

    def compute_repetition(self, boards: List[chess.Board]) -> torch.Tensor:
        """Use multiprocessing to compute repetition count for a batch of boards."""
        bs = len(boards)
        num_workers = min(bs, max(1, os.cpu_count()//2 if os.cpu_count else 1))
        if bs < num_workers * 2: # avoid overhead for very small batches per worker
            num_workers = max(1, bs//2)

        try:
            if num_workers > 1 and bs > 1:
                board_copies = [board.copy(stack=True) for board in boards]
                with multiprocessing.Pool(processes=num_workers) as pool:
                    counts = pool.map(_compute_repetition_single, board_copies)
            else:
                # Run sequentially if only one worker needed or very small batch
                counts = [_compute_repetition_single(b.copy(stack=True)) for b in boards]

            counts_tensor = torch.tensor(counts, dtype=torch.long, device=self.device)
            return counts_tensor # (B,)
        except Exception as e:
            print(f"Error during batch repetition computation: {e}")
            # Fall back to single board computation if multiprocessing fails
            return torch.ones((bs,),dtype=torch.long, device=self.device)

    def _raw_chessformer_move(self, board: chess.Board, return_perplexity: bool=False) -> Tuple[str,float]:
        """Get the next move from ChessFormer model with optional tactical verification."""
        # Get FEN
        fen = board.fen()

        # Compute repetition
        count_tensor = self.compute_repetition([board])

        move_logits, value = self.model([fen],count_tensor)
        move_logits = move_logits.squeeze(0) # remove batch dimension since it will always be 1
        value = value.squeeze(0).item()
        
        # Calculate invalid mask
        legal_moves = list(board.legal_moves)
        if not legal_moves and not board.can_claim_draw():
            # No legal moves. Should not happen if this function is called correctly, but it wouldn't hurt to add a check
            return None
        legal_move_ids = [UCI_MOVE_TO_IDX[move.uci()] for move in legal_moves]
        invalid_mask = torch.ones_like(move_logits) * (-torch.inf)
        invalid_mask[legal_move_ids] = 0
        if board.can_claim_draw():
            invalid_mask[0] = 0
        move_logits = move_logits + invalid_mask

        if return_perplexity:
            probs = torch.softmax(move_logits, dim=-1)
            perplexity = torch.exp(-torch.sum(probs*torch.log(probs+1e-8))).item()
        
        top_k_ids = torch.topk(move_logits, self.top_k, dim=-1).indices
        top_k_mask = torch.ones_like(move_logits) * (-torch.inf)
        top_k_mask[top_k_ids] = 0
        move_logits = move_logits + top_k_mask
        move_logits = move_logits / self.temperature

        # Sample
        dist = Categorical(logits=move_logits)
        move_id = dist.sample().item()
        move = IDX_TO_UCI_MOVE[move_id]
        if return_perplexity:
            return move, value, perplexity
        else:
            return move, value

    def _search_enhanced_move(self, board: chess.Board, return_perplexity: bool=False, verbose: bool=False) -> Tuple[str,float]:
        """Get move from chessformer using tactical search"""
        # Step 1: Build search tree level by level
        current_boards = [board] # aggregate board to a list for batch inference
        board_probs = [1] # the probabilities of getting to this position (estimated)

        terminal_leaves = [] # (root_move, prob, game_result_value) ^from white's perspective
        search_leaves = [] # (root_move, prob, board) - leaves not terminal but reached max depth therefore needs evaluation from model

        # Track which root_move each board came from
        board_to_root_move = [None] # root board has no parent move

        for depth in range(self.depth+1):
            if not current_boards:
                break
            k = max(1,int(self.initial_k*(self.decay_rate**depth)))

            fens = [b.fen() for b in current_boards]
            reps = self.compute_repetition(current_boards)

            with torch.no_grad():
                logits, values = self.model(fens,reps)

            next_boards = []
            next_board_probs = []
            next_board_to_root_move = []

            # Process each board at current depth
            for board_idx, current_board in enumerate(current_boards):
                board_logits = logits[board_idx]
                board_prob = board_probs[board_idx]
                parent_root_move = board_to_root_move[board_idx]
                
                # Check if game is over
                if current_board.is_game_over(claim_draw=True):
                    outcome = current_board.outcome(claim_draw=True)
                    if outcome.winner == chess.WHITE:
                        game_value = 1.0
                    elif outcome.winner == chess.BLACK:
                        game_value = -1.0
                    else:
                        game_value = 0.0
                    terminal_leaves.append((parent_root_move, board_prob, game_value))
                    continue

                # If we've reached max depth, add to search leaves
                if depth == self.depth:
                    search_leaves.append((parent_root_move, board_prob, current_board))
                    continue

                # Otherwise, recursively search deeper
                invalid_mask = self.get_invalid_mask([current_board])[0]
                masked_logits = board_logits + invalid_mask

                top_k_values, top_k_indices = torch.topk(masked_logits,k=min(k,torch.sum(invalid_mask==0).item()))
                top_k_probs = torch.softmax(top_k_values,dim=0)
                if depth==0:
                    initial_masked_logits = masked_logits.squeeze(0)
                    initial_invalid_mask = invalid_mask.squeeze(0)
                    initial_top_k_indices = top_k_indices

                # Expand each top k move
                for i,move_idx in enumerate(top_k_indices):
                    move_prob = top_k_probs[i].item()
                    move_uci = IDX_TO_UCI_MOVE[move_idx.item()]

                    root_move = parent_root_move if parent_root_move is not None else move_uci

                    new_board = current_board.copy()

                    if move_uci == "<claim_draw>":
                        if new_board.can_claim_draw():
                            terminal_leaves.append((root_move,board_prob*move_prob,0.0))
                            continue
                        else:
                            continue # should not happen, invalid draw claim
                    else:
                        move = chess.Move.from_uci(move_uci)
                        new_board.push(move)
                    
                    next_boards.append(new_board)
                    next_board_probs.append(board_prob*move_prob)
                    next_board_to_root_move.append(root_move)

            current_boards = next_boards
            board_probs = next_board_probs
            board_to_root_move = next_board_to_root_move

        # Step 2: Evaluate all search leaves
        if search_leaves:
            search_boards = [leaf[2] for leaf in search_leaves]
            search_fens = [b.fen() for b in search_boards]
            search_reps = self.compute_repetition(search_boards)

            with torch.no_grad():
                _, search_values = self.model(search_fens, search_reps)

            for i, (root_move, prob, leaf_board) in enumerate(search_leaves):
                value = search_values[i].item()
                white_perspective_value = value if leaf_board.turn == chess.WHITE else -value
                terminal_leaves.append((root_move,prob,white_perspective_value))
        
        # Step 3: Aggregate all leaves using probability weights
        root_move_weighted_values = {}
        root_move_total_probs = {}
        for root_move, prob, value in terminal_leaves:
            if root_move not in root_move_weighted_values:
                root_move_weighted_values[root_move] = 0.0
                root_move_total_probs[root_move] = 0.0
            root_move_weighted_values[root_move] += prob * value
            root_move_total_probs[root_move] += prob
        
        final_value = sum(root_move_weighted_values.values())
        final_value = final_value if board.turn == chess.WHITE else -final_value
        
        root_move_values = {}
        for root_move in root_move_total_probs:
            if root_move_total_probs[root_move] > 0:
                root_move_values[root_move] = root_move_weighted_values[root_move] / root_move_total_probs[root_move]
            else:
                root_move_values[root_move] = 0

        # Step 4: Confidence-based weighting with search results
        initial_probs = torch.softmax(initial_masked_logits,dim=0)
        entropy = -torch.sum(initial_probs*torch.log(initial_probs+1e-8))
        max_entropy = math.log(torch.sum(initial_invalid_mask==0).item())
        confidence = 1.0 - (entropy/max_entropy) if max_entropy > 0 else 1.0

        if root_move_values:
            search_adjustment_logits = torch.zeros_like(initial_masked_logits)
            for move_uci,search_value in root_move_values.items():
                move_idx = UCI_MOVE_TO_IDX[move_uci]
                search_adjustment_logits[move_idx] += search_value
            # flip value according to perpective
            search_adjustment_logits = search_adjustment_logits if board.turn==chess.WHITE else -search_adjustment_logits
            search_adjustment_logits = search_adjustment_logits - search_adjustment_logits.mean()

            # Normalize search logits to be in the same norm as the initial logits

            initial_valid_norm = torch.norm(initial_masked_logits[initial_top_k_indices]) + 1e-8
            search_valid_norm = torch.norm(search_adjustment_logits[initial_top_k_indices]) + 1e-8

            normalized_search = search_adjustment_logits * initial_valid_norm / search_valid_norm
            normalized_initial = initial_masked_logits

            adjusted_logits = confidence * normalized_initial + (1 - confidence) * normalized_search
        else:
            adjusted_logits = initial_masked_logits

        # Apply temperature and top-k filtering
        top_k_mask = torch.full_like(adjusted_logits, -torch.inf)
        top_k_mask[initial_top_k_indices] = 0
        adjusted_logits = adjusted_logits + top_k_mask
        adjusted_logits = adjusted_logits / self.temperature

        dist = Categorical(logits=adjusted_logits)
        move_idx = dist.sample().item()
        move_uci = IDX_TO_UCI_MOVE[move_idx]

        if return_perplexity:
            final_probs = torch.softmax(adjusted_logits,dim=0)
            perplexity = torch.exp(-torch.sum(final_probs * torch.log(final_probs + 1e-8))).item()

            if verbose and self.depth > 0:
                print(f"\n--- Search Enhanced Move Debug Info ({board.fen()}) ---")
                print(f"Confidence: {confidence:.4f}")

                print("\nMove Analysis (Initial Top-K Candidates):")
                print(f"{'Move':<8} {'Initial Logit':<15} {'Search Adj. Logit':<19} {'Final Adj. Logit':<18} {'Final Prob':<12}")
                print(f"{'-'*8:<8} {'-'*15:<15} {'-'*19:<19} {'-'*18:<18} {'-'*12:<12}")

                for i, idx in enumerate(initial_top_k_indices):
                    move_uci_v = IDX_TO_UCI_MOVE[idx.item()]
                    initial_logit = normalized_initial[idx].item()
                    
                    search_adj_logit_val = normalized_search[idx].item() if root_move_values else 0.0
                    
                    final_adj_logit = adjusted_logits[idx].item()
                    final_prob_val = final_probs[idx].item()

                    print(f"{move_uci_v:<8} {initial_logit:<15.4f} {search_adj_logit_val:<19.4f} {final_adj_logit:<18.4f} {final_prob_val:<12.4f}")

                print(f"\nPerplexity: {perplexity:.4f}")
                print(f"Predicted Value (White's POV): {final_value:.4f}")

                print("\nLeaf Node Values (Root Move, Probability, Value from White's POV):")
                for rm, prob, val in terminal_leaves:
                    print(f"  Root Move: {rm:<8}, Prob: {prob:<.4f}, Value: {val:<.4f}")
                print("--------------------------------------------------")

            return move_uci, final_value, perplexity
        else:
            return move_uci, final_value

    def _chessformer_move(self, board: chess.Board, return_perplexity: bool=False, verbose: bool=False) -> Tuple[str,float]:
        """Get move from chessformer with optional search enhance"""
        if self.depth == 0:
            return self._raw_chessformer_move(board,return_perplexity)
        else:
            return self._search_enhanced_move(board,return_perplexity,verbose)

    def _stockfish_move(self, board: chess.Board, return_perplexity: bool=False) -> Tuple[str,float]:
        """Get best move from stockfish"""
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
            print(f"Stockfish error: {e}")
            return None

        loss_threshold = -0.4

        score_obj = info.get("score")
        can_claim_draw = board.can_claim_draw()
        if score_obj is None or info.get("pv") is None or not info.get("pv"):
            # Invalid analysis result
            return None
        
        pv = info["pv"]
        pov_score = score_obj.pov(chess.WHITE)
        cp = None
        if pov_score.is_mate():
            mate_score = pov_score.mate()
            cp = 10000.0 if mate_score > 0 else -10000.0
            relative_score = score_obj.relative
            if relative_score.is_mate():
                cp = 10000.0 if relative_score.mate() > 0 else -10000.0
            else:
                if relative_score.cp is not None:
                    cp = float(relative_score.cp)
                else:
                    return None

        elif pov_score.cp is not None:
            relative_score = score_obj.relative
            if relative_score.cp is not None:
                cp = float(relative_score.cp)
            else:
                return None

        else:
            return None
        
        if cp is not None:
            normalized_score = 2 / (1+math.exp(-0.004*cp)) - 1
        else:
            return None

        if can_claim_draw and normalized_score < loss_threshold:
            best_move_uci = "<claim_draw>"
        else:
            best_move_uci = pv[0].uci()

        if engine:
            engine.quit()

        if return_perplexity:
            return best_move_uci, normalized_score, None
        else:
            return best_move_uci, normalized_score

    def _batch_chessformer_move(self, boards: List[chess.Board]) -> List[Tuple[str, float]]:
        """Get the next moves from Chessformer model using batch inference."""
        bs = len(boards)
        if bs > self.max_batch_size:
            raise ValueError(f"num boards ({bs}) exceeded max batch size ({self.max_batch_size}).")
        fens = [board.fen() for board in boards]

        count_tensor = self.compute_repetition(boards) # shape (bs, 1)
        count_tensor = count_tensor.to(self.device)

        with torch.no_grad():
            move_logits, values = self.model(fens, count_tensor)

        invalid_mask = self.get_invalid_mask(boards)

        # Apply mask
        move_logits = move_logits + invalid_mask

        all_masked = torch.all(torch.isinf(move_logits), dim=-1)

        # Apply top-p filtering
        if 0.0 < self.top_p < 1.0: # Apply only if top_p is strictly between 0 and 1
            sorted_logits, sorted_indices = torch.sort(move_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(move_logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            move_logits[indices_to_remove] = -torch.inf

        # Apply temperature
        temp = self.temperature if self.temperature > 0 else 1.0
        move_logits = move_logits / temp

        # Sample moves
        dist = Categorical(logits=move_logits)
        try:
            sampled_indices = dist.sample()
        except RuntimeError as e:
            print(f"Error sampling moves: {e}. Checking logit values...")
            results = []
            for i in range(bs):
                print(f"Board {i} logits sum: {torch.logsumexp(move_logits[i], dim=-1)}")
                results.append(None) # indicate failure
            return results
        
        results = []
        for i in range(bs):
            if all_masked[i]:
                results.append(None) # Game already over
                continue

            move_id = sampled_indices[i].item()
            move_uci = IDX_TO_UCI_MOVE.get(move_id)
            value = values[i].item()

            if move_uci is None:
                print(f"Warning: Sampled move ID {move_id} not in IDX_TO_UCI_MOVE map")
                results.append(None)
                continue

            results.append((move_uci, value))

        
        return results

    def _batch_stockfish_move(self, boards: List[chess.Board], allow_claim_draw: bool=False) -> List[Tuple[str, float]]:
        """Get the next moves from Stockfish engine using multiprocessing"""
        if allow_claim_draw:
            """Use sequential processing to maintain board history"""
            return [self._stockfish_move(board) for board in boards]
        else:
            """Use multiprocessing to speed up if no need to include claim draw logic"""
            bs = len(boards)
            num_workers = min(bs, max(1, os.cpu_count()//2 if os.cpu_count() else 1))
            if bs < num_workers * 2:
                num_workers = max(1, bs//2)
                if bs == 1: num_workers = 1

            board_fens = [board.fen() for board in boards]

            worker_func = partial(_stockfish_worker,
                                  engine_path=self.engine_path,
                                  depth=self.depth)
            results: List[Optional[Tuple[str,float]]] = [None] * bs

            active_indices = [i for i,b in enumerate(boards) if not b.is_game_over(claim_draw=True)]
            active_fens = [board_fens[i] for i in active_indices]

            if not active_fens:
                # All games are over
                return results # list of None
            
            try:
                if num_workers > 1 and len(active_fens) > 1:
                    with multiprocessing.Pool(processes=num_workers) as pool:
                        worker_results = pool.map(worker_func, active_fens)
                else:
                    worker_results = [worker_func(fen) for fen in active_fens]

                for i, res in enumerate(worker_results):
                    original_index = active_indices[i]
                    results[original_index] = res

            except Exception as e:
                print(f"Error during batch Stockfish move: {e}")
            
            return results

    def move(self, board: chess.Board, return_perplexity: bool=False) -> Tuple[str, float]:
        if self.type == "stockfish":
            return self._stockfish_move(board, return_perplexity)
        elif self.type == "chessformer":
            return self._chessformer_move(board, return_perplexity)
        else:
            raise ValueError(f"Invalid engine type: {self.type}")
        
    def batch_move(self, boards: List[chess.Board]) -> List[Tuple[str, float]]:
        if self.type == "stockfish":
            return self._batch_stockfish_move(boards)
        elif self.type == "chessformer":
            return self._batch_chessformer_move(boards)
        else:
            raise ValueError(f"Invalid engine type: {self.type}")
        
    def analyze_position(self, board: chess.Board) -> Optional[float]:
        """
        Analyzes the given **single board** position using the engine.
        For Stockfish, returns list of centipawn scores from white's perspective;
        For ChessFormer, returns list of models's value estimates
        Returns None if analysis failed.
        """  
        if self.type == "stockfish":
            try:
                engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                info = engine.analyse(board,chess.engine.Limit(depth=self.depth))
                engine.quit()
            except Exception as e:
                print(f"Stockfish error: {e}")
                return None
            
            score_obj = info.get("score")
            pov_score = score_obj.pov(chess.WHITE)
            cp = None
            if pov_score.is_mate():
                mate_score = pov_score.mate()
                cp = 10000.0 if mate_score > 0 else -10000.0
                relative_score = score_obj.relative
                if relative_score.is_mate():
                    cp = 10000.0 if relative_score.mate() > 0 else -10000.0
                else:
                    if relative_score.cp is not None:
                        cp = float(relative_score.cp)
                    else:
                        return None
            elif pov_score.cp is not None:
                relative_score = score_obj.relative
                if relative_score.cp is not None:
                    cp = float(relative_score.cp)
                else:
                    return None
            else:
                return None

            if cp is not None:
                normalized_score = 2 / (1+math.exp(-0.004*cp)) - 1
                return normalized_score if board.turn == chess.WHITE else -normalized_score
            else:
                return None
            
        
        elif self.type == "chessformer":
            fen = board.fen()
            count_tensor = self.compute_repetition([board.copy(stack=True)])

            with torch.no_grad():
                _, value = self.model([fen],count_tensor)

            value = value.item()
            return value if board.turn == chess.WHITE else -value
    
        else:
            raise ValueError(f"Invalid engine type.")


def test_search_enhanced_move(model_path,device):
    """Test the search-enhanced move functionality"""
    print("\n--- Testing Search-Enhanced ChessFormer ---")
    
    import sys
    sys.path.append("./")
    try:
        from model import ChessFormerModel
    except ImportError:
        from model import ChessFormerModel
    
    # Load the trained model
    checkpoint = torch.load(model_path,map_location=device)
    config = checkpoint["config"]
    model = ChessFormerModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)
    
    # Test different search configurations
    test_configs = [
        #{"depth": 0, "top_k": 8, "decay_rate": 0.6, "temperature": 0.2},  # No search (baseline)
        #{"depth": 1, "top_k": 8, "decay_rate": 0.6, "temperature": 0.2},  # Shallow search
        {"depth": 8, "top_k": 8, "decay_rate": 0.5, "temperature": 0.5},  # Medium search
    ]
    
    # Test positions
    test_positions = [
        "r3k1nr/pp3ppp/1qn1p3/2bpP3/5B2/2P2N1P/PP3PP1/R2QKB1R w KQkq - 1 11", # blunder: d1b3, best: d1c2
        "r3kbnr/1p3ppp/2n1p3/pBPpq3/1P6/2P1BQ2/P4PPP/RN2K2R w KQkq - 0 11", # blunder: b1d2, best: a5a4
        "r3kbnr/1p3ppp/2n1p3/1BPp4/1p6/2q1BQ2/P2N1PPP/R4RK1 w kq - 0 13", # blunder: a1c1, best: f3f4
        "4kb1r/4nppp/2p1p3/2Pp1q2/rp6/4BN2/P3QPPP/1R3RK1 w k - 6 18", # blunder: a2a3, best: f3d4
        "4k2r/4npbp/2p1p3/2Pp1qp1/1P6/4BN2/4QPPP/5RK1 w k - 1 21", # blunder: e3d4, best: e3g5
        "5rk1/4npb1/2p1p3/2PpNq1p/1P1B2p1/8/4QPPP/4R1K1 w - - 0 24", # blunder: f2f4, best: b4b5
        "5rk1/4npb1/2p1p3/2PpN2p/1P1B1qp1/8/4Q1PP/4R1K1 w - - 0 25", # blunder: e5f3, best: d4c3
        "rnr3k1/3bqppp/1p1bpn2/p7/8/1N1QPN1P/PP2BPP1/R1BR2K1 b - - 3 15", # blunder: d7c6, best: d6c7
        "2kr1b1r/1p3ppp/2n1p3/1BP2q2/PP2pB2/6Q1/5PPP/R2K3R w - - 1 17", # blunder: d1c1
        "rnbqkb1r/1p3ppp/4pn2/2Pp4/1P6/4PN2/P4PPP/RNBQKB1R b KQkq - 0 7", # blunder: b8c6
        "2kr3r/pp2nppp/4p3/1Bb1q3/8/2P2Q2/PP1N1PPP/R3K2R w KQ - 0 14", # blunder: d2e4
        "rn1qkb1r/3b1ppp/4p3/1P1p4/P1p1n3/4P3/3NBPPP/RNBQK2R b KQkq - 1 12", # blunder: e4d6
        "r3k1nr/ppq2ppp/2n1p3/2b1P3/8/2P2Q2/PP3PPP/RNB1KB1R w KQkq - 0 10", # blunder: b2b4
        "rn1qkb1r/3B1ppp/1p2pn2/p1Pp4/1P6/2P1PN2/P4PPP/RNBQK2R b KQkq - 0 8", # blunder: b8d7
        "r4rk1/1q2bppp/1p2p3/3p4/P7/2B1PN2/5PPP/1R1Q1RK1 b - - 0 17", # blunder: a8a4
    ]

    test_positions = [
        "r3kbnr/ppq1pppp/2n5/2PpP3/6b1/2P2N2/PP3PPP/RNBQKB1R w KQkq - 1 7",
        "r3kbnr/pp3ppp/2n1p3/1BPpq3/1P6/2P2Q2/P4PPP/RNB1K2R w KQkq - 0 10",
        "r3kb1r/1p3ppp/2n1p3/pBPp1q2/PP2nB2/2P2Q2/3N1PPP/R2K3R w kq - 4 14",
        "2kr1b1r/1p3ppp/2n1p3/1BP2q2/PP2pB2/6Q1/5PPP/R2K3R w - - 1 17",
        "2kr1b1r/1p3ppp/4p3/1BP2q2/Pn2pB2/6Q1/5PPP/R1K4R w - - 0 18",
        "rnbqkb1r/1p3ppp/4pn2/2Pp4/1P6/4PN2/P4PPP/RNBQKB1R b KQkq - 0 7",
        "r1bq1rk1/1p2bppp/2n1pn2/2Pp4/PP6/1Q2PN2/3N1PPP/R1B1KB1R b KQ - 2 10",
        "r1bq1rk1/1p2bppp/2n2n2/2Ppp3/PP6/1Q2PN2/1B1N1PPP/R3KB1R b KQ - 1 11",
        "r3kbnr/ppq2ppp/2n1p3/2P1P3/4Q3/2P2b2/PP3PPP/RNB1KB1R w KQkq - 0 9",
        "2kr3r/pp2nppp/4p3/1Bb1q3/8/2P2Q2/PP1N1PPP/R3K2R w KQ - 0 14",
        "2kr3r/pp2n1pp/4p3/1Bb1qp2/4N3/2P2Q2/PP3PPP/R3K2R w KQ - 0 15",
        "7r/ppk1n1pp/1b1rB3/8/1P6/2P1P3/P5PP/R4RK1 w - - 1 22",
        "7r/ppk1nRpp/4r3/8/1P6/2P1b3/P5PP/R6K w - - 0 24",
        "rn1qkb1r/3b1ppp/4pn2/1Ppp4/8/4PN2/P3BPPP/RNBQK2R b KQkq - 0 10",
        "rn1q1rk1/3bbppp/3np3/1P1p4/P1p1P3/8/1B1NBPPP/RN1Q1RK1 b - - 2 15",
        "rn1qkb1r/3b1ppp/4p3/1P1p4/P1p1n3/4P3/3NBPPP/RNBQK2R b KQkq - 1 12",
        "rn1q1rk1/3b1ppp/3npb2/1P1pP3/P1p5/8/1B1NBPPP/RN1Q1RK1 b - - 0 16",
        "rn1q1rk1/3b1ppp/3np3/1P1pB3/P1p5/8/3NBPPP/RN1Q1RK1 b - - 0 17",
        "rn1q2k1/5r1p/5pp1/1P1ppb2/P1p5/B1N2B2/3N1PPP/R2Q1RK1 b - - 3 22",
        "r2q2k1/5r1p/5p2/PP2nbp1/2p1p3/2NpB3/4BPPP/RN1Q1RK1 b - - 1 28",
        "r3kbnr/ppq2ppp/2n1p3/2P1P3/4Q3/2P2b2/PP3PPP/RNB1KB1R w KQkq - 0 9",
        "r3k1nr/ppq2ppp/2n1p3/2b1P3/8/2P2Q2/PP3PPP/RNB1KB1R w KQkq - 0 10",
        "r3k1nr/ppq2ppp/2n1p3/4P3/1b6/2P2Q2/P4PPP/RNB1KB1R w KQkq - 0 11",
        "r3k1nr/pp3ppp/2n1p3/4q3/1P6/5Q2/P4PPP/RNB1KB1R w KQkq - 0 12",
        "3rk1nr/pp3ppp/2n1p3/4q3/1P6/5Q2/P4PPP/RNBK1B1R w k - 2 13",
        "3rk2r/pp2nppp/2n1p3/8/1P6/3B1Q2/P2B1PPP/qN1K3R w k - 2 15",
        "rn1qkb1r/3B1ppp/1p2pn2/p1Pp4/1P6/2P1PN2/P4PPP/RNBQK2R b KQkq - 0 8",
        "r2qkb1r/3n1ppp/1pP1pn2/p2p4/1P6/2P1PN2/P4PPP/RNBQK2R b KQkq - 0 9",
        "r4rk1/1q2bppp/1p2p3/3p4/P7/2B1PN2/5PPP/1R1Q1RK1 b - - 0 17",
        "2r3k1/4b1p1/5p1p/3pp3/8/1Q2PN2/5PPP/B4RK1 b - - 0 26",
    ]

    test_boards = [chess.Board(p) for p in test_positions]

    for i, cfg in enumerate(test_configs):
        print(f"\n--- Test Configuration {i+1}: Depth={cfg['depth']}, Top-K={cfg['top_k']}, Decay={cfg['decay_rate']}, Temp={cfg['temperature']} ---")
        chessformer_config = ChessformerConfig(
            chessformer=model,
            device=device,
            temperature=cfg['temperature'],
            depth=cfg['depth'],
            top_k=cfg['top_k'],
            decay_rate=cfg['decay_rate']
        )
        engine = Engine(type="chessformer", chessformer_config=chessformer_config)

        for j, board in enumerate(test_positions):
            print(f"\n--- Analyzing Position {j+1}: {board.fen()} ---")
            try:
                move, value, perplexity = engine._chessformer_move(board, return_perplexity=True, verbose=True)
                print(f"Selected Move: {move}, Predicted Value (White's POV): {value:.4f}, Perplexity: {perplexity:.4f}")
            except Exception as e:
                print(f"Error analyzing position {board.fen()}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    model_path = "./ckpts/chessformer-sl_06.pth"
    device = torch.device("cpu")
    test_search_enhanced_move(model_path,device)

    