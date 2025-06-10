"""
Evaluate ChessFormerModel's different checkpoints on several metrics:
- loss on kaupane/lichess-2023-01-stockfish-annotated dataset's depth27 split
- stockfish (Stockfish 17 depth 24) analyzed game quality + move annotation (best/excellent/good/inaccuracy/mistake/blunder)
"""

import torch
import chess
from tqdm.auto import tqdm
import math
import multiprocessing
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader
import huggingface_hub

from model import ChessFormerModel
from utils import Engine, UCI_MOVE_TO_IDX, IDX_TO_UCI_MOVE, ChessformerConfig, StockfishConfig


def load_model(checkpoint_path: str, device: torch.device) -> ChessFormerModel:
    print(f"Loading model from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path,map_location=device)
        try:
            config = checkpoint.get('config',{})
        except Exception as e:
            config = {
                "num_blocks": 20,
                "hidden_size": 640,
                "intermediate_size": 1728,
                "num_heads": 8,
                "dropout": 0.00,
                "possible_moves": 1969
            }
        model = ChessFormerModel(
            num_blocks=config.get('num_blocks'),
            hidden_size=config.get("hidden_size"),
            intermediate_size=config.get("intermediate_size"),
            num_heads=config.get("num_heads"),
            dropout=config.get("dropout"),
            possible_moves=config.get("possible_moves")
        )
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model
    except FileNotFoundError:
        model = ChessFormerModel.from_pretrained(checkpoint_path)
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def evaluate_loss(model: ChessFormerModel, dataset_name: str, dataset_split: str, batch_size: int, device: torch.device) -> Dict[str,float]:
    """Calculate loss on a validation dataset"""

    # Prepare dataloader and progress bar
    dataset = load_dataset(dataset_name,split=dataset_split)
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    pbar = tqdm(enumerate(dataloader),
                total=len(dataloader),
                desc=f"Validation")
    
    # Main loop
    model = model.to(device)
    model.eval()
    total_act_loss = 0.0
    total_val_loss = 0.0
    total_inv_loss = 0.0
    total_loss = 0.0
    
    with torch.no_grad(), multiprocessing.Pool(processes=12) as pool:
        for idx, sample in pbar:
            fens = sample["fen"]
            repetition_counts = sample["repetition_count"].to(device)
            best_moves_uci = sample["best_move"]
            scores = sample["score"].to(device)
            valid_moves_str_list = sample["valid_moves"]
            batch_size = len(fens)

            try:
                best_moves_indices = [UCI_MOVE_TO_IDX[move] for move in best_moves_uci]
            except KeyError as e:
                print(f"Error: Move '{e}' not found in UCI_MOVE_TO_IDX")
                continue
            best_moves_tensor = torch.tensor(best_moves_indices, dtype=torch.long).to(device)

            invalid_move_mask = torch.ones((batch_size,1969),device=device,dtype=torch.float32)
            for i in range(batch_size):
                valid_uci_moves = valid_moves_str_list[i].split(' ')
                try:
                    valid_indices = [UCI_MOVE_TO_IDX[move] for move in valid_uci_moves]
                    if valid_indices:
                        invalid_move_mask[i,valid_indices] = 0.0
                except Exception as e:
                    raise
            
            # Compute losses
            actions, values = model(fens,repetition_counts)
            act_loss = torch.nn.functional.cross_entropy(actions,best_moves_tensor)
            val_loss = torch.nn.functional.mse_loss(values,scores)
            probs = torch.softmax(actions,dim=-1)
            invalid_probs_sum = (probs*invalid_move_mask).sum(dim=-1)
            inv_loss = invalid_probs_sum.mean()


            total_act_loss += act_loss.item()
            total_val_loss += val_loss.item()
            total_inv_loss += inv_loss.item()



            pbar.set_postfix({
                "ActLoss": f"{total_act_loss/(idx+1):.4f}",
                "ValLoss": f"{total_val_loss/(idx+1):.4f}",
                "InvLoss": f"{total_inv_loss/(idx+1):.4f}"
            })

    avg_act_loss = total_act_loss / len(dataloader)
    avg_val_loss = total_val_loss / len(dataloader)
    avg_inv_loss = total_inv_loss / len(dataloader)

    results = {
        "act_loss": avg_act_loss, 
        "val_loss": avg_val_loss, 
        "invalid_move_loss": avg_inv_loss
    }

    return results

def compare_checkpoints(checkpoint_path_list: List[str], device, batch_size=512) -> Dict[str,str]:
    """Compare checkpoints based on evaluation loss"""
    dataset_name = "kaupane/lichess-2023-01-stockfish-annotated"
    dataset_split = "depth27"
    best_act_checkpoint = None
    best_val_checkpoint = None
    best_inv_checkpoint = None
    best_act_loss = math.inf
    best_val_loss = math.inf
    best_inv_loss = math.inf
    for checkpoint in checkpoint_path_list:
        print(f"Start evaluating {checkpoint}")
        model = load_model(checkpoint,device)
        result = evaluate_loss(model,dataset_name,dataset_split,batch_size=batch_size,device=device)
        if result["act_loss"] < best_act_loss:
            best_act_loss = result["act_loss"]
            best_act_checkpoint = checkpoint
        if result["val_loss"] < best_val_loss:
            best_val_loss = result["val_loss"]
            best_val_checkpoint = checkpoint
        if result["invalid_move_loss"] < best_inv_loss:
            best_inv_loss = result["invalid_move_loss"]
            best_inv_checkpoint = checkpoint
    print(f"Best act loss {best_act_loss} from {best_act_checkpoint}")
    print(f"Best val loss {best_val_loss} from {best_val_checkpoint}")
    print(f"Best inv loss {best_inv_loss} from {best_inv_checkpoint}")
    return {
        "best_act_model": best_act_checkpoint,
        "best_val_model": best_val_checkpoint,
        "best_inv_model": best_inv_checkpoint
    }
        
def play_games(engine1: Engine, engine2: Engine, max_moves: int=200, num_games: int=120) -> List[Tuple[Optional[int],str,List[str]]]:
    """
    Plays num_games games between two engines.
    Returns: List(result, termination_reason, move_list)
        - result: 1 for engine1 win, 0 for draw, -1 for engine2 win, None for error
        - termination_reason: 
            - max_moves_exceeded for over max_moves, result = None; 
            - other ordinary reasons including checkmate, stalemate, etc.
            - will not involve invalid moves because Engine class already filters out invalid moves
    """
    boards = [chess.Board() for _ in range(num_games)]
    move_lists = [[] for _ in range(num_games)]
    results = [(None, "unfinished", []) for _ in range(num_games)]
    active_game_indices = list(range(num_games))
    move_count = 0

    pbar = tqdm(total=num_games, desc=f"Playing {engine1.type} vs {engine2.type}")

    while active_game_indices and move_count < max_moves * 2:
        # One engine plays white for all games and another engine plays black for all games
        # So in evaluate_win_rate() function will switch engine1 and engine2 and call play_games() twice for fairness
        current_player_engine = engine1 if boards[active_game_indices[0]].turn == chess.WHITE else engine2

        active_boards = [boards[i] for i in active_game_indices]

        # Play move
        try:
            batch_move_results = current_player_engine.batch_move(active_boards)
        except Exception as e:
            print(f"Error during {current_player_engine.type} batch_move: {e}")
            raise

        next_active_indices = []
        processed_indices_this_turn = set()

        for idx, original_game_index in enumerate(active_game_indices):
            if original_game_index in processed_indices_this_turn:
                continue

            board = boards[original_game_index]
            move_info = batch_move_results[idx]

            if move_info is None:
                # Engine might return None if game is already over or error occurrd
                if results[original_game_index][0] is None and not board.is_game_over(claim_draw=True):
                    results[original_game_index] = (None, f"{current_player_engine.type}_move_error", move_lists[original_game_index])
                    pbar.update(1)
                processed_indices_this_turn.add(original_game_index)
                continue

            move_uci, _ = move_info

            if move_uci == "<claim_draw>":
                if board.can_claim_draw():
                    board.push(chess.Move.null())
                    move_lists[original_game_index].append("<claim_draw>")
                    results[original_game_index] = (0, "draw_by_claim", move_lists[original_game_index])
                    pbar.update(1)
                    processed_indices_this_turn.add(original_game_index)
                    continue # game ended
                else:
                    # Should't happen. Indicate something's wrong with engine calss
                    print(f"Error: Game {original_game_index} - Invalid draw claimed by {current_player_engine.type}. FEN: {boards[original_game_index].fen()}")
                    raise

            move = board.parse_uci(move_uci)
            if move not in board.legal_moves:
                # Shoudn't happen. Indicate something's wrong with the engine class
                print(f"Error: Game {original_game_index} - Invalid move made by {current_player_engine.type}. FEN: {boards[original_game_index].fen()}")
                raise
            board.push(move)
            move_lists[original_game_index].append(move_uci)

            # Check for game termination after the move
            outcome = board.outcome(claim_draw=True)
            if outcome:
                winner = outcome.winner
                result_code = None
                if winner == chess.WHITE: result_code = 1
                elif winner == chess.BLACK: result_code = -1
                elif winner is None: result_code = 0

                termination_reason = outcome.termination.name.lower()
                results[original_game_index] = (result_code,termination_reason,move_lists[original_game_index])
                pbar.update(1)
                processed_indices_this_turn.add(original_game_index)
            else:
                # Game continues, add to next_active_indices
                next_active_indices.append(original_game_index)
                processed_indices_this_turn.add(original_game_index)

        active_game_indices = next_active_indices
        move_count += 1

    pbar.close()

    for i in range(num_games):
        if results[i][0] is None and results[i][1] == "unfinished":
            results[i] = (None, "max_moves_exceeded", move_lists[i])

    return results

def evaluate_win_rate(chessformer_engine, stockfish_path: str, depths: List[int],
                      num_games: int) -> Dict[str, Dict[str, float]]:
    """
    Evaluates win rate against Stockfish at various depths.
    Returns: {depth, summary}
        - summary: wins, losses, draws, errors, total_played, win_rate, loss_rate, draw_rate, error_rate
        - 'wins' refers to ChessFormer wins.
    """
    results_per_depth = {}
    
    for depth in depths:
        print(f"\n --- Evaluating against Stockfish-17 Depth {depth} ---")
        stockfish_engine = Engine(type="stockfish",engine_path=stockfish_path,depth=depth)

        num_games_per_color = num_games // 2
        if num_games % 2 != 0:
            print(f"Warning: num_games should be even, but is odd: ({num_games})")

        all_results = []

        # Play games with ChessFormer as White
        results_white = play_games(engine1=chessformer_engine,engine2=stockfish_engine,
                                  max_moves=200,num_games=num_games_per_color)
        all_results.extend(results_white)
        # Play games with ChessFormer as Black
        results_black = play_games(engine1=stockfish_engine,engine2=chessformer_engine,
                                   max_moves=200,num_games=num_games_per_color)
        adjusted_results_black = []
        for res_code, reason, moves in results_black:
            # Adjust results from Black's perspective
            if res_code == 1: new_code = -1
            elif res_code == -1: new_code = 1
            else: new_code = res_code # why not new_code = -res_code? There might be None
            adjusted_results_black.append((new_code,reason,moves))
        all_results.extend(adjusted_results_black)

        wins = 0
        losses = 0
        draws = 0
        errors = 0
        for res_code, reason, _ in all_results:
            if res_code == 1:
                wins += 1
            elif res_code == -1:
                losses += 1
            elif res_code == 0:
                draws += 1
            else:
                errors += 1
                print(f"Game Error/Unfinished: Reason - {reason}")

        total_played = len(all_results)
        assert total_played == num_games and total_played > 0

        summary = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "errors": errors,
            "total_played": len(all_results),
            "win_rate": wins/total_played,
            "loss_rate": losses/total_played,
            "draw_rate": draws/total_played,
            "error_rate": errors/total_played
        }
        results_per_depth[depth] = summary
        print(f"Depth {depth} Summary: {summary}")

    return results_per_depth

def _classify_delta_e(delta_e: float) -> str:
    # Delta E = E_before - E_after
    # delta_e is always positive, meaning the expected score decreases after the move
    if delta_e <= 0.00: return "best" # but delta_e < 0 should not happen
    elif delta_e <= 0.02: return "excellent"
    elif delta_e <= 0.05: return "good"
    elif delta_e <= 0.10: return "inaccuracy"
    elif delta_e <= 0.20: return "mistake"
    else:                 return "blunder" 
        
def analyze_game_quality(chessformer_engine: Engine, stockfish_path: str, num_games: int,
                         opponent_depth: int, analysis_depth: int, max_moves_per_game: int=200) -> Dict[str, float]:
    """
    Analyzes the quality of moves made by ChessFormer against a Stockfish opponent.
    Plays games (<5 recommended), while analyzing ChessFormer's moves using a strong Stockfish engine.

    Should be interactive: for each position and move made, print position score and move analysis.

    Returns: Dictionary with classification rates and average Delta E.
             Keys: best_rate, excellent_rate, good_rate, inaccuracy_rate,
                   mistake_rate, blunder_rate, avg_delta_e, analysis_errors, total_moves_analyzed
    """
    opponent_engine = Engine(type="stockfish",stockfish_config=StockfishConfig(stockfish_path,opponent_depth))
    analyzer_engine = Engine(type="stockfish",stockfish_config=StockfishConfig(stockfish_path,analysis_depth))

    classification_counts = {
        "best": 0, "excellent": 0, "good": 0,
        "inaccuracy": 0, "mistake": 0, "blunder": 0
    }
    total_delta_e = 0.0
    total_moves_analyzed = 0
    analysis_errors = 0

    for game_idx in range(num_games):
        board = chess.Board()
        # Alternate colors: ChessFormer plays white in even games
        chessformer_is_white = (game_idx % 2 == 0)
        print(f"\nStarting Game {game_idx+1}/{num_games} (ChessFormer plays {'White' if chessformer_is_white else 'Black'}).")

        move_count = 0
        while not board.is_game_over(claim_draw=True) and move_count < max_moves_per_game * 2:
            is_chessformer_turn = (board.turn == chess.WHITE and chessformer_is_white) or \
                                  (board.turn == chess.BLACK and not chessformer_is_white)
            
            current_player_engine = chessformer_engine if is_chessformer_turn else opponent_engine
            player_name = "ChessFormer" if is_chessformer_turn else "Stockfish"

            # Analyze pre-move position if it's ChessFormer's turn
            score_before_pov = None
            if is_chessformer_turn:
                try:
                    # From analyzer (white)'s perspective
                    score_before_analyzer = analyzer_engine.analyze_position(board.copy(stack=True))
                    if not chessformer_is_white:
                        score_before_pov = - score_before_analyzer
                    else:
                        score_before_pov = score_before_analyzer
                    if score_before_pov is None:
                        print(f"Warning: Analysis failed for position before move {move_count // 2 + 1}")
                        analysis_errors += 1
                    else:
                        print(f"Pre-move Analysis Result: {score_before_pov} for FEN: {board.fen()}")
                except Exception as e:
                    print(f"Error during pre-move analysis: {e}")
                    analysis_errors += 1
                    score_before_pov = None

            # Move generation
            move_uci,_, perplexity = current_player_engine.move(board, return_perplexity=True)
            # Print opponent's move
            if not is_chessformer_turn:
                print(f"  Opponent's move: {move_uci}")

            # Analyze post-move position if it's ChessFormer's turn
            if is_chessformer_turn and score_before_pov is not None:
                # Handle special action of draw claim
                if move_uci == "<claim_draw>":
                    if board.can_claim_draw():
                        delta_e = score_before_pov # score_after is 0.0 for draw
                        classification = _classify_delta_e(delta_e)
                        classification_counts[classification] += 1
                        total_delta_e += delta_e
                        total_moves_analyzed += 1

                        print(f"  Move {move_count // 2 + 1} ({'W' if board.turn == chess.WHITE else 'B'}): <claim_draw> with perplexity {perplexity:.4f}")
                        print(f"    Score Before: {score_before_pov:+.3f}")
                        print(f"    Score After:  {0.0:+.3f}")
                        print(f"    Delta E:      {delta_e:+.3f} ({classification})")
                    else:
                        # Should not happen with Engine class filtering, but check anyway
                        print(f"Warning: ChessFormer proposed illegal move: <claim_draw> for FEN: {board.fen()}")
                        analysis_errors += 1
                # Not draw claim, should be normal uci move
                else:
                    move = board.parse_uci(move_uci)
                    if move in board.legal_moves:
                        board_after_move = board.copy(stack=True)
                        board_after_move.push(move)

                        # This if from the analyzer(white)'s perspective
                        score_after_analyzer = analyzer_engine.analyze_position(board_after_move)

                        if score_after_analyzer is not None:
                            if not chessformer_is_white:
                                score_after_pov = -score_after_analyzer
                            else:
                                score_after_pov = score_after_analyzer
                            delta_e = score_before_pov - score_after_pov
                            classification = _classify_delta_e(delta_e)
                            classification_counts[classification] += 1
                            total_delta_e += delta_e
                            total_moves_analyzed += 1

                            print(f"  Move {move_count // 2 + 1} ({'W' if board.turn == chess.WHITE else 'B'}): {move_uci} with perplexity {perplexity:.4f}")
                            print(f"    Score Before: {score_before_pov:+.3f}")
                            print(f"    Score After:  {score_after_pov:+.3f}")
                            print(f"    Delta E:      {delta_e:+.3f} ({classification})")
                        else:
                            # Should not happen with Engine class filtering, but check anyway
                            print(f"Warning: ChessFormer proposed illegal move: {move_uci} for FEN: {board.fen()}")
                            analysis_errors += 1
                
            # Apply move to board
            if move_uci == "<claim_draw>":
                board.push(chess.Move.null())
                print(f"Game {game_idx+1}: {player_name} claimed draw.")
            elif move_uci is not None:
                move = board.parse_uci(move_uci)
                board.push(move)

            move_count += 1


        # Game End
        outcome = board.outcome(claim_draw=True)
        if outcome:
            white_player = "ChessFormer" if chessformer_is_white else "Stockfish"
            black_player = "Stockfish" if chessformer_is_white else "ChessFormer"
            print(f"Game {game_idx+1} finished: {outcome.termination.name} - Result: {white_player} {outcome.result()} {black_player}")
        elif move_count >= max_moves_per_game * 2:
            print(f"Game {game_idx+1} finished: Max moves exceeded.")
        else:
            print(f"Game {game_idx+1} finished: Unknown reason")

    # Aggregate results
    if total_moves_analyzed > 0:
        results = {
            f"{cls}_rate": count / total_moves_analyzed
            for cls, count in classification_counts.items()
        }
        results["avg_delta_e"] = total_delta_e / total_moves_analyzed
    else:
        results = {
            f"{cls}_rate": 0.0 for cls in classification_counts.keys()
        }
        results["avg_delta_e"] = 0.0

    results["analysis_errors"] = analysis_errors
    results["total_moves_analyzed"] = total_moves_analyzed

    print("\n--- Overall Analysis Summary ---")
    for key, value in results.items():
        if "_rate" in key:
            print(f"  {key:<18}: {value:.2%}")
        else:
            print(f"  {key:<18}: {value:.4f}" if isinstance(value, float) else f"  {key:<18}: {value}")

    return results


                    

                    
                


def eval_loss(model_path,device):
    """
    Will first test all model checkpoints on the loss/invalid_moves_rate,
    Since it would make no sense to evaluate models that can't even make valid moves any further
    """
    dataset_name = "kaupane/lichess-2023-01-stockfish-annotated"
    #dataset_split = "depth18[:65536]"
    dataset_split = "depth27[32768:]"
    model = load_model(model_path,device=device)
    batch_size = 4
    results = evaluate_loss(model, dataset_name, dataset_split, batch_size, device)

def main(model_path,device):
    """
    Continue to test selected models on win rate & game quality.
    """
    chessformer_model = load_model(model_path,device=device)
    config = ChessformerConfig(
        chessformer=chessformer_model,
        device=device,
        temperature=0.5,
        depth=0,
        top_k=8,
        decay_rate=0.6,
        max_batch_size=864
    )
    chessformer_engine = Engine("chessformer", config)
    analyze_game_quality(
        chessformer_engine=chessformer_engine,
        stockfish_path="/usr/games/stockfish",
        num_games=4,
        opponent_depth=0,
        analysis_depth=24,
    )


if __name__ == "__main__":
    #model_path = "./ckpts/chessformer-sl_10.pth"
    model_path = "kaupane/ChessFormer-RL"
    device = torch.device("cpu")
    eval_loss(model_path,device)
    #main(model_path,device)