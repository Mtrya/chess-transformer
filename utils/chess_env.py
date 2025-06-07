"""Provide a gym-like environment for clarity"""

import chess
import torch
import time
from typing import List, Tuple, Dict
try:
    from .mapping import IDX_TO_UCI_MOVE, UCI_MOVE_TO_IDX
except:
    from mapping import IDX_TO_UCI_MOVE, UCI_MOVE_TO_IDX

class BatchChessEnv:
    """A single chess environment with sparse terminal reward"""
    def __init__(self, batch_size: int, max_moves: int=200):
        self.batch_size = batch_size
        self.max_moves = max_moves
        self.reset()

    def reset(self) -> Tuple[List[str], torch.Tensor]:
        """
        Starts all games from the initial position
        Returns:
            fens (List[str]), repetition_counts (torch.Tensor of shape [batch_size,])
        """
        self.boards = [chess.Board() for _ in range(self.batch_size)]
        self.move_counts = [0] * self.batch_size
        self.done_flags = [False] * self.batch_size

        fens = [self.boards[0].fen()] * self.batch_size
        reps = torch.ones(self.batch_size,dtype=torch.long)
        return fens, reps # (bs,)
    
    def _compute_rep(self, board: chess.Board) -> int:
        board_copy = board.copy()
        trasposition_key = board_copy._transposition_key()
        count = 0
        while board_copy.move_stack:
            board_copy.pop()
            if board_copy._transposition_key() == trasposition_key:
                count += 1
        return count + 1 # 1 for fresh position
    
    def step(self, uci_moves: List[str]) -> Tuple[List[str],    # next fens (next state)
                                                  torch.Tensor, # next reps (next state)
                                                  List[bool],   # dones
                                                  List[Dict]]:  # infos
        """
        Apply one move per game in the batch.
        Args:
            uci_moves: list of UCI strings (plus "<claim_draw>")
        Returns:
            next_fens: new FENs for each game,                          List[str]
            reps: repetition counts,                                    Tensor[batch_size]
            dones: whether this game is now terminated,                 List[bool]
            infos: info dict with 'result' key for completed games      List[dict]
        """
        next_fens, reps, dones, infos = [], [], [], []

        for i, move in enumerate(uci_moves):
            board = self.boards[i]
            info = {
                "max_steps_exceeded": False,
                "truncation_due_to_error": False,
                "result": None
            }
            done = self.done_flags[i]

            if done:
                # Game already done, pass through the existing state
                next_fens.append(board.fen())
                reps.append(1)
                dones.append(True)
                infos.append(info)
                continue
            
            if move == "0000":
                # Skip through dummy moves
                next_fens.append(board.fen())
                reps.append(1)
                dones.append(True)
                infos.append(info)
                continue

            if board.is_game_over():
                # Game already over
                done = True
                info["result"] = board.result()
                next_fens.append(board.fen())
                reps.append(self._compute_rep(board))
                dones.append(done)
                infos.append(info)
                continue

            try:
                if move == "<claim_draw>":
                    if board.can_claim_draw():
                        done = True
                        info['result'] = "1/2-1/2"
                    else:
                        raise ValueError(f"Invalid move ('<claim_draw>') passed in.")
                else:
                    try:
                        m = chess.Move.from_uci(move)
                        if m in board.legal_moves:
                            board.push(m)
                            self.move_counts[i] += 1

                            if board.is_game_over():
                                done = True
                                info['result'] = board.result()
                        else:
                            raise ValueError(f"Invalid move ('{m.uci()}') passed in.")
                    except Exception as e:
                        done = True
                        info['truncation_due_to_error'] = True
                        print(f"Unexpected error: {e}")

                if self.move_counts[i] >= self.max_moves:
                    done = True
                    info['max_steps_exceeded'] = True
                    info['result'] = "1/2-1/2"

                next_fens.append(board.fen())
                reps.append(self._compute_rep(board))
                dones.append(done)
                infos.append(info)

            except Exception as e:
                print(f"Error processing move {move} for board {i}: {e}")
                done = True
                info["truncation_due_to_error"] = True
                next_fens.append(board.fen())
                reps.append(self._compute_rep(board))
                dones.append(done)
                infos.append(info)

            self.done_flags[i] = done
        
        reps = torch.tensor(reps,dtype=torch.long) # [bs,]
        return next_fens, reps, dones, infos

if __name__ == "__main__":
    env = BatchChessEnv(1)
    env.reset()
    board = env.boards[0]
    board.push(chess.Move.from_uci("e2e4"))
    new_board = board.copy()
    rep = env._compute_rep(new_board)
    print(rep)


