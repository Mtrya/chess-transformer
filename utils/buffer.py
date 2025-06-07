
import torch
from collections import deque
import numpy as np
from typing import List, Iterator, Tuple, Optional
import chess

class Game:
    """
    Represents a single chess game trajectory with all relevant data for RL training.
    Acts as a *temporary* buffer inside loop
    Handles:
        - Storing trajectory data (fens, reps, actions, log_probs, values, invalid_masks)
        - Tracking game status (active/complete)
    """
    def __init__(self):
        self.active = True
        self.valid = True
        self.completion_reason = None
        self.game_result = None

        self.fens = []
        self.repetition_counts = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.invalid_masks = []

    def update_trajectory(self, fen, rep, act, val, logp, inv_m):
        self.fens.append(fen)
        self.repetition_counts.append(rep)
        self.actions.append(act)
        self.values.append(val)
        self.log_probs.append(logp)
        self.invalid_masks.append(inv_m)

    def update_game_status(self, done, reason):
        if done == True:
            self.active = False
            if reason in ["1-0","0-1","1/2-1/2"]:
                self.completion_reason = reason
                self.game_result = reason
            else:
                self.completion_reason = reason
                self.game_result = None
                self.valid = False
            
    def get_white_trajectory(self):
        """Extract the trajectory for white"""
        indices = []
        for i in range(len(self.fens) - 1):
            board = chess.Board(self.fens[i])
            if board.turn:  # True if white to move
                indices.append(i)
                
        return {
            'fens': [self.fens[i] for i in indices],
            'repetition_counts': [self.repetition_counts[i] for i in indices],
            'actions': [self.actions[i] for i in indices],
            'values': [self.values[i] for i in indices],
            'log_probs': [self.log_probs[i] for i in indices],
            'invalid_masks': [self.invalid_masks[i] for i in indices]
        }
    
    def get_black_trajectory(self):
        """Extract the trajectory for black pieces."""
        indices = []
        for i in range(len(self.fens) - 1):
            board = chess.Board(self.fens[i])
            if not board.turn:  # False if black to move
                indices.append(i)
                
        return {
            'fens': [self.fens[i] for i in indices],
            'repetition_counts': [self.repetition_counts[i] for i in indices],
            'actions': [self.actions[i] for i in indices],
            'values': [self.values[i] for i in indices],
            'log_probs': [self.log_probs[i] for i in indices],
            'invalid_masks': [self.invalid_masks[i] for i in indices]
        }
                




class ReplayBuffer:
    """
    The buffer class for PPO reinforcement learning.
    Handles:
        - store samples including:
            1. fens
            2. reps
            3. actions
            4. log_probs
            5. values
            6. invalid_masks
        - calculate advantage based on reward and value (7. advantage)
        - output samples in batches
    Since PPO is largely on-policy, so the replay buffer will not be so large that deque is not appropriate
    """
    def __init__(self,
                 capacity: int,
                 batch_size: int,
                 gamma: float,
                 gae_lambda: float,
                 shuffle: bool=True
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.fens = deque(maxlen=capacity)
        self.repetition_counts = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        self.invalid_masks = deque(maxlen=capacity)
        self.advantages = deque(maxlen=capacity)

        self.batch_size = batch_size
        self.shuffle = shuffle

    def push_game(self, game: Game):
        """
        Process a completed game and add its trajectories to the buffer.
        Handles reward computation for both white and black players.
        """
        if not game.valid:
            return
        
        result = game.game_result
        if result not in ["1-0","0-1","1/2-1/2"]:
            raise ValueError(f"Result not recognized: {result}. Either an incompleted game was passed in, or game.update_game_status() method is wrong.")
        
        if result == "1-0": result = 1
        elif result == "0-1": result = -1
        elif result == "1/2-1/2": result = 0

        white_traj = game.get_white_trajectory()
        if white_traj["fens"]:
            self._process_trajectory(
                white_traj["fens"],
                white_traj["repetition_counts"],
                white_traj["actions"],
                white_traj["log_probs"],
                white_traj["values"],
                white_traj["invalid_masks"],
                result
            )

        black_traj = game.get_black_trajectory()
        if black_traj["fens"]:
            self._process_trajectory(
                black_traj["fens"],
                black_traj["repetition_counts"],
                black_traj["actions"],
                black_traj["log_probs"],
                black_traj["values"],
                black_traj["invalid_masks"],
                -result # flip reward for black's perspective
            )

    def _process_trajectory(self, fens, reps, actions, log_probs, values, invalid_masks, final_reward):
        """Process a trajectory for one player, compute advantages and add to buffer"""
        values_tensor = torch.tensor(values) if not torch.is_tensor(values) else values

        advantages = self._compute_advantage(values_tensor, final_reward)

        for i in range(len(fens)):
            self.fens.append(fens[i])
            self.repetition_counts.append(reps[i])
            self.actions.append(actions[i])
            self.log_probs.append(log_probs[i])
            self.values.append(values[i])
            self.invalid_masks.append(invalid_masks[i])
            self.advantages.append(advantages[i])

    def _compute_advantage(self, value_traj: torch.Tensor, final_reward: float) -> torch.Tensor:
        """
        Calculate GAE with only a terminal reward: r_t = 0 for t < T-1 and r_{T-1} = final_reward
        Args:
            value_traj: value prediction of the model
            final_reward: game result
        
        Returns:
            advantage, torch.Tensor of shape same with value_traj
        """

        vals = value_traj.detach().cpu().float()
        T = vals.shape[0] if vals.dim() > 0 else 1

        adv = torch.zeros(T)
        next_value = 0.0
        gae = 0.0

        for t in reversed(range(T)):
            reward = final_reward if t == T-1 else 0.0
            delta = reward + self.gamma * next_value - vals[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            adv[t] = gae
            next_value = vals[t]

        return adv
    
    def sample(self) -> Iterator[Tuple[List[str],   # fen
                                       torch.Tensor,# rep
                                       torch.Tensor,# act
                                       torch.Tensor,# logp
                                       torch.Tensor,# val
                                       torch.Tensor,# inv_m
                                       torch.Tensor]]: # adv
        """Yield minibatches of size batch_size for training"""
        n = len(self.fens)
        if n < self.batch_size:
            return
        
        idxs = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idxs)
        
        for start in range(0, n, self.batch_size):
            batch_idx = idxs[start:start+self.batch_size]
            if len(batch_idx) < self.batch_size:
                break

            fens_b = [self.fens[i] for i in batch_idx]

            reps_b = torch.stack([
                self.repetition_counts[i].detach().clone() if torch.is_tensor(self.repetition_counts[i]) 
                else torch.tensor(self.repetition_counts[i]) 
                for i in batch_idx
            ])
            
            acts_b = torch.stack([
                self.actions[i].detach().clone() if torch.is_tensor(self.actions[i])
                else torch.tensor(self.actions[i])
                for i in batch_idx
            ])
            logps_b = torch.stack([
                self.log_probs[i].detach().clone() if torch.is_tensor(self.log_probs[i])
                else torch.tensor(self.log_probs[i])
                for i in batch_idx
            ])
            
            vals_b = torch.stack([
                self.values[i].detach().clone() if torch.is_tensor(self.values[i])
                else torch.tensor(self.values[i])
                for i in batch_idx
            ])
            
            advs_b = torch.stack([
                self.advantages[i].detach().clone() if torch.is_tensor(self.advantages[i])
                else torch.tensor(self.advantages[i])
                for i in batch_idx
            ])
            
            invs_b = torch.stack([
                self.invalid_masks[i] if torch.is_tensor(self.invalid_masks[i])
                else torch.tensor(self.invalid_masks[i])
                for i in batch_idx
            ])

            yield fens_b, reps_b, acts_b, logps_b, vals_b, invs_b, advs_b       

    def __len__(self) -> int:
        return len(self.fens)
    
    def clear(self) -> None:
        self.fens.clear()
        self.repetition_counts.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.invalid_masks.clear()
        self.advantages.clear()
