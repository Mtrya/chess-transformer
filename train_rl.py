import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical
import copy
import logging
import chess
import numpy as np
from typing import List, Tuple, Dict, Optional
import time

from utils import ReplayBuffer, BatchChessEnv, Game, UCI_MOVE_TO_IDX, IDX_TO_UCI_MOVE
from model import ChessFormerModel

import swanlab


class RLTrainer:
    """Uses Selfplay + PPO"""
    def __init__(self,
                 model: ChessFormerModel,
                 learning_rate: float,
                 value_ratio: float,
                 entropy_ratio: float,
                 invalid_pen_ratio: float,
                 clip_eps: float,
                 gamma: float,
                 gae_lambda: float,
                 k_epochs: int,
                 num_episodes: int,
                 update_batch_size: int,
                 accumulation_steps: int,
                 max_grad_norm: float,
                 rollout_batch_size: int,
                 save_every_episodes: int,
                 log_every_steps: int,
                 model_config: Dict,
                 experiment_name: Optional[str]=None):
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.model = model.to(self.device)
        self.model_old = copy.deepcopy(self.model)
        self.total_moves = self.model.possible_moves
        self.model_config = model_config

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"---Using device: {self.device}, Model params: {num_params/1e6}M---")

        self.env = BatchChessEnv(batch_size=rollout_batch_size)

        self.buffer = ReplayBuffer(
            capacity=rollout_batch_size*200,
            batch_size=update_batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda
        )

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.value_ratio = value_ratio
        self.entropy_ratio = entropy_ratio
        self.invalid_pen_ratio = invalid_pen_ratio

        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.num_episodes = num_episodes
        self.update_batch_size = update_batch_size
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.rollout_batch_size = rollout_batch_size

        self.save_every_episodes = save_every_episodes
        self.log_every_steps = log_every_steps

        self.start_episode = 0
        self.current_episode = self.start_episode

        self.scaler = GradScaler(self.device_str)

        self.policy_loss_accumulator = 0.0
        self.value_loss_accumulator = 0.0
        self.entropy_loss_accumulator = 0.0
        self.invalid_loss_accumulator = 0.0
        self.samples_since_log = 0
        self.grad_norm = 0.0

        self.global_steps = 0

        # logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('./log/rl_training.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        # swanlab
        swanlab.init(
            project="chessformer",
            experiment_name=experiment_name,
            config={
                "learning_rate": self.learning_rate,
                "value_ratio": self.value_ratio,
                "entropy_ratio": self.entropy_ratio,
                "invalid_pen_ratio": self.invalid_pen_ratio,
                "clip_eps": self.clip_eps,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "k_epochs": self.k_epochs,
                "update_batch_size": self.update_batch_size * self.accumulation_steps,
                "rollout_batch_size": self.rollout_batch_size
            },
            logdir="./log"
        )

    @torch.no_grad()
    def _make_action(self, fens: List[str], reps: torch.Tensor, games: List[Game]):
        """
        Generate actions for active games only.
        Args:
            fens: List of FEN representations
            reps: repetition counts, torch.Tensor of shape [bs,]
            games: List of Game objects
        Returns:
            updates games in-place with fen/rep/act/logp/val/inv
            returns uci_moves: List of uci moves.
        """
        batch_size = len(fens)
        uci_moves = ["0000"] * batch_size

        # Find active games
        active_indices = [i for i in range(batch_size) if games[i].active]

        if not active_indices:
            return uci_moves

        active_fens = [fens[i] for i in active_indices]
        active_reps = reps[active_indices] # [bs,]

        self.model_old.eval()
        active_reps = active_reps.to(self.device)
        logits, active_values = self.model_old(active_fens,active_reps)

        # Create action masks for invalid moves (per active game)
        action_mask = torch.full((len(active_indices),self.total_moves),-1e9,device=self.device)
        for i, orig_idx in enumerate(active_indices):
            board = chess.Board(fens[orig_idx])
            valid_moves_list = [m.uci() for m in board.legal_moves]
            valid_indices = [UCI_MOVE_TO_IDX[move] for move in valid_moves_list]

            if valid_indices:
                action_mask[i, valid_indices] = 0.0
            # Handle draw claims
            if board.can_claim_draw() or active_reps[i].item() >= 3:
                #print(f"Considering <claim_draw> to be valid. FEN: {board.fen()}, rep: {active_reps[i].item()}")
                action_mask[i,0] = 0.0 # "<claim_draw>" is the first element

        masked_logits = logits + action_mask
        dist = Categorical(logits=masked_logits)
        active_actions = dist.sample()
        active_log_probs = dist.log_prob(active_actions)

        # Store results for active games and prepare UCI moves
        for i, orig_idx in enumerate(active_indices):
            games[orig_idx].update_trajectory(
                active_fens[i], # str
                active_reps[i], # []
                active_actions[i].cpu(), # [possible_moves]
                active_values[i].cpu(), # []
                active_log_probs[i].cpu(), # []
                action_mask[i].cpu() # [possible_moves]
            )
            uci_moves[orig_idx] = IDX_TO_UCI_MOVE[active_actions[i].item()]

        return uci_moves

    def _update_result(self, dones: List[bool], infos: List[Dict], games: List[Game]):
        """
        Update games' completion status.
        """
        for i, (done,info,game) in enumerate(zip(dones,infos,games)):
            if not game.active:
                continue

            if done:
                if info.get('truncation_due_to_error',False):
                    game.update_game_status(True,reason="Error during move.")
                elif info.get('max_steps_exceeded',False):
                    game.update_game_status(True,reason="Maximum steps exceeded.")
                else:
                    # Game completed normally, get result
                    game.update_game_status(True,reason=info.get('result'))
                    #print(game.active)

    def _collect_rollout(self):
        """
        Run self-play games until completion and collect data. Uses Game objects to track trajectories and status.
        Workflow:
            1. get initial fens and reps using env.reset(); initialize games
            2. call _make_action(). Get uci_moves and update games' trajectories
            3. call env.step() to get next fens&reps and dones&infos
            4. call_update_result() to update games' completion status
            5. call _make_action() again with next fens&reps,.. repeat until all completed.
            6. call buffer.push_game() to push games into buffer.
        """
        fens, reps = self.env.reset()
        batch_size = self.rollout_batch_size

        games = [Game() for _ in range(batch_size)]

        # Game loop
        step_count = 0
        active_count = batch_size
        with tqdm(desc="Self-play rollout", unit="step") as pbar:
            while active_count > 0:
                step_count += 1

                # Get actions for all games
                uci_moves = self._make_action(fens,reps,games)

                # Step environment
                # ! Need to ensure BatchChessEnv handles "0000" dummy uci move
                next_fens, next_reps, dones, infos = self.env.step(uci_moves)

                # Update game completion status
                self._update_result(dones,infos,games)

                # Update fens and repetition counts for next iteration
                fens = next_fens
                reps = next_reps

                active_count = sum(1 for game in games if game.active)


                if step_count % 1 == 0 or active_count == 0:
                    pbar.update(min(1,step_count-pbar.n))
                    pbar.set_postfix(active=active_count)

        # Add normally completed games to buffer and track stats
        white_wins = 0
        black_wins = 0
        draws = 0
        errors = 0
        max_step_exceeded = 0
        for game in games:
            if game.completion_reason == "1-0":
                self.buffer.push_game(game)
                white_wins += 1
            elif game.completion_reason == "0-1":
                self.buffer.push_game(game)
                black_wins += 1
            elif game.completion_reason == "1/2-1/2":
                self.buffer.push_game(game)
                draws += 1
            elif game.completion_reason == "Error during move.":
                errors += 1
            elif game.completion_reason == "Maximum steps exceeded.":
                max_step_exceeded += 1
            else:
                print(f"Error: game completion reason not recognized: {game.completion_reason}")
                raise
        log_message = (
            f"Episode: {self.current_episode+1}/{self.num_episodes} | "
            f"Collected {white_wins+black_wins+draws}/{self.rollout_batch_size} valid games. White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws} | "
            f"Errors during move: {errors} | "
            f"Maximum steps exceeded: {max_step_exceeded}"
        )
        self.logger.info(log_message)
        print(log_message)

        # --- Inspect Buffer ---
        if batch_size <= 10:
            self.inspect_buffer(games=games)
        # --- Inspect Buffer ---

    def _update_params(self):
        """Update model parameters using PPO with additional invalid move penalty."""
        if len(self.buffer) < self.update_batch_size:
            print(f"Not enough samples.")

        self.model.train()

        batch_count = 0
        self.optimizer.zero_grad()

        for k in range(self.k_epochs):
            batch_iterator = self.buffer.sample()
            if not batch_iterator:
                print(f"Error: No batches generated from buffer.")
                raise
            
            total_batches = len(self.buffer) // self.update_batch_size
            with tqdm(total=total_batches,desc=f"Epoch {k+1}/{self.k_epochs}") as pbar:
                for fens_b, reps_b, acts_b, old_logps_b, old_vals_b, invs_b, advs_b in batch_iterator:
                    batch_count += 1

                    reps_b = reps_b.to(self.device)
                    acts_b = acts_b.to(self.device)
                    old_logps_b = old_logps_b.to(self.device)
                    old_vals_b = old_vals_b.to(self.device)
                    advs_b = advs_b.to(self.device)
                    invs_b = invs_b.to(self.device)

                    returns = advs_b + old_vals_b

                    advs_b = (advs_b-advs_b.mean()) / (advs_b.std()+1e-8)

                    with autocast(self.device_str):
                        logits, values = self.model(fens_b,reps_b.squeeze(-1))
                        probs = torch.softmax(logits,dim=-1)
                        masked_logits = logits + invs_b
                        dist = Categorical(logits=masked_logits)

                        # Policy loss
                        new_logps = dist.log_prob(acts_b.squeeze(-1))
                        ratio = torch.exp(new_logps - old_logps_b.squeeze(-1))
                        surr1 = ratio * advs_b.squeeze(-1)
                        surr2 = torch.clamp(ratio,1.0-self.clip_eps,1.0+self.clip_eps) * advs_b.squeeze(-1)
                        policy_loss = - torch.min(surr1,surr2).mean()

                        # Value loss
                        value_loss = nn.functional.mse_loss(values,returns.squeeze(-1))

                        # Entropy loss
                        entropy_loss = - dist.entropy().mean()

                        # Invalid move penalty
                        invalid_move_mask = (invs_b==-1e9).float()
                        invalid_probs_sum = (probs*invalid_move_mask).sum(dim=-1)
                        invalid_loss = invalid_probs_sum.mean()

                        loss = (
                            policy_loss +
                            self.value_ratio * value_loss +
                            self.entropy_ratio * entropy_loss + 
                            self.invalid_pen_ratio * invalid_loss
                        )

                    self.scaler.scale(loss/self.accumulation_steps).backward()

                    self.policy_loss_accumulator += policy_loss.item()
                    self.value_loss_accumulator += value_loss.item()
                    self.entropy_loss_accumulator += entropy_loss.item()
                    self.invalid_loss_accumulator += invalid_loss.item()
                    self.samples_since_log += 1

                    if batch_count % self.accumulation_steps == 0:
                        self.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.max_grad_norm)
                        self.grad_norm = self.grad_norm.item()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.global_steps += 1
                        
                        if self.global_steps % self.log_every_steps == 0 and self.global_steps != 0:
                            self._log()

                    pbar.update(1)
                    pbar.set_postfix({
                        'Loss': f"{loss.item():.3f}",
                        'PolLoss': f"{policy_loss.item():.3f}", 
                        'ValLoss': f"{value_loss.item():.3f}",
                        'EntLoss': f"{entropy_loss.item():.3f}", 
                        'InvLoss': f"{invalid_loss.item():.3f}",
                        'Step': f"{self.global_steps}"
                    })

            

        # Update old model with new parameters
        self.model_old.load_state_dict(self.model.state_dict())

        # Clear buffer
        self.buffer.clear()

    def train(self):
        """Main training loop"""
        for episode in range(self.start_episode, self.start_episode+self.num_episodes):
            self.current_episode = episode

            # Collect rollouts
            self._collect_rollout()

            # Update parameters
            self._update_params()

            if (episode + 1) % self.save_every_episodes == 0:
                ckpt_idx = (episode+1)//self.save_every_episodes
                self._save_checkpoint(episode=episode,mark=f"{ckpt_idx:02d}")

        self._save_checkpoint(episode=self.num_episodes+self.start_episode,mark="final")
        swanlab.finish()
        print("Training complete!")

    def inspect_buffer(self, games=None, verbose=True):
        """
        Inspect the buffer contents to verify examples are properly processed.
        
        Args:
            games: Optional list of Game objects that were just pushed (for cross-reference)
            verbose: Whether to print detailed information
        
        Returns:
            dict: Statistics about the buffer contents
        """
        if len(self.buffer) == 0:
            print("Buffer is empty. Nothing to inspect.")
            return {}
        
        stats = {
            "total_examples": len(self.buffer),
            "white_examples": 0,
            "black_examples": 0,
            "avg_repetition": 0,
            "avg_advantage": 0,
            "max_advantage": float('-inf'),
            "min_advantage": float('inf'),
            "invalid_move_proportion": 0,
            "action_distribution": {},
        }
        
        # Sample inspection of buffer entries
        fens = list(self.buffer.fens)
        reps = list(self.buffer.repetition_counts)
        actions = list(self.buffer.actions)
        log_probs = list(self.buffer.log_probs)
        values = list(self.buffer.values)
        advantages = list(self.buffer.advantages)
        invalid_masks = list(self.buffer.invalid_masks)
        
        # Count white and black examples
        for fen in fens:
            board = chess.Board(fen)
            if board.turn:  # True for white, False for black
                stats["white_examples"] += 1
            else:
                stats["black_examples"] += 1
        
        # Compute averages
        stats["avg_repetition"] = sum(r.item() if torch.is_tensor(r) else r for r in reps) / len(reps)
        
        adv_values = [a.item() if torch.is_tensor(a) else a for a in advantages]
        stats["avg_advantage"] = sum(adv_values) / len(adv_values)
        stats["max_advantage"] = max(adv_values)
        stats["min_advantage"] = min(adv_values)
        
        # Check invalid move masks
        invalid_move_count = 0
        total_moves = 0
        for mask in invalid_masks:
            if torch.is_tensor(mask):
                invalid_count = (mask == -1e9).sum().item()
                total_count = mask.numel()
            else:
                invalid_count = sum(1 for x in mask if x == -1e9)
                total_count = len(mask)
            
            invalid_move_count += invalid_count
            total_moves += total_count
        
        stats["invalid_move_proportion"] = invalid_move_count / total_moves if total_moves > 0 else 0
        
        # Action distribution
        for action in actions:
            action_idx = action.item() if torch.is_tensor(action) else action
            action_uci = IDX_TO_UCI_MOVE.get(action_idx, "unknown")
            stats["action_distribution"][action_uci] = stats["action_distribution"].get(action_uci, 0) + 1
        
        # Sort action distribution by frequency
        stats["action_distribution"] = dict(sorted(
            stats["action_distribution"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])  # Only show top 10
        
        # Verify a few examples if verbose
        if verbose:
            print(f"\n===== Buffer Inspection =====")
            print(f"Total examples: {stats['total_examples']}")
            print(f"White/Black split: {stats['white_examples']}/{stats['black_examples']}")
            print(f"Average repetition count: {stats['avg_repetition']:.2f}")
            print(f"Advantage stats: Avg={stats['avg_advantage']:.4f}, Min={stats['min_advantage']:.4f}, Max={stats['max_advantage']:.4f}")
            print(f"Invalid move proportion: {stats['invalid_move_proportion']*100:.2f}%")
            
            print("\nTop actions:")
            for uci, count in stats["action_distribution"].items():
                print(f"  {uci}: {count} ({count/len(actions)*100:.1f}%)")
            
            # Detailed example inspection for the first few examples
            num_samples = min(5, len(self.buffer))
            print(f"\nDetailed inspection of {num_samples} random examples:")
            
            indices = np.random.choice(len(self.buffer), num_samples, replace=False)
            for idx in indices:
                fen = fens[idx]
                board = chess.Board(fen)
                action_idx = actions[idx].item() if torch.is_tensor(actions[idx]) else actions[idx]
                action_uci = IDX_TO_UCI_MOVE.get(action_idx, "unknown")
                
                print(f"\nExample {idx}:")
                print(f"  Turn: {'White' if board.turn else 'Black'}")
                print(f"  FEN: {fen}")
                print(f"  Action: {action_uci}")
                print(f"  Value: {values[idx].item() if torch.is_tensor(values[idx]) else values[idx]:.4f}")
                print(f"  Advantage: {advantages[idx].item() if torch.is_tensor(advantages[idx]) else advantages[idx]:.4f}")
                
                # Verify move validity
                if action_uci != "<claim_draw>" and action_uci != "unknown":
                    try:
                        move = chess.Move.from_uci(action_uci)
                        is_legal = move in board.legal_moves
                        print(f"  Move is {'legal' if is_legal else 'ILLEGAL'}")
                    except ValueError:
                        print(f"  Invalid UCI move: {action_uci}")
                elif action_uci == "<claim_draw>":
                    print(f"  Draw claim {'valid' if board.can_claim_draw() else 'INVALID'}")
                
                # Check invalid mask for this example
                mask = invalid_masks[idx]
                if torch.is_tensor(mask):
                    invalid_count = (mask == -1e9).sum().item()
                    total_count = mask.numel()
                else:
                    invalid_count = sum(1 for x in mask if x == -1e9)
                    total_count = len(mask)
                
                print(f"  Invalid moves: {invalid_count}/{total_count} ({invalid_count/total_count*100:.1f}%)")
                
            # Cross-reference with games if provided
            if games:
                print("\nCross-reference with provided games:")
                valid_game_count = sum(1 for g in games if g.valid)
                print(f"  Valid games: {valid_game_count}/{len(games)}")
                
                # Sample a valid game for detailed inspection
                for game in games:
                    if game.valid:
                        result = game.completion_reason
                        white_moves = len(game.get_white_trajectory()['fens'])
                        black_moves = len(game.get_black_trajectory()['fens'])
                        
                        print(f"  Game result: {result}")
                        print(f"  White moves: {white_moves}")
                        print(f"  Black moves: {black_moves}")
                        print(f"  Total positions: {len(game.fens)}")
                        
                        # Show the last position
                        if game.fens:
                            last_fen = game.fens[-1]
                            last_board = chess.Board(last_fen)
                            print(f"  Last position: {last_fen}")
                            print(f"  Board is in checkmate: {last_board.is_checkmate()}")
                            print(f"  Board is in stalemate: {last_board.is_stalemate()}")
                            print(f"  Board is in draw: {last_board.is_insufficient_material()}")
                        break
        
        return stats

    def _save_checkpoint(self, episode: str, mark: str):
        """Save model checkpoint"""
        checkpoint_path = f"./ckpts/chessformer-rl_{mark}.pth"
        checkpoint = {
            'episode': episode,
            'global_steps': self.global_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model_config,
        }
        torch.save(checkpoint,checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def resume(self, checkpoint_path, from_sl_checkpoint: bool=False):
        """Resume from model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_old.load_state_dict(checkpoint["model_state_dict"])
        if not from_sl_checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_episode = checkpoint["episode"] + 1
            self.global_steps = checkpoint["global_steps"]
            self.current_episode = self.start_episode
        print(f"Resumed training from checkpoint: {checkpoint_path}, starting episode {self.start_episode}.")

    def _log(self):
        if self.samples_since_log == 0:
            return
        
        avg_policy_loss = self.policy_loss_accumulator / self.samples_since_log
        avg_value_loss = self.value_loss_accumulator / self.samples_since_log
        avg_entropy_loss = self.entropy_loss_accumulator / self.samples_since_log
        avg_invalid_loss = self.invalid_loss_accumulator / self.samples_since_log

        log_message = (
            f"Episode: {self.current_episode+1}/{self.num_episodes} | "
            f"Step: {self.global_steps:06d} | "
            f"Avg Policy Loss: {avg_policy_loss:.4f} | "
            f"Avg Value Loss: {avg_value_loss:.4f} | "
            f"Avg Entropy Loss: {avg_entropy_loss:.4f} | "
            f"Avg Invalid Loss: {avg_invalid_loss:.4f} | "
            f"Grad Norm: {self.grad_norm:.4f}"
        )

        piece_emb = self.model.fen_tokenizer.piece_embed.weight.data.float() # Ensure float for stats
        piece_norm = torch.norm(piece_emb, dim=1)
        piece_norm_mean = piece_norm.mean().item()
        piece_norm_std = piece_norm.std().item()

        pos_emb = self.model.fen_tokenizer.pos_embed.weight.data.float() # Ensure float for stats
        pos_norm = torch.norm(pos_emb, dim=1)
        pos_norm_mean = pos_norm.mean().item()
        pos_norm_std = pos_norm.std().item()

        log_message += (
            f" | Piece Emb Norm (Mean/Std): {piece_norm_mean:.4f}/{piece_norm_std:.4f}"
            f" | Pos Emb Norm (Mean/Std): {pos_norm_mean:.4f}/{pos_norm_std:.4f}"
        )

        buffer_size = len(self.buffer)
        log_message += f" | Buffer Size: {buffer_size:06d}"

        log_message += f" | Learning Rate: {self.optimizer.param_groups[0]['lr']*1e4:.4f}e-4"

        self.logger.info(log_message)

        swanlab.log({
            "avg pol loss": avg_policy_loss,
            "avg val loss": avg_value_loss,
            "avg ent loss": avg_entropy_loss,
            "avg inv loss": avg_invalid_loss,
            "episode": self.current_episode,
            "buffer size": buffer_size,
            "piece_norm_mean": piece_norm_mean,
            "piece_norm_std": piece_norm_std,
            "pos_norm_mean": pos_norm_mean,
            "pos_norm_std": pos_norm_std,
            "step": self.global_steps,
            "grad_norm": self.grad_norm,
        })

        self.policy_loss_accumulator = 0.0
        self.value_loss_accumulator = 0.0
        self.entropy_loss_accumulator = 0.0
        self.invalid_loss_accumulator = 0.0
        self.samples_since_log = 0



def train():
    torch.manual_seed(1982)
    model_config = {
        "num_blocks": 20,
        "hidden_size": 640,
        "intermediate_size": 1728,
        "num_heads": 8,
        "dropout": 0.05,
        "possible_moves": len(UCI_MOVE_TO_IDX),
        "dtype": torch.float32
    }
    model = ChessFormerModel(**model_config)
    #checkpoint = torch.load("./ckpts/chessformer-sl_01.pth")
    #model = model.load_state_dict(checkpoint["model_state_dict"])
    trainer = RLTrainer(
        model=model,
        learning_rate=1e-5,
        value_ratio=0.6,
        entropy_ratio=0.01,
        invalid_pen_ratio=0.15,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        k_epochs=10,
        num_episodes=1024,
        update_batch_size=154,
        accumulation_steps=10,
        max_grad_norm=1.0,
        rollout_batch_size=384,
        save_every_episodes=24,
        log_every_steps=32,
        model_config=model_config,
        experiment_name="chessformer-rl_0"
    )
    trainer.resume("./ckpts/chessformer-sl_06.pth",from_sl_checkpoint=True)
    trainer.train()


if __name__ == "__main__":
    train()
