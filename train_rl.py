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
from typing import List, Dict, Optional

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
                 track_kl: bool,
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

        self.track_kl = track_kl
        if self.track_kl:
            self.kl_div = 0.0

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
        
        # After collecting rollouts, inspect a random game
        """if games:  # games from _collect_rollout
            valid_games = [g for g in games if g.valid]
            if valid_games:
                import random
                game_to_inspect = random.choice(valid_games)
                inspect_game_trajectory(game_to_inspect, stockfish_path="/usr/games/stockfish")"""

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

                        if self.track_kl:
                            # In _update_params, when computing KL:
                            with torch.no_grad():
                                # Put BOTH models in eval mode for consistent predictions
                                self.model_old.eval()
                                self.model.eval()
                                
                                # Get logits from the OLD model
                                old_logits, _ = self.model_old(fens_b, reps_b.squeeze(-1))
                                old_masked_logits = old_logits + invs_b
                                old_dist = Categorical(logits=old_masked_logits)
                                
                                # Get current logits (still in eval mode)
                                current_logits, _ = self.model(fens_b, reps_b.squeeze(-1))
                                current_masked_logits = current_logits + invs_b
                                current_dist = Categorical(logits=current_masked_logits)
                                
                                self.kl_div = torch.distributions.kl_divergence(old_dist, current_dist).mean()
                                
                                # Put model back in train mode
                                self.model.train()

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

        if self.track_kl:
            log_message += (
                f" | KL Div: {self.kl_div:.4f}"
            )

        buffer_size = len(self.buffer)
        log_message += f" | Buffer Size: {buffer_size:06d}"

        log_message += f" | Learning Rate: {self.optimizer.param_groups[0]['lr']*1e4:.4f}e-4"

        self.logger.info(log_message)

        swanlab_message = {
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
        }
        if self.track_kl:
            swanlab_message["kl div"] = self.kl_div
        swanlab.log(swanlab_message)

        self.policy_loss_accumulator = 0.0
        self.value_loss_accumulator = 0.0
        self.entropy_loss_accumulator = 0.0
        self.invalid_loss_accumulator = 0.0
        self.samples_since_log = 0

def inspect_game_trajectory(game, stockfish_path=None, stockfish_depth=16):
    """
    Inspect a complete game trajectory showing how values and advantages evolve.
    
    Args:
        game: A Game object containing the complete trajectory
        stockfish_path: Path to stockfish executable
        stockfish_depth: Depth for stockfish analysis
    """
    if not game.valid or not game.fens:
        print("Invalid or empty game!")
        return
    
    # Initialize Stockfish if available
    sf_engine = None
    stockfish_path="/usr/games/stockfish"
    stockfish_depth=18
    if stockfish_path:
        try:
            from utils import Engine, StockfishConfig
            config = StockfishConfig(
                engine_path=stockfish_path,
                depth=stockfish_depth
            )
            sf_engine = Engine(type="stockfish", stockfish_config=config)
            print(f"Stockfish initialized at depth {stockfish_depth}")
        except Exception as e:
            print(f"Failed to initialize Stockfish: {e}")
    
    print(f"\n{'='*120}")
    print(f"GAME TRAJECTORY ANALYSIS - Result: {game.game_result}")
    print(f"{'='*120}\n")
    
    # Convert game result to numeric
    if game.game_result == "1-0":
        numeric_result = 1.0
    elif game.game_result == "0-1":
        numeric_result = -1.0
    else:
        numeric_result = 0.0
    
    import torch
    # Prepare data for visualization
    trajectory_data = []

    for i in range(len(game.fens)):
        board = chess.Board(game.fens[i])
        turn = "White" if board.turn else "Black"
        move_number = (len(board.move_stack) + 1) // 2 + (1 if board.turn else 0)
        
        # Get move
        action_idx = game.actions[i].item() if torch.is_tensor(game.actions[i]) else game.actions[i]
        move_uci = IDX_TO_UCI_MOVE.get(action_idx, "unknown")
        
        # Get values
        value = game.values[i].item() if torch.is_tensor(game.values[i]) else game.values[i]
        log_prob = game.log_probs[i].item() if torch.is_tensor(game.log_probs[i]) else game.log_probs[i]
        
        # Get Stockfish evaluation
        sf_eval = None
        if sf_engine:
            try:
                sf_eval = sf_engine.analyze_position(board)
                sf_eval = sf_eval if board.turn==chess.WHITE else -sf_eval
            except:
                pass
        
        trajectory_data.append({
            'move_number': move_number,
            'turn': turn,
            'move': move_uci,
            'value': value,
            'log_prob': log_prob,
            'sf_eval': sf_eval,
            'fen': game.fens[i]
        })
    
    # Calculate advantages using the same GAE method
    import torch
    values_tensor = torch.tensor([d['value'] for d in trajectory_data])
    
    # Separate by color
    white_indices = [i for i, d in enumerate(trajectory_data) if d['turn'] == 'White']
    black_indices = [i for i, d in enumerate(trajectory_data) if d['turn'] == 'Black']
    
    # Calculate advantages for each side
    gamma = 0.99  # Should match your training config
    gae_lambda = 0.95
    
    def compute_advantages(indices, final_reward):
        if not indices:
            return {}
        
        values = torch.tensor([trajectory_data[i]['value'] for i in indices])
        T = len(values)
        advantages = torch.zeros(T)
        
        next_value = 0.0
        gae = 0.0
        
        for t in reversed(range(T)):
            reward = final_reward if t == T-1 else 0.0
            delta = reward + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            next_value = values[t]
        
        return {indices[i]: advantages[i].item() for i in range(T)}
    
    white_advantages = compute_advantages(white_indices, numeric_result)
    black_advantages = compute_advantages(black_indices, -numeric_result)
    all_advantages = {**white_advantages, **black_advantages}
    
    # Display trajectory
    print(f"{'Move':<6} {'Turn':<6} {'Action':<12} {'Value':<8} {'Adv':<8} {'SF Eval':<8} {'V+A':<8} {'LogP':<8}")
    print("-" * 80)
    
    for i, data in enumerate(trajectory_data):
        adv = all_advantages.get(i, 0.0)
        v_plus_a = data['value'] + adv
        sf_str = f"{data['sf_eval']:.3f}" if data['sf_eval'] is not None else "N/A"
        
        # Color code based on advantage sign
        adv_str = f"{adv:+.3f}"
        if adv > 0.1:
            adv_str = f"\033[92m{adv_str}\033[0m"  # Green for positive
        elif adv < -0.1:
            adv_str = f"\033[91m{adv_str}\033[0m"  # Red for negative
        
        print(f"{data['move_number']:<6} {data['turn']:<6} {data['move']:<12} "
              f"{data['value']:+.3f}   {adv_str}   {sf_str:<8} "
              f"{v_plus_a:+.3f}   {data['log_prob']:.3f}")
    
    # Summary statistics
    print(f"\n{'-'*80}")
    print("SUMMARY STATISTICS")
    print(f"{'-'*80}")
    
    all_values = [d['value'] for d in trajectory_data]
    all_advs = [all_advantages.get(i, 0.0) for i in range(len(trajectory_data))]
    
    # Split by game phase
    early_game = trajectory_data[:20]
    mid_game = trajectory_data[20:40]
    end_game = trajectory_data[40:]
    
    for phase_name, phase_data in [("Early Game (moves 1-20)", early_game),
                                   ("Mid Game (moves 21-40)", mid_game),
                                   ("End Game (moves 41+)", end_game)]:
        if not phase_data:
            continue
            
        phase_indices = trajectory_data.index(phase_data[0])
        phase_values = [d['value'] for d in phase_data]
        phase_advs = [all_advantages.get(trajectory_data.index(d), 0.0) for d in phase_data]
        
        print(f"\n{phase_name}:")
        print(f"  Values: mean={np.mean(phase_values):.3f}, std={np.std(phase_values):.3f}")
        print(f"  Advantages: mean={np.mean(phase_advs):.3f}, std={np.std(phase_advs):.3f}, "
              f"max_abs={max(abs(a) for a in phase_advs):.3f}")
    
    # Value prediction accuracy vs Stockfish
    if sf_engine:
        sf_errors = []
        for data in trajectory_data:
            if data['sf_eval'] is not None:
                error = abs(data['value'] - data['sf_eval'])
                sf_errors.append(error)
        
        if sf_errors:
            print(f"\nValue prediction vs Stockfish:")
            print(f"  Mean absolute error: {np.mean(sf_errors):.3f}")
            print(f"  Max error: {max(sf_errors):.3f}")
            
            # Error by game phase
            early_errors = [abs(d['value'] - d['sf_eval']) for d in early_game if d['sf_eval'] is not None]
            mid_errors = [abs(d['value'] - d['sf_eval']) for d in mid_game if d['sf_eval'] is not None]
            end_errors = [abs(d['value'] - d['sf_eval']) for d in end_game if d['sf_eval'] is not None]
            
            if early_errors:
                print(f"  Early game MAE: {np.mean(early_errors):.3f}")
            if mid_errors:
                print(f"  Mid game MAE: {np.mean(mid_errors):.3f}")
            if end_errors:
                print(f"  End game MAE: {np.mean(end_errors):.3f}")
    
    # Show critical moments (high advantage magnitude)
    critical_moves = [(i, all_advantages.get(i, 0.0)) for i in range(len(trajectory_data))]
    critical_moves.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n{'-'*80}")
    print("CRITICAL MOMENTS (highest |advantage|)")
    print(f"{'-'*80}")
    
    for i, adv in critical_moves[:5]:
        data = trajectory_data[i]
        print(f"Move {data['move_number']} ({data['turn']}): {data['move']} - "
              f"Advantage: {adv:+.3f}, Value: {data['value']:+.3f}")


def train():
    torch.manual_seed(1949)
    model_config = {
        "num_blocks": 20,
        "hidden_size": 640,
        "intermediate_size": 1728,
        "num_heads": 8,
        "dropout": 0.00, # No dropout for RL
        "possible_moves": len(UCI_MOVE_TO_IDX),
        "dtype": torch.float32
    }
    model = ChessFormerModel(**model_config)
    trainer = RLTrainer(
        model=model,
        learning_rate=1e-5,
        value_ratio=0.2,
        entropy_ratio=0.015,
        invalid_pen_ratio=0.15,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.85,
        k_epochs=3,
        num_episodes=1024,
        update_batch_size=192,
        accumulation_steps=8,
        max_grad_norm=100,
        rollout_batch_size=384,
        save_every_episodes=16,
        log_every_steps=4,
        track_kl=False, # True for debug use, will be very slow.
        model_config=model_config,
        experiment_name="chessformer-rl_0"
    )
    #trainer.resume("./ckpts/chessformer-rl_init.pth",from_sl_checkpoint=True)
    trainer.train()


if __name__ == "__main__":
    train()
