import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import logging
from typing import Dict, Optional
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from utils import UCI_MOVE_TO_IDX
from model import ChessFormerModel

import swanlab


class SLTrainer:
    def __init__(self,
                 model: ChessFormerModel,
                 dataloader: DataLoader,
                 learning_rate: float,
                 value_ratio: float,
                 invalid_pen_ratio: float,
                 num_epochs: float,
                 accumulation_steps: int,
                 save_every_steps: int,
                 log_every_steps: int,
                 warmup_ratio: int,
                 lr_scheduler_type: str,
                 model_config: Dict,
                 experiment_name: Optional[str]=None):
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.model = model.to(self.device)
        self.total_moves = self.model.possible_moves
        self.model_config = model_config

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"---Using device: {self.device}, Model params: {num_params/1e6}M---")

        self.value_ratio = value_ratio
        self.invalid_pen_ratio = invalid_pen_ratio # penalize invalid moves

        self.dataloader = dataloader

        self.num_epochs = num_epochs
        self.start_epoch = 0
        self.global_steps = 0
        self.accumulation_steps = accumulation_steps
        num_steps_per_epoch = len(self.dataloader) // self.accumulation_steps
        self.total_optim_steps = int(num_steps_per_epoch * self.num_epochs)
        self.num_epochs = math.floor(self.num_epochs+1-1e-8)
        self.current_epoch = self.start_epoch

        self.save_every_steps = save_every_steps
        self.log_every_steps = log_every_steps

        self.optimizer = optim.AdamW(model.parameters(),lr=learning_rate)
        self.learning_rate = learning_rate
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.scaler = GradScaler(self.device_str)

        self.total_loss_accumulator = 0.0
        self.act_loss_accumulator = 0.0
        self.val_loss_accumulator = 0.0
        self.inv_loss_accumulator = 0.0

        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self._prepare_lr_scheduler(warmup_ratio, lr_scheduler_type)

        # logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('./log/sl_training.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # Add the handlers to the logger
        if not self.logger.handlers: # Avoid adding multiple handlers if __init__ is called again
            self.logger.addHandler(file_handler)

        # swanlab
        swanlab.init(
            project="chessformer",
            experiment_name=experiment_name,
            config = {
                "value_ratio": self.value_ratio,
                "invalid_pen_ratio": self.invalid_pen_ratio,
                "learning_rate": self.learning_rate,
                "total_optim_steps": self.total_optim_steps,
                "lr_scheduler_type": self.lr_scheduler_type,
                "warmup_ratio": self.warmup_ratio,
                "model_size": num_params
            },
            logdir="./log"
        )

    def _prepare_lr_scheduler(self, warmup_ratio: float, scheduler_type: str, initial_last_epoch: int=-1):
        self.num_warmup_steps = int(self.total_optim_steps * warmup_ratio)

        print(f"Total training steps: {self.total_optim_steps}")
        print(f"Warmup steps: {self.num_warmup_steps}")

        if scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.total_optim_steps,
                last_epoch=initial_last_epoch
            )
        elif scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.total_optim_steps,
                last_epoch=initial_last_epoch           
            )
        else:
            self.lr_scheduler = None

        if self.lr_scheduler is not None:
            self.lr_scheduler.base_lrs = [self.learning_rate for _ in self.optimizer.param_groups]

    def train(self):
        self.model.train()

        for epoch_idx in range(self.start_epoch,self.start_epoch+self.num_epochs):
            self.current_epoch = epoch_idx
            print(f"Epoch {self.current_epoch+1}/{self.start_epoch+self.num_epochs} Started!")
            total_epoch_loss = 0.0
            total_act_loss = 0.0
            total_val_loss = 0.0
            total_inv_loss = 0.0
            steps_in_epoch = 0

            # Initialize tqdm progress bar
            pbar = tqdm(enumerate(self.dataloader),
                        total=len(self.dataloader),
                        desc=f"Epoch {epoch_idx+1}/{self.start_epoch+self.num_epochs}")
            
            self.optimizer.zero_grad()

            for idx, sample in pbar:
                fens = sample["fen"]
                repetition_counts = sample["repetition_count"].to(self.device) # integer
                best_moves_uci = sample["best_move"]
                scores = sample["score"].to(self.device)
                valid_moves_str_list = sample["valid_moves"]
                batch_size = len(fens)

                # Convert UCI moves to tensor
                try:
                    best_moves_indices = [UCI_MOVE_TO_IDX[move] for move in best_moves_uci]
                except KeyError as e:
                    print(f"Error: Move '{e}' not found in UCI_MOVE_TO_IDX")
                    continue
                best_moves_tensor = torch.tensor(best_moves_indices,dtype=torch.long).to(self.device)

                # Create invalid move mask
                invalid_move_mask = torch.ones((batch_size, self.total_moves),device=self.device,dtype=torch.float32)
                for i in range(batch_size):
                    valid_uci_moves = valid_moves_str_list[i].split(' ')
                    try:
                        valid_indices = [UCI_MOVE_TO_IDX[move] for move in valid_uci_moves]
                        if valid_indices:
                            invalid_move_mask[i, valid_indices] = 0.0
                    except Exception as e:
                        self.logger.error(f"Error processing valid_moves. Error: {e}")
                
                with autocast(self.device_str):
                    action_logits, values = self.model(fens,repetition_counts)

                    # Standard CE loss for the best move
                    act_loss = self.ce_loss(action_logits,best_moves_tensor)
                    # Standard MSE loss for the value
                    val_loss = self.mse_loss(values,scores)
                    # Penalty for invalid moves
                    probs = torch.softmax(action_logits, dim=-1)
                    invalid_probs_sum = (probs * invalid_move_mask).sum(dim=-1)
                    inv_loss = invalid_probs_sum.mean()

                    total_loss = act_loss + self.value_ratio * val_loss + self.invalid_pen_ratio * inv_loss

                self.scaler.scale(total_loss/self.accumulation_steps).backward()
                #loss_to_backward = total_loss/self.accumulation_steps
                #loss_to_backward.backward()

                total_epoch_loss += total_loss.item()
                total_act_loss += act_loss.item()
                total_val_loss += val_loss.item()
                total_inv_loss += inv_loss.item()
                steps_in_epoch += 1

                if (idx+1)%self.accumulation_steps == 0 or (idx+1) == len(self.dataloader):
                    #self.scaler.unscale_(self.optimizer)
                    #torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.001)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.global_steps += 1
                    
                    self.total_loss_accumulator += total_loss.item()
                    self.act_loss_accumulator += act_loss.item()
                    self.val_loss_accumulator += val_loss.item()
                    self.inv_loss_accumulator += inv_loss.item()

                    pbar.set_postfix({
                        "Loss": f"{total_loss.item():.3f}",
                        "ActLoss": f"{act_loss.item():.3f}",
                        "ValLoss": f"{val_loss.item():.3f}",
                        "InvLoss": f"{inv_loss.item():.3f}",
                        "Step": f"{self.global_steps}",
                        "LR": f"{self.optimizer.param_groups[0]['lr']*1e4:.3f}e-4"
                    })

                    if (self.global_steps)%self.save_every_steps == 0 and self.global_steps != 0:
                        ckpt_idx = self.global_steps // self.save_every_steps
                        self._save_checkpoint(self.current_epoch,f"{ckpt_idx:02d}")

                    if (self.global_steps)%self.log_every_steps == 0 and self.global_steps != 0:
                        self._log()

                    if self.global_steps >= self.total_optim_steps:
                        break

                if self.global_steps >= self.total_optim_steps:
                    break

            avg_total_loss = total_epoch_loss / steps_in_epoch
            avg_act_loss = total_act_loss / steps_in_epoch
            avg_val_loss = total_val_loss / steps_in_epoch
            avg_inv_loss = total_inv_loss / steps_in_epoch
            message = f"Epoch {epoch_idx+1} ended. Average Stats: [total: {avg_total_loss:.4f}, act: {avg_act_loss:.4f}, val: {avg_val_loss:.4f}, inv: {avg_inv_loss:.4f}]"
            print(message)
            self.logger.info(message)

        self._save_checkpoint(self.current_epoch,"final")
        swanlab.finish()

    def _save_checkpoint(self, epoch: int, mark: str):
        checkpoint_path = f"./ckpts/chessformer-sl_{mark}.pth"
        checkpoint = {
            'epoch': epoch,
            'global_steps': self.global_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model_config,
            "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None
        }
        torch.save(checkpoint,checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        log_message = (
            f"Step: {self.global_steps:06d} | "
            f"Checkpoint saved to {checkpoint_path}"
        )
        self.logger.info(log_message)

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_steps = checkpoint["global_steps"]
        self.current_epoch = self.start_epoch

        print(f"Resetting lr_scheduler due to resume...")
        self.total_optim_steps = self.total_optim_steps + self.global_steps
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        self._prepare_lr_scheduler(
            self.warmup_ratio,
            self.lr_scheduler_type,
            initial_last_epoch=self.global_steps
            ) # reset lr_scheduler
        print(f"Resumed training from checkpoint: {checkpoint_path}, starting epoch {self.start_epoch}.")
        log_message = (
            f"Step: {self.global_steps:06d} | "
            f"Resumed from {checkpoint_path}"
        )
        self.logger.info(log_message)


    @torch.no_grad
    def _log(self):
        """
        log metrics with Logger

        Metrics include:
            - total loss, act loss, val loss
            - norm, std, variance of embeddings (position, pieces)
        """

        # 1. Standard Training Metrics
        avg_total_loss = self.total_loss_accumulator/self.log_every_steps
        avg_act_loss = self.act_loss_accumulator/self.log_every_steps
        avg_val_loss = self.val_loss_accumulator/self.log_every_steps
        avg_inv_loss = self.inv_loss_accumulator/self.log_every_steps
        log_message = (
            f"Step: {self.global_steps:06d} | "
            f"Learning Rate: {self.optimizer.param_groups[0]['lr']*1e4:.4f}e-4 | "
            f"Avg Total Loss: {avg_total_loss:.4f} | "
            f"Avg Actor Loss: {avg_act_loss:.4f} | "
            f"Avg Value Loss: {avg_val_loss:.4f} | "
            f"Avg Invalid Loss: {avg_inv_loss:.4f}"
        )

        # Embedding Analysis
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

        self.logger.info(log_message)

        swanlab.log({
            "avg_total_loss": avg_total_loss,
            "avg_actor_loss": avg_act_loss,
            "avg_value_loss": avg_val_loss,
            "avg_invalid_loss": avg_inv_loss,
            "piece_norm_mean": piece_norm_mean,
            "piece_norm_std": piece_norm_std,
            "pos_norm_mean": pos_norm_mean,
            "pos_norm_std": pos_norm_std,
            "step": self.global_steps,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        })

        # Reset accumulators
        self.total_loss_accumulator = 0.0
        self.act_loss_accumulator = 0.0
        self.val_loss_accumulator = 0.0
        self.inv_loss_accumulator = 0.0

def train():
    torch.manual_seed(640)
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
    ds = load_dataset("kaupane/lichess-2023-01-stockfish-annotated",split="depth18")
    ds = ds.with_format("torch")
    dataloader = DataLoader(
        ds,
        batch_size=192,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    trainer = SLTrainer(
        model=model,
        dataloader=dataloader,
        learning_rate=1.5e-4,
        value_ratio=4.0, # increasing value_ratio results in faster actor_loss and invalid_loss drop, but has minor effect on value_loss curve. Weird, right? Maybe value does significantly help the model learn
        invalid_pen_ratio=0.1,
        num_epochs=4.0,
        accumulation_steps=8,
        save_every_steps=6144,
        log_every_steps=24,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine", 
        model_config=model_config,
        experiment_name="chessformer-sl_0"
    )
    trainer.train()


if __name__ == "__main__":
    train()


