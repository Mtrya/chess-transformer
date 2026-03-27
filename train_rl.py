import torch

from chess_core import UCI_MOVE_TO_IDX
from model import ChessFormerModel
from rl import RLTrainer


def train():
    torch.manual_seed(1949)
    model_config = {
        "num_blocks": 20,
        "hidden_size": 640,
        "intermediate_size": 1728,
        "num_heads": 8,
        "dropout": 0.00,  # No dropout for RL
        "possible_moves": len(UCI_MOVE_TO_IDX),
        "dtype": torch.float32,
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
        track_kl=False,  # True for debug use, will be very slow.
        model_config=model_config,
        experiment_name="chessformer-rl_0",
    )
    # trainer.resume("./ckpts/chessformer-rl_init.pth", from_sl_checkpoint=True)
    trainer.train()


if __name__ == "__main__":
    train()
