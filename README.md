# ChessFormer: Training Chess Models Without MCTS

A research project exploring the feasibility of training chess-playing neural networks without Monte Carlo Tree Search (MCTS), using pure supervised learning from engine evaluations and self-play reinforcement learning.

## Overview

ChessFormer investigates whether competitive chess models can be trained using only neural networks, without the traditional search algorithms that power engines like AlphaZero. We explore two main approaches:

1. **Supervised Learning (SL)**: Distillation from Stockfish evaluations
2. **Reinforcement Learning (RL)**: Self-play using Proximal Policy Optimization (PPO)

The project demonstrates both the potential and limitations of search-free chess models, providing valuable insights for future research.

## Key Findings

### ChessFormer-SL: Partial Success

- Shows reasonable opening and endgame play
- Struggles with midgame tactics, leading to frequent blunders
- Estimated ELO rating: ~1500 (informal assessment)
- With search enhancement, can occasionally defeat Stockfish
- Outperforms existing transformer-based chess models trained with next-token prediction

### ChessFormer-RL: Training Challenges

- Encountered significant training instabilities (gradient explosion, noisy rewards)
- Performance degraded from initial SL checkpoint
- Highlights the difficulty of RL training for complex strategic games

## Architecture

ChessFormer uses a custom transformer architecture optimized for chess:

- **Size**: 100.7M parameters (20 blocks, 640 hidden size, 8 heads, 1728 intermediate size)
- **Input**: FEN tokenization with 75-token sequences (64 board positions + 9 metadata tokens + 2 special tokens)
- **Output**: Policy head (1,969 possible moves) + Value head (position evaluation)
- **Features**: RMSNorm, SwiGLU FFN, custom FEN tokenizer

### FEN Tokenization Strategy

```markdown
64 piece tokens + 1 side-to-move + 4 castling rights + 1 en-passant + 
1 halfmove clock + 1 fullmove number + 1 repetition count = 73 tokens
+ 2 special tokens (action, value) = 75 total sequence length
```

## Installation & Setup

```bash
git clone https://github.com/Mtrya/chess-transformer
cd chess-transformer
pip install -r requirements.txt
```

### Running the Demo

For the interactive chess application, simply run app.py:

```bash
python app.py
```

It will download model checkpoints from huggingface.

Alternatively, try the [HuggingFace Space demo](https://huggingface.co/spaces/kaupane/Chessformer_Demo) without local setup.

### Training

```bash
# Supervised Learning
python train_sl.py

# Reinforcement Learning  
python train_rl.py
```

## Training Details

### Dataset

- **Source**: `kaupane/lichess-2023-01-stockfish-annotated`
- **Training**: 56M positions (depth18 split)
- **Validation**: depth27 split
- **Hardware**: RTX 4060Ti 16GB, ~2 weeks training time

### Performance Metrics

| Model | Action Loss | Value Loss | Invalid Loss |
|-------|-------------|------------|--------------|
| ChessFormer-SL(step 79872 checkpoint) | 1.7171 | 0.0424 | 0.0325 |
| ChessFormer-RL(intermediate ChessFormer-SL checkpoint) | 1.8329 | 0.0501 | 0.0484 |

Invalid loss measures probability assigned to illegal moves.

## Technical Insights & Limitations

### Key Challenges Identified

1. **Computational Inefficiency**: Heavy CPU operations (and it's python for-loop) in FEN tokenization during training. Future work should pre-process the entire dataset.

2. **Embedding Competition**: Piece embeddings and positional embeddings are additively combined, potentially competing for representational space. The consistently lower norm of piece embeddings compared to positional embeddings may contribute to tactical blunders (especially free captures).

3. **Scale Limitations**: Current 100.7M parameter model significantly outperforms the smaller 86.6M variant, suggesting benefits from further scaling.

4. **RL Training Instability**: Self-play RL proved challenging due to:
   - Gradient norm explosion
   - Noisy reward signals from sparse terminal rewards
   - Complex action space (1,969 possible moves)
   - Not optimized hyperparameters

### Architecture Insights

The custom transformer design shows promise but reveals areas for improvement:

- Model benefits from depth more than width
- Model scaling appears more beneficial than initially expected

## Models Released

- **[ChessFormer-SL](https://huggingface.co/kaupane/ChessFormer-SL)**: Supervised learning checkpoint (~130k steps)
- **[ChessFormer-RL](https://huggingface.co/kaupane/ChessFormer-RL)**: RL initialization checkpoint (~50k steps)

## Usage Example

```python
import torch
from model import ChessFormerModel
from huggingface_hub import hf_hub_download

# Load model
model = ChessFormerModel.from_pretrained("kaupane/ChessFormer-SL")
model.eval()

# Analyze position
fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
repetitions = torch.tensor([1])

with torch.no_grad():
    move_logits, position_value = model(fens, repetitions)
```

### With Chess Engine Interface

```python
from engine import Engine, ChessformerConfig
import chess

# Create engine
config = ChessformerConfig(
    chessformer=model,
    temperature=0.5,
    depth=2  # Enable search enhancement
)
engine = Engine(type="chessformer", chessformer_config=config)

# Play move
board = chess.Board()
move_uci, value = engine.move(board)
print(f"Suggested move: {move_uci}, Value: {value:.3f}")
```

## Contributing

This project seeks to set the ball rolling. Contributions addressing the identified limitations are welcome, particularly:

- RL training improvements
- Alternative tokenization strategies  
- Scaling experiments
- Performance benchmarking

*ChessFormer demonstrates both the potential and current limitations of training chess models without traditional search. While not achieving competitive play strength, it's an interesting project, and helped me learn alot about transformer models and reinforcement learning.*
