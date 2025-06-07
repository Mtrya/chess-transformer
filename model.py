import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from huggingface_hub import PyTorchModelHubMixin

from utils import MAX_HALFMOVES, MAX_FULLMOVES, EMPTY_SQ_IDX, PIECE_TO_IDX, SQUARE_TO_IDX, IDX_TO_UCI_MOVE

# --- Tokenizer --- #
class FENTokenizer(nn.Module):
    """Convert FEN (and repetitions) to a sequence of tokens"""
    def __init__(self, hidden_size,dtype):
        super().__init__()

        self.side_embed = nn.Embedding(2,hidden_size,dtype=dtype) # black/white embedding

        self.castling_embed_k = nn.Parameter(torch.randn(1,1,hidden_size,dtype=dtype))
        self.castling_embed_q = nn.Parameter(torch.randn(1,1,hidden_size,dtype=dtype))
        self.castling_embed_K = nn.Parameter(torch.randn(1,1,hidden_size,dtype=dtype))
        self.castling_embed_Q = nn.Parameter(torch.randn(1,1,hidden_size,dtype=dtype))
        self.no_castling_embed = nn.Parameter(torch.randn(1,1,hidden_size,dtype=dtype))

        self.piece_embed = nn.Embedding(13,hidden_size,dtype=dtype) # 6 for white pieces, 6 for black pieces, 1 for empty

        self.no_en_passant_embed = nn.Parameter(torch.randn(1,1,hidden_size,dtype=dtype)) # use positional embed for the target square, or a special one for '-'

        self.half_move_embed = nn.Embedding(MAX_HALFMOVES,hidden_size,dtype=dtype)

        self.full_move_embed = nn.Embedding(MAX_FULLMOVES,hidden_size,dtype=dtype)

        self.repetition_embed = nn.Embedding(3,hidden_size,dtype=dtype)
        
        self.pos_embed = nn.Embedding(64,hidden_size,dtype=dtype) # positional embedding

    def _parse_fen_string(self, fen_str: str) -> Dict:
        parts = fen_str.split()
        if len(parts) != 6:
            raise ValueError(f"Invalid FEN string: {fen_str}. Expected 6 fields")
        return {
            "piece_placement": parts[0],
            "side_to_move": parts[1],
            "castling": parts[2],
            "en_passant": parts[3],
            "halfmove_clock": parts[4],
            "fullmove_number": parts[5],
        }

    def forward(self, fen_list: List[str], repetitions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fen: List of fen strings
        
        Returns:
            torch tensor of shape (n_fen,73,hidden_size) where 73 tokens consists of:
                64 piece tokens (fen's first field) +
                1 which-side-to-move token (fen's second field) +
                4 casting rights tokens (fen's third field) + 
                1 en-passant target token (fen's fourth field) + 
                1 half move clock token (fen's fifth field) +
                1 full move number token (fen's fifth field) +
                1 repetition count token (repetitions input)
        """
        batch_size = len(fen_list)
        assert batch_size == repetitions.shape[0]
        assert len(repetitions.size()) == 1
        batch_tokens = []
        device = self.side_embed.weight.device

        # Precompute all square indices
        square_indices = torch.arange(64, device=device)
        all_pos_embeds = self.pos_embed(square_indices) # (64,D)

        for fen_str in fen_list:
            parsed_fen = self._parse_fen_string(fen_str)
            tokens = []

            # --- 1. Piece Placement (64 tokens) ---
            piece_indices = torch.full((64,), EMPTY_SQ_IDX, dtype=torch.long, device=device)
            current_rank = 7 # Start from rank 8
            current_file = 0 # Start from file 'a'
            for char in parsed_fen["piece_placement"]:
                if char == '/':
                    current_rank -= 1
                    current_file = 0
                elif char.isdigit():
                    current_file += int(char)
                elif char in PIECE_TO_IDX:
                    sq_idx = current_rank * 8 + current_file
                    if 0 <= sq_idx < 64:
                         piece_indices[sq_idx] = PIECE_TO_IDX[char]
                    else:
                         raise ValueError(f"Invalid FEN piece placement: {parsed_fen['piece_placement']}")
                    current_file += 1
                else:
                     raise ValueError(f"Invalid character in FEN piece placement: {char}")

            piece_embeds = self.piece_embed(piece_indices) # (64, D)
            # Add positional embeddings
            board_tokens = piece_embeds + all_pos_embeds # (64, D)
            tokens.append(board_tokens)

            # --- 2. Side to Move (1 token) ---
            side_idx = 0 if parsed_fen["side_to_move"] == 'w' else 1
            side_token = self.side_embed(torch.tensor(side_idx, device=device)).unsqueeze(0) # (1, D)
            tokens.append(side_token)

            # --- 3. Castling Rights (4 tokens) ---
            castling_str = parsed_fen["castling"]
            castling_tokens = torch.cat([
                self.castling_embed_K if 'K' in castling_str else self.no_castling_embed.expand(1, 1, -1),
                self.castling_embed_Q if 'Q' in castling_str else self.no_castling_embed.expand(1, 1, -1),
                self.castling_embed_k if 'k' in castling_str else self.no_castling_embed.expand(1, 1, -1),
                self.castling_embed_q if 'q' in castling_str else self.no_castling_embed.expand(1, 1, -1)
            ], dim=1).squeeze(0) # (4, D)
            tokens.append(castling_tokens)

            # --- 4. En Passant Target (1 token) ---
            en_passant_str = parsed_fen["en_passant"]
            if en_passant_str == '-':
                en_passant_token = self.no_en_passant_embed.squeeze(0) # (1, D)
            else:
                if en_passant_str in SQUARE_TO_IDX:
                    sq_idx = SQUARE_TO_IDX[en_passant_str]
                    en_passant_token = self.pos_embed(torch.tensor(sq_idx, device=device)).unsqueeze(0) # (1, D)
                else:
                    raise ValueError(f"Invalid en passant square: {en_passant_str}")
            tokens.append(en_passant_token)

            # --- 5. Half Move Clock (1 token) ---
            try:
                half_move_int = int(parsed_fen["halfmove_clock"])
            except ValueError:
                 raise ValueError(f"Invalid halfmove clock value: {parsed_fen['halfmove_clock']}")
            # Clamp value before embedding lookup
            half_move_clamped = torch.clamp(torch.tensor(half_move_int, device=device), 0, MAX_HALFMOVES - 1)
            half_move_token = self.half_move_embed(half_move_clamped).unsqueeze(0) # (1, D)
            tokens.append(half_move_token)

            # --- 6. Full Move Number (1 token) ---
            try:
                full_move_int = int(parsed_fen["fullmove_number"])
            except ValueError:
                 raise ValueError(f"Invalid fullmove number value: {parsed_fen['fullmove_number']}")
             # Clamp value (min 1 for full moves) before embedding lookup (adjusting for 0-based index)
            full_move_clamped = torch.clamp(torch.tensor(full_move_int, device=device), 1, MAX_FULLMOVES) - 1
            full_move_token = self.full_move_embed(full_move_clamped).unsqueeze(0) # (1, D)
            tokens.append(full_move_token)

            # Concatenate all tokens for this FEN string
            # Shapes: (64, D), (1, D), (4, D), (1, D), (1, D), (1, D) -> Total 72 tokens
            fen_embedding = torch.cat(tokens, dim=0) # (72, D)
            batch_tokens.append(fen_embedding)

        # Stack into a batch
        batch_tokens = torch.stack(batch_tokens, dim=0) # (B,72,D)

        # ---7. Repetition Count (1 token) ---
        repetitions = repetitions - 1 # from 1~3 to 0~2
        repetitions = torch.clamp(repetitions,0,2) # if repetition count >3 but no player claimed a draw, it will be treated as 3 repetitions
        repetition_tokens = self.repetition_embed(repetitions) # (B,D)
        repetition_tokens = repetition_tokens.unsqueeze(1) # (B,1,D)

        return torch.cat([batch_tokens,repetition_tokens], dim=1) # (B, 73, D)

# --- Helper Modules --- #
class SwiGLUFFN(nn.Module):
    def __init__(self,
                 d_model, 
                 dim_feedforward,
                 dropout: float,
                 bias_up: bool=False,
                 bias_gate: bool=False,
                 bias_down: bool=True,
                 dtype=None):
        super().__init__()
        self.up_proj = nn.Linear(d_model,dim_feedforward,bias=bias_up,dtype=dtype)
        self.gate_proj = nn.Linear(d_model,dim_feedforward,bias=bias_gate,dtype=dtype)
        self.down_proj = nn.Linear(dim_feedforward,d_model,bias=bias_down,dtype=dtype)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x) * self.dropout(nn.functional.silu(self.gate_proj(x)))
        return self.down_proj(x)

class TransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with RMSNorm and SwiGLUFFN"""
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 batch_first: bool=True,
                 norm_first: bool=False,
                 dtype=None):
        super().__init__()
        self.norm_first = norm_first

        self.norm1 = nn.RMSNorm(d_model,dtype=dtype)
        self.dropout_sa = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=False,
            batch_first=batch_first,
            dtype=dtype
        )

        self.norm2 = nn.RMSNorm(d_model,dtype=dtype)
        self.dropout_ff = nn.Dropout(dropout)
        self.mlp = SwiGLUFFN(
            d_model,
            dim_feedforward,
            dropout=dropout,
            bias_up=False,
            bias_gate=False,
            bias_down=True,
            dtype=dtype
            )

    def forward(self, x, return_attention=False):
        if self.norm_first:
            if return_attention:
                x_norm = self.norm1(x)
                attn_output, attn_weights = self._sa_block(x_norm,return_attention=True)
                x = x + attn_output
                x = x + self._ff_block(self.norm2(x))
                return x, attn_weights
            else:
                x = x + self._sa_block(self.norm1(x))
                x = x + self._ff_block(self.norm2(x))
                return x
        else:
            if return_attention:
                attn_output, attn_weights = self._sa_block(x, return_attention=True)
                x = self.norm1(x + attn_output)
                x = self.norm2(x + self._ff_block(x))
                return x, attn_weights
            else:
                x = self.norm1(x + self._sa_block(x))
                x = self.norm2(x + self._ff_block(x))
                return x
    
    def _sa_block(self, x, return_attention=False):
        if return_attention:
            attn_output, attn_weights = self.self_attn(x,x,x,need_weights=True,average_attn_weights=False)
            return self.dropout_sa(attn_output), attn_weights
        else:
            x = self.self_attn(x,x,x)[0]
            return self.dropout_sa(x)
    
    def _ff_block(self,x):
        x = self.mlp(x)
        return self.dropout_ff(x)
    nn.TransformerEncoderLayer

# --- Model Arch --- #
class ChessFormerModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 num_blocks,
                 hidden_size,
                 intermediate_size,
                 num_heads,
                 dropout: float=0.00,
                 possible_moves: int=len(IDX_TO_UCI_MOVE), # 1969 structurally valid moves
                 dtype=None):
        super().__init__()
        self.fen_tokenizer = FENTokenizer(hidden_size,dtype=dtype)

        self.act_token = nn.Parameter(torch.randn((1,1,hidden_size),dtype=dtype) * 0.02)
        self.val_token = nn.Parameter(torch.randn((1,1,hidden_size),dtype=dtype) * 0.02)

        self.act_proj = nn.Linear(hidden_size,possible_moves,dtype=dtype)
        self.val_proj = nn.Linear(hidden_size,1,dtype=dtype)

        self.blocks = nn.ModuleList(
            TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                dtype=dtype               
            ) for _ in range(num_blocks)
        )
        self.dtype=dtype
        self.possible_moves = possible_moves

        self.final_norm = nn.RMSNorm(hidden_size)

        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.weight, 0.0)
            elif isinstance(m, nn.RMSNorm):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1.0)

        tokenizer_params = dict(self.fen_tokenizer.named_parameters())

        params_to_init = [
            self.act_token, self.val_token,
            tokenizer_params.get('castling_embed_k'), tokenizer_params.get('castling_embed_q'),
            tokenizer_params.get('castling_embed_K'), tokenizer_params.get('castling_embed_Q'),
            tokenizer_params.get('no_castling_embed'), tokenizer_params.get('no_en_passant_embed')
        ]

        for param in params_to_init:
            if param is not None and param.requires_grad:
                nn.init.normal_(param, std=0.02)


    def forward(self, fen: List[str], repetitions: torch.Tensor, return_attention: bool=False) -> torch.Tensor:
        x = self.fen_tokenizer(fen,repetitions) # (B,73,D), pos embed are added here
        bs = x.shape[0]
        x = torch.cat([x,self.act_token.expand(bs,-1,-1),self.val_token.expand(bs,-1,-1)],dim=1) # (B,75,D)

        attention_maps = [] if return_attention else None

        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x = block(x)

        x = self.final_norm(x)

        act = x[:,-2,:]
        val = x[:,-1,:]
        act_logits = self.act_proj(act) # (B,1969)
        val = self.val_proj(val) # (B,1)

        if return_attention:
            return act_logits, val.squeeze(1), attention_maps
        else:
            return act_logits, val.squeeze(1)

def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model_config = checkpoint["model_config"]
    model = ChessFormerModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

if __name__ == "__main__":
    checkpoint = torch.load("./ckpts/chessformer-sl_06.pth",map_location=torch.device("cpu"))
    model = ChessFormerModel(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    model.push_to_hub("kaupane/ChessFormer-SL")