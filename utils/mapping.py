from typing import List, Dict, Tuple, Set

# --- Constants --- #
MAX_HALFMOVES = 128 # cap for embedding table size
MAX_FULLMOVES = 256 # cap for embedding table size

# --- Helper Mappings --- #
PIECE_TO_IDX: Dict[str, int] = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    '.': 12
}
IDX_TO_PIECE: Dict[int, str] = {v: k for k, v in PIECE_TO_IDX.items()}
EMPTY_SQ_IDX = PIECE_TO_IDX['.']
# Map algebraic square notation (e.g., 'a1', 'h8') to 0-63 index
# a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
SQUARE_TO_IDX: Dict[str, int] = {
    f"{file}{rank}": (rank - 1) * 8 + (ord(file) - ord('a'))
    for rank in range(1, 9)
    for file in 'abcdefgh'
}
IDX_TO_SQUARE: Dict[int, str] = {v: k for k, v in SQUARE_TO_IDX.items()}



# --- Coordinate and Notation Helpers ---

# Precompute maps for efficiency
_IDX_TO_COORDS: Dict[int, Tuple[int, int]] = {i: (i // 8, i % 8) for i in range(64)} # (rank, file) 0-7
_COORDS_TO_IDX: Dict[Tuple[int, int], int] = {v: k for k, v in _IDX_TO_COORDS.items()}
_IDX_TO_ALG: Dict[int, str] = {
    i: f"{chr(ord('a') + file)}{rank + 1}"
    for i, (rank, file) in _IDX_TO_COORDS.items()
}
_ALG_TO_IDX: Dict[str, int] = {v: k for k, v in _IDX_TO_ALG.items()}

def _coords_to_alg(r: int, f: int) -> str:
    """Converts 0-indexed (rank, file) to algebraic notation."""
    if 0 <= r < 8 and 0 <= f < 8:
        return f"{chr(ord('a') + f)}{r + 1}"
    # This should not happen with valid indices, but good for safety
    raise ValueError(f"Invalid coordinates: ({r}, {f})")

def generate_structurally_valid_move_map() -> Dict[str, int]:
    """
    Generates a dictionary mapping chess moves that are geometrically possible
    by *some* standard piece (K, Q, R, B, N, or P) to unique integer indices.
    It excludes moves that are structurally impossible for any piece to make
    in one turn (e.g., a1->h5 for non-knight).

    Includes standard UCI promotions (e.g., "e7e8q"), replacing the
    corresponding simple pawn move to the final rank (e.g., "e7e8").
    This is based purely on piece movement geometry, not the current board state.

    Returns:
        Dict[str, int]: A map from the valid UCI move string to a unique
                        integer index (0 to N-1). The size N is expected
                        to be around 1800-1900.
    """
    valid_moves: Set[str] = set()
    # Keep track of base moves (like 'e7e8') that are replaced by promotions
    # according to UCI standard.
    promo_base_moves_to_exclude: Set[str] = set()

    # 1. Generate all geometrically possible non-promotion moves
    for from_idx in range(64):
        from_r, from_f = _IDX_TO_COORDS[from_idx]
        from_alg = _IDX_TO_ALG[from_idx]

        for to_idx in range(64):
            if from_idx == to_idx:
                continue

            to_r, to_f = _IDX_TO_COORDS[to_idx]
            to_alg = _IDX_TO_ALG[to_idx]
            dr, df = to_r - from_r, to_f - from_f
            abs_dr, abs_df = abs(dr), abs(df)

            # Check if the geometry matches any standard piece movement
            # Note: Queen moves are covered by Rook + Bishop checks.
            # Note: Pawn single pushes/captures are covered by King/Rook/Bishop geometry.
            # Note: Pawn double pushes are covered by Rook geometry.
            is_king_move = max(abs_dr, abs_df) == 1
            is_knight_move = (abs_dr == 2 and abs_df == 1) or (abs_dr == 1 and abs_df == 2)
            is_rook_move = dr == 0 or df == 0 # Includes King horiz/vert & pawn double push
            is_bishop_move = abs_dr == abs_df # Includes King diagonal & pawn capture/push

            if is_king_move or is_knight_move or is_rook_move or is_bishop_move:
                 uci_move = f"{from_alg}{to_alg}"
                 valid_moves.add(uci_move)


    # 2. Generate promotion moves explicitly and mark base moves for exclusion
    promo_pieces = ['q', 'r', 'b', 'n']
    for from_f in range(8):
        # White promotions (from rank 7 (idx 6) to rank 8 (idx 7))
        from_r_w, to_r_w = 6, 7
        if from_r_w != 7: # Ensure we are on the correct rank before promotion
            from_alg_w = _coords_to_alg(from_r_w, from_f)
            # Possible destinations: push (df=0), capture left (df=-1), capture right (df=1)
            for df in [-1, 0, 1]:
                to_f_w = from_f + df
                if 0 <= to_f_w < 8:
                    to_alg_w = _coords_to_alg(to_r_w, to_f_w)
                    base_move = f"{from_alg_w}{to_alg_w}"
                    #promo_base_moves_to_exclude.add(base_move) # Mark e.g. "e7e8" for exclusion
                    for p in promo_pieces:
                        valid_moves.add(f"{base_move}{p}") # Add e.g. "e7e8q"

        # Black promotions (from rank 2 (idx 1) to rank 1 (idx 0))
        from_r_b, to_r_b = 1, 0
        if from_r_b != 0: # Ensure we are on the correct rank before promotion
            from_alg_b = _coords_to_alg(from_r_b, from_f)
            # Possible destinations: push (df=0), capture left (df=-1), capture right (df=1)
            for df in [-1, 0, 1]:
                to_f_b = from_f + df
                if 0 <= to_f_b < 8:
                    to_alg_b = _coords_to_alg(to_r_b, to_f_b)
                    base_move = f"{from_alg_b}{to_alg_b}"
                    #promo_base_moves_to_exclude.add(base_move) # Mark e.g. "e2e1" for exclusion
                    for p in promo_pieces:
                        valid_moves.add(f"{base_move}{p}") # Add e.g. "e2e1q"

    # 3. Remove the base moves that were replaced by promotions
    final_valid_moves = valid_moves - promo_base_moves_to_exclude

    # 4. Add draw claim
    final_valid_moves.add("<claim_draw>")

    # 5. Create the final map with sorted keys for deterministic indices
    sorted_moves = sorted(list(final_valid_moves))
    move_map = {move: i for i, move in enumerate(sorted_moves)}

    # Optional: Print the number of moves found for verification
    # print(f"Generated {len(move_map)} structurally valid unique UCI moves.")

    return move_map


UCI_MOVE_TO_IDX = generate_structurally_valid_move_map()
IDX_TO_UCI_MOVE = {v:k for k,v in UCI_MOVE_TO_IDX.items()}