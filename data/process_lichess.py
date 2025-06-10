import chess
import chess.engine
from datasets import load_dataset
from typing import List, Optional, Tuple, Dict
import math
import csv
import os
import multiprocessing
from functools import partial
import time
import random
from tqdm import tqdm
import collections

# Build a dataset from "nsarrazin/lichess-games-2023-01" dataset consisting of move series.
# Columns: 
# 1. fen: str (FEN notation)   
# 2. repetition_count: int  (starts from 1)
# 3. best_moves: str (best move given by stockfish, including a special "<claim_draw>") 
# 4. scores: float (normalized score giben by stockfish)
# 5. phase: str ("opening", "middlegame" or "endgame")
# 6. valid_moves: List[str] (all valid moves in uci representation, including a special "<claim_draw>" if board.can_claim_draw() is true)

# Stockfish does not claim draw by itself, we consider <claim_draw> is the best move if board.can_claim_draw() is true and score is below a certain threshold
# Game phases are determined using heuristic rules which are not necessarily accurate
# Uses rather shallow depth (18 for "depth18" split and 27 for "depth27" split) due to not optimized script and hardware constraints

# Follow the example below 'if __name__ == "__main__":' to use this script to annotate other similar datasets.


def get_score_and_best_move(info: chess.engine.InfoDict, can_claim_draw: bool):
    """
    Analyzes Stockfish info to get the best move and a score.
    Returns None if analysis info is incomplete or invalid
    """
    loss_threshold = -0.4

    score_obj = info.get("score")
    if score_obj is None or info.get("pv") is None or not info.get("pv"):
        # Invalid analysis result
        return None
    
    pv = info["pv"]
    pov_score = score_obj.pov(chess.WHITE)

    cp = None
    if pov_score.is_mate():
        mate_score = pov_score.mate()
        cp = 10000.0 if mate_score > 0 else -10000.0
        relative_score = score_obj.relative
        if relative_score.is_mate():
            cp = 10000.0 if relative_score.mate() > 0 else -10000.0
        else:
            if relative_score.cp is not None:
                cp = float(relative_score.cp)
            else:
                return None

    elif pov_score.cp is not None:
        relative_score = score_obj.relative
        if relative_score.cp is not None:
            cp = float(relative_score.cp)
        else:
            return None

    else:
        return None
    
    if cp is not None:
        normalized_score = 2 / (1+math.exp(-0.0037*cp)) - 1
    else:
        return None

    if can_claim_draw and normalized_score < loss_threshold:
        best_move_uci = "<claim_draw>"
    else:
        best_move_uci = pv[0].uci()

    return best_move_uci, normalized_score

def get_phase(board: chess.Board) -> str:
    """Determine the phase of a chess board (opening, middlegame or endgame) using a few heuristic rules"""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    total_material = 0
    for piece_type, value in piece_values.items():
        count_white = len(board.pieces(piece_type, chess.WHITE))
        count_black = len(board.pieces(piece_type, chess.BLACK))
        total_material += (count_white + count_black) * value

    #print(f"Total material: {total_material}")
    
    has_white_queen = bool(board.pieces(chess.QUEEN,chess.WHITE))
    has_black_queen = bool(board.pieces(chess.QUEEN,chess.BLACK))

    if not has_black_queen and not has_white_queen:
        return "endgame"
    
    if total_material >= 68:
        return "opening"
    elif total_material <= 34:
        return "endgame"
    else:
        return "middlegame"

def process_game(game_moves: List[str], stockfish_path: str, depth: int) -> List[Dict]:
    """Processes a single game, analyzes positions, and return data rows"""
    board = chess.Board()
    engine = None
    game_data = []

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1})

        for move_uci in game_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    break # Stop processing this game
                board.push(move)
            except ValueError:
                break # Stop processing this game

            # Get valid moves
            valid_moves_list = [m.uci() for m in board.legal_moves]
            valid_moves_str = " ".join(valid_moves_list)

            repetition_count = 1
            if board.is_repetition(3):
                repetition_count = 3
            elif board.is_repetition(2):
                repetition_count = 2

            # Analyze with stockfish
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
                print(f"Stockfish analysis error: {e}. Skipping rest of game.")
                break

            # Check draw claim
            can_claim_draw = board.can_claim_draw()
            
            # Get best move and score
            analysis_result = get_score_and_best_move(info, can_claim_draw)

            if analysis_result:
                best_move, score = analysis_result
                phase = get_phase(board)

                if board.fullmove_number >= 12:
                    game_data.append({
                        "fen": board.fen(),
                        "repetition_count": repetition_count,
                        "best_move": best_move,
                        "score": score,
                        "phase": phase,
                        "valid_moves": valid_moves_str
                    })


    except Exception as e:
        print(f"Error processing game: {e}")
    finally:
        if engine:
            engine.quit()

    return game_data

def convert_dataset(dataset_name: str, 
                    output_csv: str="./data/processed_lichess/lichess_annotated_train.csv", 
                    stockfish_path: str="/usr/games/stockfish", 
                    depth: int=20, 
                    split:str="train",
                    num_workers: Optional[int]=None,
                    batch_size: int=256,
                    target_positions: int=25165824, # 24M
                    ):
    """
    Loads a dataset, analyzes games using Stockfish in parallel, and saves results to CSV.

    Args:
        dataset_name: Name of the Hugging Face dataset (e.g., "nsarrazin/lichess-games-2014-11").
        output_csv: Path to save the resulting CSV file.
        stockfish_path: Path to the Stockfish executable.
        depth: Stockfish analysis depth.
        split: Dataset split to process (e.g., "train", "test").
        num_workers: Number of worker processes. Defaults to os.cpu_count().
        batch_size: Number of positions to accumulate before writing to CSV.
        target_positions: Target number of positions to analyze before stopping.
    """
    print(f"Loading dataset '{dataset_name}', split '{split}'...")
    ds = load_dataset(dataset_name,split=split,streaming=False)
    
    game_moves_list = [sample["moves"] for sample in ds] # Extract moves for all games first

    if not game_moves_list:
        print("No games found in the specified dataset split/limit.")
        return

    if num_workers is None:
        num_workers = os.cpu_count()
        print(f"Using {num_workers} worker processes.")

    # Create a partial function with fixed arguments for stockfish_path and depth
    worker_func = partial(process_game, stockfish_path=stockfish_path, depth=depth)

    print(f"Starting analysis of {len(game_moves_list)} games with depth {depth}...")
    start_time = time.time()

    fieldnames = ["fen", "repetition_count", "best_move", "score", "phase", "valid_moves"]
    initial_positions = 0
    file_exists = os.path.exists(output_csv)
    write_header = not file_exists

    if file_exists:
        try:
            import pandas as pd
            df_iter = pd.read_csv(output_csv,chunksize=100000,usecols={fieldnames[0]})
            for chunk in df_iter:
                initial_positions += len(chunk)

            if initial_positions > 0:
                write_header = False
            else:
                write_header = True
        except pd.error.EmptyDataError:
            write_header = True
        except Exception as e:
            raise

    if initial_positions >= target_positions:
        return
    
    start_time = time.time()

    if write_header:
        try:
            with open(output_csv, 'w', newline='',encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
                writer.writeheader()
        except IOError as e:
            print(f"Error creating CSV file: {e}")
            return
    
    current_batch = []
    total_positions = initial_positions
    pbar = tqdm(total=target_positions,initial=initial_positions,desc='Processing games')

    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for potentially better memory usage and seeing progress
        results_iterator = pool.imap_unordered(worker_func, game_moves_list, chunksize=16)

        for game_result in results_iterator:
            current_batch.extend(game_result)

            if len(current_batch) >= batch_size:
                batch_to_write = current_batch[:batch_size]
                random.shuffle(batch_to_write)

                total_positions += batch_size
                pbar.update(batch_size)
                try:
                    with open(output_csv, 'a', newline='',encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
                        writer.writerows(batch_to_write)
                except IOError as e:
                    print(f"Error writing to CSV file")

                if len(current_batch) == batch_size:
                    current_batch = []
                else:
                    current_batch = current_batch[batch_size:]

            if total_positions >= target_positions:
                break

        pbar.close()

    end_time = time.time()
    print(f"Analysis finished. Total time: {end_time - start_time:.2f} seconds.")
    print(f"Collected {total_positions} positions.")
    print(f"Results written to {output_csv}")

def upload_dataset_to_huggingface(
    csv_files: Dict[str, str],  # Dict mapping split names to file paths
    repo_id: str,  # Hugging Face repo ID (e.g. "username/dataset-name")
    token: Optional[str] = None,  # Hugging Face token
    private: bool = False,  # Whether the dataset should be private
    description: Optional[str] = None,  # Description for the dataset
    max_shard_size: Optional[str] = "500MB",  # Max shard size for large datasets
):
    """
    Uploads large CSV files as splits of a dataset to Hugging Face.
    
    Args:
        csv_files: Dictionary mapping split names to file paths
        repo_id: Hugging Face repository ID (e.g. "username/dataset-name")
        token: Hugging Face token for authentication
        private: Whether the dataset should be private
        description: Description for the dataset
        max_shard_size: Maximum size of dataset shards (e.g. "500MB")
    """
    from datasets import load_dataset
    from huggingface_hub import create_repo, HfApi
    import os
    import time
    
    print(f"Preparing to upload dataset to {repo_id}...")
    
    # Create the repository if it doesn't exist
    create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        repo_type="dataset",
        exist_ok=True,
    )
    
    # Create data_files dictionary
    data_files = {}
    for split_name, file_path in csv_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        print(f"Verified {split_name} split file: {file_path}")
        data_files[split_name] = file_path
    
    # Load the dataset
    print(f"Loading dataset from CSV files...")
    start_time = time.time()
    dataset = load_dataset('csv', data_files=data_files)
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds")
    
    # Print dataset info
    for split_name, split_dataset in dataset.items():
        print(f"Split '{split_name}': {len(split_dataset)} examples")
    
    # Push to the hub
    print(f"Pushing dataset to {repo_id}...")
    print(f"This may take a while for large datasets. Using max_shard_size={max_shard_size}")
    push_start_time = time.time()
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        max_shard_size=max_shard_size,
    )
    push_time = time.time() - push_start_time
    print(f"Dataset pushed in {push_time:.2f} seconds")
    
    # Update dataset metadata if description is provided
    if description:
        api = HfApi(token=token)
        # Create/update README.md
        readme_content = f"# {repo_id.split('/')[-1]}\n\n{description}"
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    print(f"Successfully uploaded dataset to {repo_id}")
    print(f"To load the dataset, use: dataset = load_dataset('{repo_id}')")

def upload_additional_split_to_huggingface(
    csv_file_path: str,
    split_name: str,
    repo_id: str,
    token: Optional[str] = None,
    max_shard_size: Optional[str] = "500MB",
    description_update: Optional[str] = None
):
    """
    Uploads an additional split to an existing Hugging Face dataset without modifying existing splits.
    
    Args:
        csv_file_path: Path to the CSV file containing the new split data
        split_name: Name of the new split (e.g., "test", "extra_train")
        repo_id: Hugging Face repository ID (e.g. "username/dataset-name")
        token: Hugging Face token for authentication
        max_shard_size: Maximum size of dataset shards (e.g. "500MB")
        description_update: Additional text to append to the dataset description
    """
    from datasets import load_dataset, Dataset
    from huggingface_hub import HfApi
    import os
    import time
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    print(f"Loading new split '{split_name}' from {csv_file_path}...")
    start_time = time.time()
    
    # Load only the new split
    new_split = load_dataset('csv', data_files={split_name: csv_file_path})
    load_time = time.time() - start_time
    print(f"New split loaded in {load_time:.2f} seconds")
    print(f"Split '{split_name}': {len(new_split[split_name])} examples")
    
    # Push only the new split to the hub
    print(f"Pushing new split '{split_name}' to {repo_id}...")
    print(f"This may take a while for large splits. Using max_shard_size={max_shard_size}")
    push_start_time = time.time()
    
    # The config parameter makes sure we don't overwrite existing splits
    new_split.push_to_hub(
        repo_id=repo_id,
        token=token,
        config_name=split_name,  # This creates a config just for this split
        max_shard_size=max_shard_size,
    )
    
    push_time = time.time() - push_start_time
    print(f"New split pushed in {push_time:.2f} seconds")
    
    # Update dataset description if requested
    if description_update:
        api = HfApi(token=token)
        
        # Try to fetch existing README
        try:
            readme_content = api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="README.md"
            )
            with open(readme_content, 'r') as f:
                current_readme = f.read()
            
            # Append the new description
            new_readme = current_readme + f"\n\n## Update: New {split_name} Split\n\n{description_update}"
            
        except:
            # If README doesn't exist, create a new one
            new_readme = f"# {repo_id.split('/')[-1]}\n\n## {split_name.capitalize()} Split\n\n{description_update}"
        
        # Upload updated README
        api.upload_file(
            path_or_fileobj=new_readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    print(f"Successfully added '{split_name}' split to {repo_id}")
    print(f"To load the full dataset with all splits, use: dataset = load_dataset('{repo_id}')")
    print(f"To load just this new split, use: dataset = load_dataset('{repo_id}', '{split_name}')")

def write_dedup_loose(
    dataset, 
    output_csv: str, 
    batch_size: int = 1344, 
    fen_queue_max_size: int = 131072, # Max FENs to keep in memory for deduplication
    dedup_prob: float = 0.7,         # Probability of dropping a duplicate
    fieldnames: List[str] = None     # Optional: list of column names for the CSV
):
    """
    Performs a loose deduplication on a dataset and writes unique/lucky samples to a CSV.
    Maintains a FEN queue to check for recent duplicates and drops them with a given probability.

    Args:
        dataset: The input dataset object (e.g., loaded from datasets.load_dataset).
        output_csv: Path to save the resulting CSV file.
        batch_size: Number of samples to accumulate before writing to CSV.
        fen_queue_max_size: Maximum number of FENs to keep in the in-memory queue for deduplication.
        dedup_prob: The probability (0.0 to 1.0) of dropping a sample if its FEN is found in the queue.
        fieldnames: A list of column names to use for the CSV header. If None, it tries to infer from the first sample.
    """
    print(f"Starting loose deduplication and writing to {output_csv}...")
    print(f"Note: '{output_csv}' will be overwritten.")
    
    fen_queue = collections.deque(maxlen=fen_queue_max_size)
    batch_to_write = []
    total_written = 0

    # Determine fieldnames if not provided
    if fieldnames is None:
        if dataset and len(dataset) > 0:
            # Assuming the first sample has all the keys
            fieldnames = list(dataset[0].keys())
        else:
            # Fallback to default fieldnames if dataset is empty or no sample to infer from
            fieldnames = ["fen", "repetition_count", "best_move", "score", "phase", "valid_moves"]
            print(f"Warning: Could not infer fieldnames from dataset. Using default: {fieldnames}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Write header if file doesn't exist or is empty
    file_exists = os.path.exists(output_csv)
    write_header = True

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for i, sample in tqdm(enumerate(dataset), desc="Deduplicating and writing", total=len(dataset)):
                fen = sample.get("fen")
                if not fen:
                    continue # Skip samples without a FEN

                is_duplicate_in_queue = fen in fen_queue
                
                # Decide whether to drop the sample
                should_drop = False
                if is_duplicate_in_queue:
                    if random.random() < dedup_prob:
                        should_drop = True
                
                if not should_drop:
                    # Sample is kept (either unique or lucky duplicate)
                    batch_to_write.append(sample)
                    
                    # If it was truly unique (not in queue), add its FEN to the queue
                    # This ensures that even "lucky" duplicates don't get added to the queue again,
                    # only truly new FENs (within the queue's scope) are added.
                    if not is_duplicate_in_queue:
                        fen_queue.append(fen)

                    if len(batch_to_write) >= batch_size:
                        writer.writerows(batch_to_write)
                        total_written += len(batch_to_write)
                        batch_to_write = []

            # Write any remaining samples in the batch
            #if batch_to_write:
            #    writer.writerows(batch_to_write)
            #    total_written += len(batch_to_write)

    except Exception as e:
        print(f"An error occurred during deduplication: {e}")
    
    print(f"Loose deduplication finished. Total positions written: {total_written}")
    print(f"Results written to {output_csv}")

def analyze_single_fen(fen: str, depth: int=18, output_csv: Optional[str]=None):
    board = chess.Board(fen)
    engine = None
    import pandas as pd

    fieldnames = ["fen","repetition_count","best_move","score","phase","valid_moves"]

    """initial_row_count = 0
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            initial_row_count = pd.read_csv(output_csv, usecols=[fieldnames[0]]).shape[0]
        except pd.errors.EmptyDataError:
            initial_row_count = 0 # File exists but is empty
        except Exception as e:
            print(f"Error reading CSV for initial count: {e}")
            initial_row_count = 0
    print(f"Initial sample count in '{output_csv}': {initial_row_count}")"""

    try:
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        
        valid_moves_list = [m.uci() for m in board.legal_moves]
        valid_moves_str = " ".join(valid_moves_list)

        info = engine.analyse(board,chess.engine.Limit(depth=depth))
        can_claim_draw = board.can_claim_draw()

        analysis_result = get_score_and_best_move(info,can_claim_draw)

        if analysis_result:
            best_move, score = analysis_result
            phase = get_phase(board)

            data_row = {
                "fen": fen,
                "repetition_count": 1,
                "best_move": best_move,
                "score": score,
                "phase": phase,
                "valid_moves": valid_moves_str
            }
            print(f"Sample: fen {data_row["fen"]}, best_move {data_row["best_move"]}")

            if output_csv is not None:
                file_exists = os.path.exists(output_csv)
                write_header = not file_exists or os.path.getsize(output_csv) == 0
                try:
                    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(data_row)
                    #print(f"Successfully analyzed FEN and appended to {output_csv}")

                    """final_row_count = 0
                    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
                        try:
                            final_row_count = pd.read_csv(output_csv, usecols=[fieldnames[0]]).shape[0]
                        except pd.errors.EmptyDataError:
                            final_row_count = 0
                        except Exception as e:
                            print(f"Error reading CSV for final count: {e}")
                            final_row_count = 0
                    print(f"Final sample count in '{output_csv}': {final_row_count}")"""
                except IOError as e:
                    print(f"Error writing to CSV file {output_csv}: {e}")
        else:
            print(f"Could not get valid analysis result for FEN: {fen}")

    except Exception as e:
        print(f"Error processing FEN {fen}: {e}")
    finally:
        if engine:
            engine.quit()



if __name__ == "__main__":
    dataset_name = "nsarrazin/lichess-games-2023-01"
    depth27_split = "train[90000:95000]"
    depth27_path = "./data/processed_lichess/lichess_annotated_depth27.csv"
    convert_dataset(dataset_name,depth=27,split=depth27_split,target_positions=65536,output_csv=depth27_path,num_workers=9)
    depth18_split = "train[:3600000]"
    depth18_path = "./data/processed_lichess/lichess_annotated_depth18.csv"
    convert_dataset(dataset_name,depth=20,split=depth18_split,target_positions=6000000,output_csv=depth18_path,num_workers=8)
    ds = load_dataset("csv",data_files="./data/processed_lichess/lichess_annotated_depth27.csv",split="train")
    write_dedup_loose(
        ds,
        "./data/processed_lichess/lichess_annotated_depth27_dedup.csv",
        batch_size=128,
        dedup_prob=1.0
    )
    ds = load_dataset("csv",data_files="./data/processed_lichess/lichess_annotated_depth18.csv",split="train")
    write_dedup_loose(
        ds,
        "./data/processed_lichess/lichess_annotated_depth18_dedup.csv",
        fen_queue_max_size=98304,
        batch_size=1200,
        dedup_prob=0.7
    )