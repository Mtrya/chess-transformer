import argparse
import collections
import csv
import itertools
import json
import multiprocessing
import os
import shutil
import sqlite3
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import chess
import chess.engine
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    load_dataset,
    load_dataset_builder,
    load_from_disk,
)
from huggingface_hub import create_repo
from tqdm import tqdm

FINAL_FEATURES = Features(
    {
        "fen": Value("string"),
        "next_move": Value("string"),
        "count": Value("int64"),
    }
)
SPLIT_HASH_DENOMINATOR = 1_000_000

_ANNOTATION_ENGINE: Optional[chess.engine.SimpleEngine] = None
_ANNOTATION_ENGINE_PATH: Optional[str] = None
_ANNOTATION_DEPTH: Optional[int] = None
_ANNOTATION_THREADS: Optional[int] = None


@dataclass
class CollapseBatchResult:
    processed_rows: int
    valid_rows: int
    total_count: int
    aggregated_rows: List[Tuple[int, str, int]]


@dataclass
class AnnotationBucketTask:
    bucket_id: int
    input_path: str
    output_path: str


@dataclass
class AnnotationBucketResult:
    bucket_id: int
    positions_seen: int
    rows_written: int
    skipped_positions: int
    error_message: Optional[str] = None


def stable_hash_int(text: str) -> int:
    return int.from_bytes(blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def bucket_for_fen(fen: str, bucket_count: int) -> int:
    return stable_hash_int(fen) % bucket_count


def split_for_fen(fen: str, val_ratio: float) -> str:
    val_threshold = int(val_ratio * SPLIT_HASH_DENOMINATOR)
    hashed = stable_hash_int(fen) % SPLIT_HASH_DENOMINATOR
    return "validation" if hashed < val_threshold else "train"


def normalize_count(raw_count: object) -> int:
    try:
        count = int(raw_count)
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def write_manifest(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_clean_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not (output_dir / "manifest.json").exists()
    ):
        raise FileExistsError(
            f"Refusing to reuse non-empty output directory without a manifest: {output_dir}. "
            "Pass --overwrite to clear it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


class FenCountSqliteManager:
    def __init__(self, bucket_dir: Path, max_open_dbs: int = 64):
        self.bucket_dir = bucket_dir
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
        self.max_open_dbs = max_open_dbs
        self._connections: "collections.OrderedDict[int, sqlite3.Connection]" = (
            collections.OrderedDict()
        )

    def _bucket_path(self, bucket_id: int) -> Path:
        return self.bucket_dir / f"bucket-{bucket_id:05d}.sqlite"

    def _configure_connection(self, conn: sqlite3.Connection):
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS counts (
                fen TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (fen)
            )
            """
        )

    def _open_connection(self, bucket_id: int) -> sqlite3.Connection:
        if bucket_id in self._connections:
            conn = self._connections.pop(bucket_id)
            self._connections[bucket_id] = conn
            return conn

        if len(self._connections) >= self.max_open_dbs:
            _, oldest = self._connections.popitem(last=False)
            oldest.commit()
            oldest.close()

        conn = sqlite3.connect(self._bucket_path(bucket_id))
        self._configure_connection(conn)
        self._connections[bucket_id] = conn
        return conn

    def write_rows(self, rows: Sequence[Tuple[int, str, int]]) -> int:
        grouped_rows: Dict[int, List[Tuple[str, int]]] = collections.defaultdict(list)

        for bucket_id, fen, count in rows:
            grouped_rows[bucket_id].append((fen, count))

        merged_rows = 0
        insert_sql = (
            "INSERT INTO counts (fen, count) VALUES (?, ?) "
            "ON CONFLICT(fen) DO UPDATE SET count = count + excluded.count"
        )
        for bucket_id, bucket_rows in grouped_rows.items():
            conn = self._open_connection(bucket_id)
            conn.executemany(insert_sql, bucket_rows)
            conn.commit()
            merged_rows += len(bucket_rows)

        return merged_rows

    def close(self):
        for conn in self._connections.values():
            conn.commit()
            conn.close()
        self._connections.clear()


def parse_source_splits(source_splits: str) -> Optional[List[str]]:
    if source_splits.strip().lower() == "all":
        return None
    splits = [part.strip() for part in source_splits.split(",") if part.strip()]
    if not splits:
        raise ValueError(
            "--source-splits must be 'all' or a comma-separated list of split names"
        )
    return splits


def resolve_dataset_splits(
    dataset_obj: Union[Dataset, DatasetDict],
    requested_splits: Optional[Sequence[str]],
) -> List[Tuple[str, Iterable[Dict]]]:
    if isinstance(dataset_obj, DatasetDict):
        if requested_splits is None:
            return [
                (split_name, dataset_obj[split_name])
                for split_name in dataset_obj.keys()
            ]

        missing_splits = [
            split_name
            for split_name in requested_splits
            if split_name not in dataset_obj
        ]
        if missing_splits:
            raise ValueError(f"Requested splits not found in dataset: {missing_splits}")
        return [
            (split_name, dataset_obj[split_name]) for split_name in requested_splits
        ]

    if requested_splits is not None and requested_splits != ["train"]:
        raise ValueError("Specific source splits require a DatasetDict source.")
    return [("train", dataset_obj)]


def iter_source_rows(
    source: str,
    requested_splits: Optional[Sequence[str]],
) -> Iterator[Dict]:
    source_path = Path(source)

    if source_path.exists():
        dataset_obj = load_from_disk(str(source_path))
        split_iterables = resolve_dataset_splits(dataset_obj, requested_splits)
        for _, split_dataset in split_iterables:
            for row in split_dataset:
                yield row
        return

    if requested_splits is None:
        dataset_obj = load_dataset(source)
        split_iterables = resolve_dataset_splits(dataset_obj, requested_splits)
        for _, split_dataset in split_iterables:
            for row in split_dataset:
                yield row
        return

    for split_name in requested_splits:
        split_dataset = load_dataset(source, split=split_name)
        for row in split_dataset:
            yield row


def batched_rows(
    rows: Iterable[Dict],
    rows_per_batch: int,
    skip_rows: int = 0,
    max_rows: Optional[int] = None,
) -> Iterator[List[Dict]]:
    skipped = 0
    yielded = 0
    batch: List[Dict] = []

    for row in rows:
        if skipped < skip_rows:
            skipped += 1
            continue

        if max_rows is not None and yielded >= max_rows:
            break

        batch.append(row)
        yielded += 1

        if len(batch) >= rows_per_batch:
            yield batch
            batch = []

    if batch:
        yield batch


def collapse_batch(rows: Sequence[Dict], bucket_count: int) -> CollapseBatchResult:
    counts: collections.Counter[str] = collections.Counter()
    valid_rows = 0
    total_count = 0

    for row in rows:
        fen = row.get("fen")
        if not isinstance(fen, str) or not fen:
            continue

        count = normalize_count(row.get("count"))
        if count <= 0:
            continue

        counts[fen] += count
        valid_rows += 1
        total_count += count

    aggregated_rows = [
        (bucket_for_fen(fen, bucket_count), fen, count) for fen, count in counts.items()
    ]

    return CollapseBatchResult(
        processed_rows=len(rows),
        valid_rows=valid_rows,
        total_count=total_count,
        aggregated_rows=aggregated_rows,
    )


def _collapse_batch_star(args) -> CollapseBatchResult:
    return collapse_batch(*args)


def open_stockfish_engine(
    stockfish_path: str, depth: int, engine_threads: int
) -> chess.engine.SimpleEngine:
    del depth
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure({"Threads": engine_threads})
    except chess.engine.EngineError:
        pass
    return engine


def init_annotation_worker(stockfish_path: str, depth: int, engine_threads: int):
    global \
        _ANNOTATION_ENGINE, \
        _ANNOTATION_ENGINE_PATH, \
        _ANNOTATION_DEPTH, \
        _ANNOTATION_THREADS
    _ANNOTATION_ENGINE_PATH = stockfish_path
    _ANNOTATION_DEPTH = depth
    _ANNOTATION_THREADS = engine_threads
    _ANNOTATION_ENGINE = open_stockfish_engine(stockfish_path, depth, engine_threads)


def close_annotation_engine():
    global _ANNOTATION_ENGINE
    if _ANNOTATION_ENGINE is not None:
        _ANNOTATION_ENGINE.quit()
        _ANNOTATION_ENGINE = None


def get_annotation_engine() -> chess.engine.SimpleEngine:
    global _ANNOTATION_ENGINE
    if _ANNOTATION_ENGINE is None:
        if (
            _ANNOTATION_ENGINE_PATH is None
            or _ANNOTATION_DEPTH is None
            or _ANNOTATION_THREADS is None
        ):
            raise RuntimeError("Stockfish worker engine is not initialized.")
        _ANNOTATION_ENGINE = open_stockfish_engine(
            _ANNOTATION_ENGINE_PATH,
            _ANNOTATION_DEPTH,
            _ANNOTATION_THREADS,
        )
    return _ANNOTATION_ENGINE


def best_move_from_info(info: chess.engine.InfoDict) -> Optional[str]:
    pv = info.get("pv")
    if pv is None or not pv:
        return None
    return pv[0].uci()


def analyze_board(board: chess.Board) -> Optional[str]:
    for _ in range(2):
        engine = get_annotation_engine()
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=_ANNOTATION_DEPTH))
            return best_move_from_info(info)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            close_annotation_engine()

    return None


def create_output_bucket_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE rows (
            fen TEXT NOT NULL,
            next_move TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (fen)
        )
        """
    )
    return conn


def annotate_bucket(task: AnnotationBucketTask) -> AnnotationBucketResult:
    input_path = Path(task.input_path)
    output_path = Path(task.output_path)
    temp_output_path = output_path.with_suffix(".tmp.sqlite")

    if temp_output_path.exists():
        temp_output_path.unlink()

    positions_seen = 0
    rows_written = 0
    skipped_positions = 0
    input_conn = None
    output_conn = None

    try:
        input_conn = sqlite3.connect(f"file:{input_path}?mode=ro", uri=True)
        output_conn = create_output_bucket_db(temp_output_path)

        cursor = input_conn.execute("SELECT fen, count FROM counts")
        insert_sql = "INSERT INTO rows (fen, next_move, count) VALUES (?, ?, ?)"
        buffer: List[Tuple[str, str, int]] = []

        for fen, count in cursor:
            positions_seen += 1

            try:
                board = chess.Board(fen)
            except ValueError:
                skipped_positions += 1
                continue

            if board.is_game_over():
                skipped_positions += 1
                continue

            best_move = analyze_board(board)
            if best_move is None:
                skipped_positions += 1
                continue

            buffer.append((fen, best_move, int(count)))

            if len(buffer) >= 1024:
                output_conn.executemany(insert_sql, buffer)
                output_conn.commit()
                rows_written += len(buffer)
                buffer = []

        if buffer:
            output_conn.executemany(insert_sql, buffer)
            output_conn.commit()
            rows_written += len(buffer)

        output_conn.close()
        output_conn = None
        temp_output_path.replace(output_path)

        return AnnotationBucketResult(
            bucket_id=task.bucket_id,
            positions_seen=positions_seen,
            rows_written=rows_written,
            skipped_positions=skipped_positions,
        )
    except Exception as exc:
        return AnnotationBucketResult(
            bucket_id=task.bucket_id,
            positions_seen=positions_seen,
            rows_written=rows_written,
            skipped_positions=skipped_positions,
            error_message=str(exc),
        )
    finally:
        if input_conn is not None:
            input_conn.close()
        if output_conn is not None:
            output_conn.close()
        if temp_output_path.exists():
            temp_output_path.unlink()


def push_dataset(
    dataset: DatasetDict,
    repo_id: str,
    token: Optional[str],
    private: bool,
):
    create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        repo_type="dataset",
        exist_ok=True,
    )
    dataset.push_to_hub(repo_id=repo_id, token=token, private=private)


def load_csv_split(csv_path: Path) -> Dataset:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        has_rows = sum(1 for _ in handle) > 1

    if not has_rows:
        return Dataset.from_dict(
            {
                "fen": [],
                "next_move": [],
                "count": [],
            },
            features=FINAL_FEATURES,
        )

    return load_dataset(
        "csv",
        data_files=str(csv_path),
        features=FINAL_FEATURES,
        split="train",
    )


def assemble_dataset(
    train_csv: Path,
    validation_csv: Path,
    dataset_dir: Path,
) -> DatasetDict:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset = DatasetDict(
        {
            "train": load_csv_split(train_csv),
            "validation": load_csv_split(validation_csv),
        }
    )
    dataset.save_to_disk(str(dataset_dir))
    return dataset


def reduce_bucket_to_splits(
    bucket_path: Path,
    train_writer: csv.DictWriter,
    validation_writer: csv.DictWriter,
    val_ratio: float,
) -> Tuple[int, int]:
    conn = sqlite3.connect(f"file:{bucket_path}?mode=ro", uri=True)

    train_count = 0
    validation_count = 0
    try:
        cursor = conn.execute("SELECT fen, next_move, count FROM rows")

        for fen, next_move, count in cursor:
            row = {
                "fen": fen,
                "next_move": next_move,
                "count": int(count),
            }
            if split_for_fen(fen, val_ratio) == "validation":
                validation_writer.writerow(row)
                validation_count += 1
            else:
                train_writer.writerow(row)
                train_count += 1
    finally:
        conn.close()
    return train_count, validation_count


def total_source_rows(
    source: str,
    requested_splits: Optional[Sequence[str]],
) -> Optional[int]:
    """Return the total number of source rows from metadata only, without iterating.

    Returns None if the count cannot be determined (e.g. network error, unknown
    dataset format), in which case the progress bar will run without a total.
    """
    try:
        source_path = Path(source)
        if source_path.exists():
            dataset_obj = load_from_disk(str(source_path))
            if isinstance(dataset_obj, DatasetDict):
                splits = (
                    list(dataset_obj.keys())
                    if requested_splits is None
                    else list(requested_splits)
                )
                return sum(len(dataset_obj[s]) for s in splits)
            return len(dataset_obj)
        # HuggingFace Hub path: fetch split metadata without downloading data.
        builder = load_dataset_builder(source)
        split_infos = builder.info.splits or {}
        if requested_splits is None:
            return sum(info.num_examples for info in split_infos.values())
        return sum(
            split_infos[s].num_examples for s in requested_splits if s in split_infos
        )
    except Exception:
        return None


def run_collapse(
    source: str,
    source_splits: Optional[Sequence[str]],
    output_dir: Path,
    num_workers: int,
    rows_per_batch: int,
    bucket_count: int,
    skip_source_rows: int,
    max_source_rows: Optional[int],
    manifest_path: Path,
    manifest: Dict,
):
    collapse_state = manifest.setdefault("collapse", {})
    if collapse_state.get("completed"):
        print("Collapse already completed; skipping.")
        return

    row_batches = batched_rows(
        rows=iter_source_rows(source, source_splits),
        rows_per_batch=rows_per_batch,
        skip_rows=skip_source_rows,
        max_rows=max_source_rows,
    )

    bucket_dir = output_dir / "tmp" / "count_buckets"
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    writer = FenCountSqliteManager(bucket_dir)

    total_rows = 0
    total_valid_rows = 0
    total_count = 0
    total_aggregated_rows = 0

    pool = None
    if max_source_rows is not None:
        progress_total: Optional[int] = max_source_rows
    else:
        raw_total = total_source_rows(source, source_splits)
        progress_total = (
            max(0, raw_total - skip_source_rows) if raw_total is not None else None
        )
    pbar = tqdm(total=progress_total, desc="Collapsing source rows", unit="row")

    try:
        if num_workers == 1:
            for batch in row_batches:
                result = collapse_batch(batch, bucket_count)
                total_rows += result.processed_rows
                total_valid_rows += result.valid_rows
                total_count += result.total_count
                total_aggregated_rows += writer.write_rows(result.aggregated_rows)
                result.aggregated_rows = []  # free the large list immediately
                pbar.update(result.processed_rows)
                pbar.set_postfix({"valid": total_valid_rows, "count": total_count})
        else:
            pool = multiprocessing.Pool(processes=num_workers)
            # Feed the pool in bounded chunks so that Pool._handle_tasks does not
            # eagerly drain the entire generator into its unbounded internal task
            # queue, which would materialise every batch in memory at once.
            batch_args = ((batch, bucket_count) for batch in row_batches)
            chunk_size = num_workers * 2
            while True:
                chunk = list(itertools.islice(batch_args, chunk_size))
                if not chunk:
                    break
                for result in pool.imap_unordered(
                    _collapse_batch_star, chunk, chunksize=1
                ):
                    total_rows += result.processed_rows
                    total_valid_rows += result.valid_rows
                    total_count += result.total_count
                    total_aggregated_rows += writer.write_rows(result.aggregated_rows)
                    result.aggregated_rows = []  # free the large list immediately
                    pbar.update(result.processed_rows)
                    pbar.set_postfix({"valid": total_valid_rows, "count": total_count})
    finally:
        pbar.close()
        writer.close()
        if pool is not None:
            pool.close()
            pool.join()

    collapse_state.update(
        {
            "completed": True,
            "source": source,
            "source_splits": list(source_splits)
            if source_splits is not None
            else "all",
            "processed_rows": total_rows,
            "valid_rows": total_valid_rows,
            "total_count": total_count,
            "aggregated_rows_merged": total_aggregated_rows,
            "bucket_count": bucket_count,
        }
    )
    write_manifest(manifest_path, manifest)


def existing_completed_annotation_buckets(annotated_dir: Path) -> List[int]:
    completed: List[int] = []
    for bucket_path in sorted(annotated_dir.glob("bucket-*.sqlite")):
        try:
            bucket_id = int(bucket_path.stem.split("-")[-1])
        except ValueError:
            continue
        completed.append(bucket_id)
    return completed


def run_annotation(
    output_dir: Path,
    stockfish_path: str,
    depth: int,
    num_workers: int,
    engine_threads: int,
    manifest_path: Path,
    manifest: Dict,
):
    annotate_state = manifest.setdefault("annotate", {})
    count_bucket_dir = output_dir / "tmp" / "count_buckets"
    if not count_bucket_dir.exists():
        raise FileNotFoundError(f"Missing count bucket directory: {count_bucket_dir}")

    annotated_dir = output_dir / "tmp" / "annotated_buckets"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    completed_buckets = set(annotate_state.get("completed_bucket_ids", []))
    completed_buckets.update(existing_completed_annotation_buckets(annotated_dir))
    annotate_state["completed_bucket_ids"] = sorted(completed_buckets)
    write_manifest(manifest_path, manifest)

    bucket_paths = sorted(count_bucket_dir.glob("bucket-*.sqlite"))
    pending_tasks: List[AnnotationBucketTask] = []

    for bucket_path in bucket_paths:
        bucket_id = int(bucket_path.stem.split("-")[-1])
        if bucket_id in completed_buckets:
            continue
        output_path = annotated_dir / bucket_path.name
        pending_tasks.append(
            AnnotationBucketTask(
                bucket_id=bucket_id,
                input_path=str(bucket_path),
                output_path=str(output_path),
            )
        )

    if not pending_tasks:
        annotate_state["completed"] = True
        write_manifest(manifest_path, manifest)
        print("Annotation already completed; skipping.")
        return

    total_positions = int(annotate_state.get("positions_seen", 0))
    total_rows_written = int(annotate_state.get("rows_written", 0))
    total_skipped_positions = int(annotate_state.get("skipped_positions", 0))

    pool = None
    pbar = tqdm(total=len(pending_tasks), desc="Annotating buckets", unit="bucket")

    try:
        if num_workers == 1:
            init_annotation_worker(stockfish_path, depth, engine_threads)
            for task in pending_tasks:
                result = annotate_bucket(task)
                if result.error_message:
                    raise RuntimeError(
                        f"Bucket {result.bucket_id:05d} annotation failed: {result.error_message}"
                    )
                total_positions += result.positions_seen
                total_rows_written += result.rows_written
                total_skipped_positions += result.skipped_positions
                completed_buckets.add(result.bucket_id)
                annotate_state.update(
                    {
                        "completed_bucket_ids": sorted(completed_buckets),
                        "positions_seen": total_positions,
                        "rows_written": total_rows_written,
                        "skipped_positions": total_skipped_positions,
                        "stockfish_path": stockfish_path,
                        "depth": depth,
                        "engine_threads": engine_threads,
                    }
                )
                write_manifest(manifest_path, manifest)
                pbar.update(1)
                pbar.set_postfix(
                    {"rows": total_rows_written, "skip": total_skipped_positions}
                )
        else:
            pool = multiprocessing.Pool(
                processes=num_workers,
                initializer=init_annotation_worker,
                initargs=(stockfish_path, depth, engine_threads),
            )
            for result in pool.imap_unordered(
                annotate_bucket, pending_tasks, chunksize=1
            ):
                if result.error_message:
                    raise RuntimeError(
                        f"Bucket {result.bucket_id:05d} annotation failed: {result.error_message}"
                    )
                total_positions += result.positions_seen
                total_rows_written += result.rows_written
                total_skipped_positions += result.skipped_positions
                completed_buckets.add(result.bucket_id)
                annotate_state.update(
                    {
                        "completed_bucket_ids": sorted(completed_buckets),
                        "positions_seen": total_positions,
                        "rows_written": total_rows_written,
                        "skipped_positions": total_skipped_positions,
                        "stockfish_path": stockfish_path,
                        "depth": depth,
                        "engine_threads": engine_threads,
                    }
                )
                write_manifest(manifest_path, manifest)
                pbar.update(1)
                pbar.set_postfix(
                    {"rows": total_rows_written, "skip": total_skipped_positions}
                )
    finally:
        pbar.close()
        if pool is not None:
            pool.close()
            pool.join()
        else:
            close_annotation_engine()

    annotate_state["completed"] = True
    write_manifest(manifest_path, manifest)


def run_assembly(
    output_dir: Path,
    val_ratio: float,
    push_to_hub_flag: bool,
    repo_id: Optional[str],
    token: Optional[str],
    private: bool,
    manifest_path: Path,
    manifest: Dict,
):
    assemble_state = manifest.setdefault("assemble", {})
    dataset_dir = output_dir / "dataset"
    if assemble_state.get("completed") and not push_to_hub_flag:
        print("Dataset assembly already completed; skipping.")
        return

    annotated_dir = output_dir / "tmp" / "annotated_buckets"
    if not annotated_dir.exists():
        raise FileNotFoundError(f"Missing annotated bucket directory: {annotated_dir}")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    train_csv = final_dir / "train.csv"
    validation_csv = final_dir / "validation.csv"

    with (
        train_csv.open("w", encoding="utf-8", newline="") as train_handle,
        validation_csv.open("w", encoding="utf-8", newline="") as validation_handle,
    ):
        train_writer = csv.DictWriter(
            train_handle, fieldnames=list(FINAL_FEATURES.keys())
        )
        validation_writer = csv.DictWriter(
            validation_handle, fieldnames=list(FINAL_FEATURES.keys())
        )
        train_writer.writeheader()
        validation_writer.writeheader()

        total_train_rows = 0
        total_validation_rows = 0
        bucket_paths = sorted(annotated_dir.glob("bucket-*.sqlite"))

        for bucket_path in tqdm(bucket_paths, desc="Assembling splits", unit="bucket"):
            train_rows, validation_rows = reduce_bucket_to_splits(
                bucket_path=bucket_path,
                train_writer=train_writer,
                validation_writer=validation_writer,
                val_ratio=val_ratio,
            )
            total_train_rows += train_rows
            total_validation_rows += validation_rows

    dataset = assemble_dataset(train_csv, validation_csv, dataset_dir)
    assemble_state.update(
        {
            "completed": True,
            "dataset_dir": str(dataset_dir),
            "val_ratio": val_ratio,
            "splits": {
                split_name: len(split_ds) for split_name, split_ds in dataset.items()
            },
            "train_rows": total_train_rows,
            "validation_rows": total_validation_rows,
        }
    )
    write_manifest(manifest_path, manifest)

    if push_to_hub_flag:
        if not repo_id:
            raise ValueError("--repo-id is required when --push-to-hub is set.")
        push_dataset(dataset=dataset, repo_id=repo_id, token=token, private=private)
        manifest["push"] = {
            "completed": True,
            "repo_id": repo_id,
            "private": private,
        }
        write_manifest(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Stockfish-labeled chess dataset from a deduplicated human-move source dataset."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-splits", default="all")
    parser.add_argument("--output-dir", default="./data/processed_lichess_stockfish")
    parser.add_argument("--stockfish-path", default="/usr/bin/stockfish")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--rows-per-batch", type=int, default=8192)
    parser.add_argument("--bucket-count", type=int, default=2048)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--skip-source-rows", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.engine_threads < 1:
        raise ValueError("--engine-threads must be >= 1")
    if args.rows_per_batch < 1:
        raise ValueError("--rows-per-batch must be >= 1")
    if args.bucket_count < 1:
        raise ValueError("--bucket-count must be >= 1")
    if args.skip_source_rows < 0:
        raise ValueError("--skip-source-rows must be >= 0")
    if args.max_source_rows is not None and args.max_source_rows < 1:
        raise ValueError("--max-source-rows must be >= 1 when provided")
    if args.depth < 1:
        raise ValueError("--depth must be >= 1")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0.0, 1.0)")
    if args.push_to_hub and not args.repo_id:
        raise ValueError("--repo-id is required when --push-to-hub is set")


def main():
    args = parse_args()
    validate_args(args)

    requested_splits = parse_source_splits(args.source_splits)
    output_dir = Path(args.output_dir)
    ensure_clean_output_dir(output_dir, overwrite=args.overwrite)

    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest["config"] = {
        "source": args.source,
        "source_splits": list(requested_splits)
        if requested_splits is not None
        else "all",
        "output_dir": str(output_dir),
        "stockfish_path": args.stockfish_path,
        "depth": args.depth,
        "num_workers": args.num_workers,
        "engine_threads": args.engine_threads,
        "rows_per_batch": args.rows_per_batch,
        "bucket_count": args.bucket_count,
        "max_source_rows": args.max_source_rows,
        "skip_source_rows": args.skip_source_rows,
        "val_ratio": args.val_ratio,
    }
    write_manifest(manifest_path, manifest)

    run_collapse(
        source=args.source,
        source_splits=requested_splits,
        output_dir=output_dir,
        num_workers=args.num_workers,
        rows_per_batch=args.rows_per_batch,
        bucket_count=args.bucket_count,
        skip_source_rows=args.skip_source_rows,
        max_source_rows=args.max_source_rows,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    run_annotation(
        output_dir=output_dir,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        num_workers=args.num_workers,
        engine_threads=args.engine_threads,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    run_assembly(
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        push_to_hub_flag=args.push_to_hub,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        manifest_path=manifest_path,
        manifest=manifest,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
