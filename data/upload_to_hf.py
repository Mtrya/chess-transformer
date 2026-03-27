import argparse
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


def load_local_dataset(dataset_path: Path) -> Union[Dataset, DatasetDict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {dataset_path}")
    return load_from_disk(str(dataset_path))


def dataset_split_sizes(dataset: Union[Dataset, DatasetDict]) -> dict:
    if isinstance(dataset, DatasetDict):
        return {
            split_name: len(split_dataset)
            for split_name, split_dataset in dataset.items()
        }
    return {"train": len(dataset)}


def repo_exists(api: HfApi, repo_id: str, token: Optional[str]) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        return True
    except RepositoryNotFoundError:
        return False
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return False
        raise


def push_dataset(
    dataset: Union[Dataset, DatasetDict],
    repo_id: str,
    token: Optional[str],
    private: bool,
    max_shard_size: str,
):
    create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        repo_type="dataset",
        exist_ok=True,
    )
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        max_shard_size=max_shard_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local Hugging Face dataset saved with save_to_disk() to the Hub."
    )
    parser.add_argument(
        "--dataset-path", required=True, help="Local path produced by save_to_disk()."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target dataset repo, for example username/dataset-name.",
    )
    parser.add_argument(
        "--token", default=None, help="Optional Hugging Face token override."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Maximum shard size passed to push_to_hub().",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Push even if the target repo already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    dataset = load_local_dataset(dataset_path)
    split_sizes = dataset_split_sizes(dataset)

    print(f"Loaded local dataset from {dataset_path}")
    print(f"Split sizes: {split_sizes}")

    api = HfApi(token=args.token)
    if repo_exists(api=api, repo_id=args.repo_id, token=args.token) and not args.force:
        print(f"Dataset repo already exists; skipping upload: {args.repo_id}")
        return

    push_dataset(
        dataset=dataset,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        max_shard_size=args.max_shard_size,
    )
    print(f"Dataset uploaded to {args.repo_id}")


if __name__ == "__main__":
    main()
