"""Publish the prepared ActiveQA Telugu sample to the signed-in HF account."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/indic_rag_te")
    parser.add_argument("--repo-name", default="activeqa-telugu-rl-data")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder)
    required = ("README.md", "train.jsonl", "dev.jsonl", "reformulations_review.jsonl", "metadata.json")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset folder is incomplete: {', '.join(missing)}")

    api = HfApi()
    identity = api.whoami()
    owner = identity.get("name") or identity.get("fullname")
    if not owner:
        raise RuntimeError("Could not determine the signed-in Hugging Face username")
    repo_id = f"{owner}/{args.repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="dataset",
                      commit_message="Add leakage-aware Telugu QA and review paraphrase sample")
    print(f"Published dataset: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
