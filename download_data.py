"""
download_data.py — fetch EntroPE's datasets and/or forecasting checkpoints from
the HuggingFace Hub into this repo, preserving layout.

The benchmark CSVs (`dataset/`) and the trained forecasting checkpoints
(`checkpoints/`) are too large for GitHub, so they live on the Hub:
    https://huggingface.co/datasets/sachithabey/EntroPE

The frozen entropy models (`entropy_model_checkpoints/dm16/`) are small and ship
in the GitHub repo already — you do not need to download them.

Usage:
    pip install huggingface_hub
    python download_data.py                      # datasets + checkpoints
    python download_data.py --what datasets      # just the CSVs (to train from scratch)
    python download_data.py --what checkpoints   # just the checkpoints (to eval only)
"""

import argparse

HF_REPO = "sachithabey/EntroPE"
HF_REPO_TYPE = "dataset"

PATTERNS = {
    "datasets":    ["dataset/*"],
    "checkpoints": ["checkpoints/*/*"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=HF_REPO, help="HuggingFace dataset repo id")
    ap.add_argument("--what", choices=["datasets", "checkpoints", "all"], default="all",
                    help="which assets to fetch (default: all)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Please `pip install huggingface_hub` first.")

    which = ["datasets", "checkpoints"] if args.what == "all" else [args.what]
    allow = [p for w in which for p in PATTERNS[w]]

    print(f"Downloading {which} from {args.repo} …")
    snapshot_download(repo_id=args.repo, repo_type=HF_REPO_TYPE,
                      local_dir=".", allow_patterns=allow)
    print("Done. Files written under dataset/ and/or checkpoints/.")


if __name__ == "__main__":
    main()
