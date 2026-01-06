"""
Download The Stack (Python subset) for D3PM training.
Usage:
    python scripts/download_stack.py --n_examples 100000 --token hf_xxxxx
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm


def download_stack(
    n_examples: int,
    output_dir: str,
    token: str,
    min_lines: int = 10,
    max_lines: int = 500,
):
    """Download Python examples from The Stack."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets huggingface_hub")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {n_examples:,} Python examples from The Stack...")
    print(f"Output: {output_dir}")
    print(f"Filter: {min_lines}-{max_lines} lines\n")

    # Stream to avoid downloading entire dataset
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train",
        streaming=True,
        trust_remote_code=True,
        token=token,
    )

    examples = []
    skipped = {"too_short": 0, "too_long": 0, "error": 0}

    pbar = tqdm(total=n_examples, desc="Downloading")

    for item in dataset:
        if len(examples) >= n_examples:
            break

        try:
            code = item["content"]
            lines = code.split("\n")
            n_lines = len(lines)

            if n_lines < min_lines:
                skipped["too_short"] += 1
                continue
            if n_lines > max_lines:
                skipped["too_long"] += 1
                continue

            examples.append(
                {
                    "code": code,
                    "size": item.get("size", len(code)),
                    "path": item.get("path", ""),
                    "n_lines": n_lines,
                }
            )
            pbar.update(1)

        except Exception as e:
            skipped["error"] += 1
            continue

    pbar.close()

    print(f"\nCollected: {len(examples):,} examples")
    print(f"Skipped: {skipped}")

    # Save in chunks for memory efficiency
    chunk_size = 10000
    for i in range(0, len(examples), chunk_size):
        chunk = examples[i : i + chunk_size]
        chunk_file = output_dir / f"python_{i:06d}.jsonl"

        with open(chunk_file, "w") as f:
            for ex in chunk:
                f.write(json.dumps(ex) + "\n")

        print(f"Saved {chunk_file.name} ({len(chunk):,} examples)")

    # Save metadata
    meta = {
        "total_examples": len(examples),
        "min_lines": min_lines,
        "max_lines": max_lines,
        "source": "bigcode/the-stack-dedup",
        "language": "python",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Downloaded {len(examples):,} examples to {output_dir}")

    # Show storage usage
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.jsonl"))
    print(f"  Total size: {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download The Stack Python subset")
    parser.add_argument(
        "--n_examples", type=int, default=100000, help="Number of examples"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/the-stack", help="Output directory"
    )
    parser.add_argument("--min_lines", type=int, default=10, help="Min lines per file")
    parser.add_argument("--max_lines", type=int, default=500, help="Max lines per file")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    # Token from arg or environment
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Provide --token or set HF_TOKEN environment variable")

    download_stack(
        args.n_examples, args.output_dir, token, args.min_lines, args.max_lines
    )


if __name__ == "__main__":
    main()
