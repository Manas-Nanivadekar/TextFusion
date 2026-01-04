import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional
import json

from .code_tokenizer import CodeBPETokenizer
from .code_infilling import create_infilling_example


class PythonCodeDataset(Dataset):
    def __init__(
        self,
        data_source: str,
        tokenizer: CodeBPETokenizer,
        max_seq_len: int = 256,
        mask_ratio: tuple = (0.2, 0.5),
        cache_dir: str = "data/code_cache",
        use_cache: bool = True,
        max_files: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = self.cache_dir / f"{Path(data_source).name}_cache.json"

        if use_cache and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            self.examples = self._load_cache(cache_file)
        else:
            print(f"Creating dataset from: {data_source}")

            if data_source == "the-stack":
                self.examples = self._load_from_stack(max_files)
            else:
                self.examples = self._load_from_directory(data_source, max_files)

            if use_cache:
                self._save_cache(cache_file)

        print(f"Dataset size: {len(self.examples)} examples")

    def _chunk_code(self, code: str, target_lines: int = 80) -> List[str]:
        lines = code.split("\n")

        if len(lines) <= target_lines:
            return [code]

        chunks = []
        current_chunk = []
        indent_stack = []

        for i, line in enumerate(lines):
            current_chunk.append(line)

            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                indent_stack.append(len(line) - len(stripped))

            if len(current_chunk) >= target_lines:
                if (
                    stripped.startswith("def ")
                    or stripped.startswith("class ")
                    or (not stripped and i < len(lines) - 1)
                ):

                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    indent_stack = []

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _load_from_directory(
        self, directory: str, max_files: Optional[int]
    ) -> List[Dict]:
        directory = Path(directory)

        py_files = list(directory.rglob("*.py"))

        py_files = [
            f
            for f in py_files
            if not any(x in str(f) for x in [".venv", "__pycache__", "site-packages"])
        ]

        if max_files:
            py_files = py_files[:max_files]

        print(f"Found {len(py_files)} Python files")

        examples = []
        total_chunks = 0

        for file_path in py_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                lines = code.split("\n")
                if len(lines) < 10:
                    continue

                if len(lines) > 120:
                    code_chunks = self._chunk_code(code, target_lines=80)
                    print(
                        f"  {file_path.name}: {len(lines)} lines → {len(code_chunks)} chunks"
                    )
                else:
                    code_chunks = [code]

                total_chunks += len(code_chunks)

                for chunk_idx, code_chunk in enumerate(code_chunks):
                    infill_ex = create_infilling_example(code_chunk, self.mask_ratio)

                    before_tokens = self.tokenizer.encode(infill_ex["before"])
                    masked_tokens = self.tokenizer.encode(infill_ex["masked"])
                    after_tokens = self.tokenizer.encode(infill_ex["after"])

                    total_len = (
                        len(before_tokens) + len(masked_tokens) + len(after_tokens)
                    )

                    if total_len <= self.max_seq_len:
                        examples.append(
                            {
                                "before": before_tokens,
                                "masked": masked_tokens,
                                "after": after_tokens,
                                "file": f"{file_path.name}_chunk{chunk_idx}",
                            }
                        )
                    else:
                        pass

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        print(f"Total chunks created: {total_chunks}")
        return examples

    def _load_from_stack(self, max_samples: Optional[int]) -> List[Dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        print("Loading The Stack (Python)...")

        dataset = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
        )

        examples = []
        count = 0

        for item in dataset:
            if max_samples and count >= max_samples:
                break

            code = item["content"]

            lines = code.split("\n")
            if len(lines) < 10 or len(lines) > 2000:
                continue

            infill_ex = create_infilling_example(code, self.mask_ratio)

            before_tokens = self.tokenizer.encode(infill_ex["before"])
            masked_tokens = self.tokenizer.encode(infill_ex["masked"])
            after_tokens = self.tokenizer.encode(infill_ex["after"])

            total_len = len(before_tokens) + len(masked_tokens) + len(after_tokens)

            if total_len <= self.max_seq_len:
                examples.append(
                    {
                        "before": before_tokens,
                        "masked": masked_tokens,
                        "after": after_tokens,
                        "file": f"stack_{count}",
                    }
                )
                count += 1

                if count % 1000 == 0:
                    print(f"  Processed {count} examples...")

        return examples

    def _save_cache(self, cache_file: Path):
        print(f"Saving cache to {cache_file}...")
        with open(cache_file, "w") as f:
            json.dump(self.examples, f)

    def _load_cache(self, cache_file: Path) -> List[Dict]:
        with open(cache_file, "r") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        full_sequence = example["before"] + example["masked"] + example["after"]

        if len(full_sequence) < self.max_seq_len:
            full_sequence += [self.tokenizer.vocab["<PAD>"]] * (
                self.max_seq_len - len(full_sequence)
            )
        else:
            full_sequence = full_sequence[: self.max_seq_len]

        return {
            "tokens": torch.tensor(full_sequence, dtype=torch.long),
            "before_len": len(example["before"]),
            "masked_len": len(example["masked"]),
            "after_len": len(example["after"]),
        }


if __name__ == "__main__":
    print("Testing PythonCodeDataset...")

    from .code_tokenizer import CodeBPETokenizer

    tokenizer = CodeBPETokenizer()
    tokenizer.load("data/code_tokenizer.json")

    print(f"\nTokenizer loaded: {tokenizer.vocab_size} tokens")

    print("\n" + "=" * 60)
    print("Testing on local TextFusion code...")
    print("=" * 60)

    dataset = PythonCodeDataset(
        data_source=".",
        tokenizer=tokenizer,
        max_seq_len=1024,
        max_files=10,
        use_cache=False,
    )

    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Tokens shape: {sample['tokens'].shape}")
    print(f"  Before length: {sample['before_len']}")
    print(f"  Masked length: {sample['masked_len']}")
    print(f"  After length: {sample['after_len']}")

    tokens = sample["tokens"].tolist()

    tokens = [t for t in tokens if t != tokenizer.vocab["<PAD>"]]
    decoded = tokenizer.decode(tokens)
    print(f"\nDecoded (first 200 chars):")
    print(f"  {decoded[:200]}...")

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    print(f"\nBatch:")
    print(f"  Tokens: {batch['tokens'].shape}")
    print(f"  Before lengths: {batch['before_len']}")

    print("\n PythonCodeDataset works!")
