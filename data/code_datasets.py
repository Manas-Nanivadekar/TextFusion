import json
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class CodeDataset(Dataset):
    """
    Dataset for code generation with D3PM.

    Loads from:
    - Downloaded The Stack subset (JSONL files with metadata.json)
    - Local directory of .py files (fallback)
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_seq_len: int = 256,
        max_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.pad_id = tokenizer.vocab.get("<PAD>", 0)
        self.mask_id = tokenizer.vocab.get("<MASK>", 1)
        self.bos_id = tokenizer.vocab.get("<BOS>", 3)
        self.eos_id = tokenizer.vocab.get("<EOS>", 4)

        data_dir = Path(data_dir)

        if (data_dir / "metadata.json").exists():
            self.examples = self._load_jsonl(data_dir, max_examples)
        else:
            self.examples = self._load_py_files(data_dir, max_examples)

        print(f"  Loaded {len(self.examples):,} code examples")

    def _load_jsonl(self, data_dir: Path, max_examples: Optional[int]) -> list[str]:
        """Load from JSONL files (The Stack format)."""
        examples = []

        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    if max_examples and len(examples) >= max_examples:
                        return examples
                    item = json.loads(line)
                    examples.append(item["code"])

        return examples

    def _load_py_files(self, data_dir: Path, max_examples: Optional[int]) -> list[str]:
        """Load from local .py files."""
        examples = []
        skip = {".venv", "__pycache__", "site-packages", ".git", "node_modules"}

        for py_file in data_dir.rglob("*.py"):
            if any(d in py_file.parts for d in skip):
                continue
            if max_examples and len(examples) >= max_examples:
                break
            try:
                code = py_file.read_text(encoding="utf-8")
                lines = len(code.split("\n"))
                if 10 <= lines <= 1000:
                    examples.append(code)
            except:
                continue

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        code = self.examples[idx]

        tokens = self.tokenizer.encode(code)

        max_content = self.max_seq_len - 2
        if len(tokens) > max_content:
            start = random.randint(0, len(tokens) - max_content)
            tokens = tokens[start : start + max_content]

        tokens = [self.bos_id] + tokens + [self.eos_id]

        pad_len = self.max_seq_len - len(tokens)
        if pad_len > 0:
            tokens = tokens + [self.pad_id] * pad_len

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        tokens = [t for t in tokens if t >= 5]
        return self.tokenizer.decode(tokens)


PythonCodeDataset = CodeDataset


if __name__ == "__main__":
    print("Testing CodeDataset...")

    from data.code_tokenizer import CodeBPETokenizer

    tokenizer = CodeBPETokenizer()
    if Path("data/code_tokenizer.json").exists():
        tokenizer.load("data/code_tokenizer.json")
    else:
        print("Building tokenizer on The Stack...")
        tokenizer.build_vocab("data/the-stack", vocab_size=8000)
        tokenizer.save("data/code_tokenizer.json")

    print(f"Vocab: {tokenizer.vocab_size}")

    ds = CodeDataset("data/the-stack", tokenizer, max_seq_len=256, max_examples=1000)

    sample = ds[0]
    print(f"Shape: {sample.shape}")
    print(f"First 20: {sample[:20].tolist()}")

    decoded = tokenizer.decode([t for t in sample.tolist() if t >= 5])
    print(f"Decoded:\n{decoded[:200]}...")

    print("\nWorks!")
