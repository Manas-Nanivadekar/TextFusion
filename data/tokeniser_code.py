"""Fast BPE Tokenizer using HuggingFace tokenizers library."""

import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


class FastCodeTokenizer:
    """Wrapper around HuggingFace tokenizers for code."""

    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.special_tokens = ["<PAD>", "<MASK>", "<UNK>", "<BOS>", "<EOS>"]
        self.tokenizer = None
        self._vocab = None

    def build_vocab(self, data_dir: str, max_examples: int = 50000):
        """Build BPE vocab from The Stack JSONL or .py files."""
        data_dir = Path(data_dir)

        # Collect code
        print(f"Loading code from {data_dir}...")
        texts = []

        if (data_dir / "metadata.json").exists():
            for jsonl_file in sorted(data_dir.glob("*.jsonl")):
                with open(jsonl_file) as f:
                    for line in f:
                        if len(texts) >= max_examples:
                            break
                        texts.append(json.loads(line)["code"])
                if len(texts) >= max_examples:
                    break
        else:
            for py_file in data_dir.rglob("*.py"):
                if len(texts) >= max_examples:
                    break
                try:
                    texts.append(py_file.read_text())
                except:
                    pass

        print(f"  Loaded {len(texts):,} examples")

        # Create BPE tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Train
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
        )

        print(f"Training BPE (vocab_size={self.vocab_size})...")
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        # Add ByteLevel decoder to convert Ġ back to spaces
        from tokenizers import decoders

        self.tokenizer.decoder = decoders.ByteLevel()

        self._build_vocab_dict()
        print(f"  Final vocab: {len(self.vocab)}")

    def _build_vocab_dict(self):
        """Build vocab dict from tokenizer."""
        self._vocab = self.tokenizer.get_vocab()

    @property
    def vocab(self) -> dict:
        if self._vocab is None:
            self._build_vocab_dict()
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, val):
        self._vocab_size = val

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        # Filter special tokens (0-4)
        filtered = [i for i in ids if i >= 5]
        # Use the tokenizer's built-in decode which handles ByteLevel
        return self.tokenizer.decode(filtered, skip_special_tokens=True)

    def save(self, path: str):
        """Save tokenizer."""
        self.tokenizer.save(path)
        print(f"Saved to {path}")

    def load(self, path: str):
        """Load tokenizer."""
        self.tokenizer = Tokenizer.from_file(path)
        # Ensure ByteLevel decoder is set
        from tokenizers import decoders

        self.tokenizer.decoder = decoders.ByteLevel()
        self._build_vocab_dict()
        print(f"Loaded tokenizer: {len(self.vocab)} tokens")


# Backwards compatible alias
CodeBPETokenizer = FastCodeTokenizer


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/the-stack"

    t = FastCodeTokenizer(vocab_size=16000)
    t.build_vocab(data_dir, max_examples=50000)

    # Test
    test = """def calculate_total(items):
    total = 0
    for item in items:
        total += item
    return total

if __name__ == "__main__":
    print(calculate_total([1, 2, 3]))
"""

    print("\n" + "=" * 50)
    enc = t.encode(test)
    print(f"Encoded: {len(enc)} tokens")
    print(f"First 20 IDs: {enc[:20]}")
    print(f"Decoded:\n{t.decode(enc)}")

    # Check keywords
    print("\nKeywords:")
    for kw in ["import", "def", "class", "return", "__name__"]:
        e = t.encode(kw)
        print(f"  {kw}: {len(e)} tokens -> {e}")

    t.save("data/code_tokenizer_fast.json")
