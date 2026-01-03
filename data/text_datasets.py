import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import urllib.request


class ShakespeareDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/shakespeare",
        max_seq_len: int = 64,
        vocab_size: int = 5000,
    ):
        self.data_dir = Path(data_dir)
        self.data_file = self.data_dir / "train.txt"
        self.max_seq_len = max_seq_len

        if not self.data_file.exists():
            self._download()

        print(f"Loading Shakespeare from {self.data_file}...")

        self.text = self._load_text()
        print(f"  Characters: {len(self.text):,}")

        self.vocab = self._build_vocab(vocab_size)
        print(f"  Vocab size: {self.vocab_size}")

        self.tokens = self._tokenize(self.text)
        print(f"  Tokens: {len(self.tokens):,}")

        self.sequences = self._create_sequences()
        print(f"  Sequences: {len(self.sequences)}")

    def _download(self):
        """Download Shakespeare if not exists"""
        print("Downloading Tiny Shakespeare...")

        self.data_dir.mkdir(parents=True, exist_ok=True)

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        try:
            urllib.request.urlretrieve(url, self.data_file)
            print(f"✓ Downloaded to {self.data_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download: {e}")

    def _load_text(self) -> str:
        with open(self.data_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _build_vocab(self, max_vocab_size: int) -> dict:
        """Word-level vocabulary"""
        words = self.text.lower().split()

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        vocab = {
            "<PAD>": 0,
            "<MASK>": 1,
            "<UNK>": 2,
            "<BOS>": 3,
            "<EOS>": 4,
        }

        for word, _ in sorted_words[: max_vocab_size - 5]:
            vocab[word] = len(vocab)

        self.vocab_size = len(vocab)
        self.idx_to_word = {v: k for k, v in vocab.items()}

        return vocab

    def _tokenize(self, text: str) -> List[int]:
        words = text.lower().split()
        return [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]

    def _create_sequences(self) -> List[List[int]]:
        sequences = []
        step_size = self.max_seq_len // 2

        for i in range(0, len(self.tokens) - self.max_seq_len, step_size):
            seq = self.tokens[i : i + self.max_seq_len]
            seq = [self.vocab["<BOS>"]] + seq + [self.vocab["<EOS>"]]

            target_len = self.max_seq_len + 2
            if len(seq) < target_len:
                seq += [self.vocab["<PAD>"]] * (target_len - len(seq))

            sequences.append(seq[:target_len])

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.sequences[idx], dtype=torch.long)

    def decode(self, tokens: List[int]) -> str:
        words = [self.idx_to_word.get(tok, "<UNK>") for tok in tokens]
        words = [w for w in words if w not in ["<PAD>", "<BOS>", "<EOS>", "<MASK>"]]
        return " ".join(words)

    @property
    def mask_token_id(self) -> int:
        return self.vocab["<MASK>"]

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<PAD>"]


if __name__ == "__main__":
    print("Testing Shakespeare Dataset...\n")

    dataset = ShakespeareDataset(
        data_dir="data/shakespeare", max_seq_len=32, vocab_size=3000
    )

    print("\n" + "=" * 60)

    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Shape: {sample.shape}")
    print(f"  Decoded: {dataset.decode(sample.tolist())}")

    print(f"\n Shakespeare dataset works!")
