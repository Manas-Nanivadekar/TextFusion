import re
from typing import List, Dict
from collections import Counter
from pathlib import Path
import json


class CodeBPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<PAD>": 0,
            "<MASK>": 1,
            "<UNK>": 2,
            "<BOS>": 3,
            "<EOS>": 4,
        }

        self.vocab = self.special_tokens.copy()
        self.merges = {}
        self.idx_to_token = {v: k for k, v in self.vocab.items()}

    def pre_tokenize(self, code: str) -> List[str]:
        pattern = r"(\s+|[+\-*/%=<>!&|^~]|[(){}\[\],.:;])"

        tokens = re.split(pattern, code)

        tokens = [t for t in tokens if t and not t.isspace()]

        return tokens

    def build_vocab_from_corpus(self, code_files: List[str], max_files: int = 100):
        print(f"Building BPE vocab from up to {max_files} files...")

        all_tokens = []
        for file_path in code_files[:max_files]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                tokens = self.pre_tokenize(code)
                all_tokens.extend(tokens)
            except Exception as e:
                continue

        print(f"  Total pre-tokens: {len(all_tokens):,}")

        word_freqs = Counter(all_tokens)
        print(f"  Unique words: {len(word_freqs):,}")

        base_vocab_size = min(1000, len(word_freqs))
        most_common_words = [
            word for word, _ in word_freqs.most_common(base_vocab_size)
        ]

        for word in most_common_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        print(f"  Base vocab (whole words): {len(self.vocab)}")

        remaining_words = {w: f for w, f in word_freqs.items() if w not in self.vocab}

        splits = {word: list(word) for word in remaining_words.keys()}

        num_merges = self.vocab_size - len(self.vocab)
        num_merges = max(num_merges, 0)

        print(f"  Learning {num_merges} BPE merges...")

        for i in range(num_merges):

            pair_freqs = self._count_pairs(splits, remaining_words)

            if not pair_freqs:
                print(f"  No more pairs to merge (stopped at {i} merges)")
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)

            splits = self._merge_pair(best_pair, splits)

            self.merges[best_pair] = len(self.merges)

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

            if (i + 1) % 500 == 0:
                print(
                    f"    Merge {i+1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' = '{merged_token}'"
                )

        for word, split in splits.items():
            for token in split:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.idx_to_token = {v: k for k, v in self.vocab.items()}

        print(f"  Final vocab size: {len(self.vocab)}")
        print(f"  Total merges learned: {len(self.merges)}")

    def _count_pairs(self, splits, word_freqs):
        pair_freqs = Counter()

        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) < 2:
                continue

            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _merge_pair(self, pair, splits):
        new_splits = {}

        for word, split in splits.items():
            if len(split) < 2:
                new_splits[word] = split
                continue

            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1

            new_splits[word] = new_split

        return new_splits

    def encode(self, code: str) -> List[int]:
        tokens = self.pre_tokenize(code)

        encoded = []
        for token in tokens:
            chars = list(token)

            while len(chars) > 1:
                pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]

                applicable_merges = [
                    (pair, self.merges[pair]) for pair in pairs if pair in self.merges
                ]

                if not applicable_merges:
                    break

                pair_to_merge = min(applicable_merges, key=lambda x: x[1])[0]

                new_chars = []

                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == pair_to_merge:
                        new_chars.append(chars[i] + chars[i + 1])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars

            for subtoken in chars:
                encoded.append(self.vocab.get(subtoken, self.vocab["<UNK>"]))

        return encoded

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.idx_to_token.get(idx, "<UNK>") for idx in token_ids]

        tokens = [t for t in tokens if t not in self.special_tokens]

        # Join tokens (simple: space between)
        # TODO: Smarter joining (no space before :,.)
        return " ".join(tokens)

    def save(self, path: str):
        data = {
            "vocab": self.vocab,
            "merges": {str(k): v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.vocab_size = data["vocab_size"]
        self.idx_to_token = {int(v): k for k, v in self.vocab.items()}

        self.merges = {eval(k): v for k, v in data["merges"].items()}


if __name__ == "__main__":
    print("Testing CodeBPETokenizer...\n")

    tokenizer = CodeBPETokenizer(vocab_size=3000)

    code = "def add(a, b):\n    return a + b"
    pre_tokens = tokenizer.pre_tokenize(code)
    print(f"Pre-tokenization:")
    print(f"  Input: {repr(code)}")
    print(f"  Output: {pre_tokens}\n")

    print("Building vocab from TextFusion code...")

    all_files = []
    for pattern in ["**/*.py"]:
        files = Path(".").glob(pattern)
        all_files.extend(
            [
                f
                for f in files
                if ".venv" not in str(f)
                and "__pycache__" not in str(f)
                and "site-packages" not in str(f)
            ]
        )

    print(f"Found {len(all_files)} Python files (excluding venv/cache)")

    tokenizer.build_vocab_from_corpus([str(f) for f in all_files], max_files=50)

    print(f"\nTesting encoding:")
    print(f"  Input: {repr(code)}")
    encoded = tokenizer.encode(code)
    print(f"  Encoded IDs: {encoded}")
    print(f"  Encoded tokens: {[tokenizer.idx_to_token[i] for i in encoded]}")

    decoded = tokenizer.decode(encoded)
    print(f"  Decoded: {decoded}")

    complex_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item
    return total
""".strip()

    print(f"\nTesting on complex code:")
    print(f"  Input: {repr(complex_code)}")
    encoded2 = tokenizer.encode(complex_code)
    print(f"  Num tokens: {len(encoded2)}")
    decoded2 = tokenizer.decode(encoded2)
    print(f"  Decoded: {decoded2[:100]}...")

    # Save
    Path("data").mkdir(exist_ok=True)
    tokenizer.save("data/code_tokenizer.json")
    print(f"\nTokenizer saved to data/code_tokenizer.json")

    print(f"\nVocab samples:")
    sample_tokens = list(tokenizer.vocab.keys())[5:25]
    print(f"  {sample_tokens}")
