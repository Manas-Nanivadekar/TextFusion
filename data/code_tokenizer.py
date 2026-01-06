import re
import json
from pathlib import Path
from collections import Counter
from typing import Optional


class CodeBPETokenizer:
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.special_tokens = ["<PAD>", "<MASK>", "<UNK>", "<BOS>", "<EOS>"]
        self.vocab = {t: i for i, t in enumerate(self.special_tokens)}
        self.merges = {}
        self.idx_to_token = {i: t for t, i in self.vocab.items()}

    def pre_tokenize(self, code: str) -> list[str]:
        """Split code into words, keeping operators and punctuation separate."""
        # Split on whitespace and operators, keep delimiters
        pattern = (
            r'(\s+|[+\-*/%=<>!&|^~]+|[(){}\[\],.:;@#]|"""|\'\'\' |"[^"]*"|\'[^\']*\')'
        )
        parts = re.split(pattern, code)
        # Filter empty and pure whitespace
        return [p for p in parts if p and not p.isspace()]

    def build_vocab(
        self, data_dir: str, vocab_size: Optional[int] = None, max_examples: int = 50000
    ):
        """Build BPE vocab from The Stack JSONL files or .py files."""
        if vocab_size:
            self.vocab_size = vocab_size

        data_dir = Path(data_dir)

        # Collect code from source
        print(f"Building BPE vocab (target size: {self.vocab_size})...")

        if (data_dir / "metadata.json").exists():
            corpus = self._load_jsonl_corpus(data_dir, max_examples)
        else:
            corpus = self._load_py_corpus(data_dir, max_examples)

        # Pre-tokenize all code
        print("Pre-tokenizing...")
        word_freqs = Counter()
        for code in corpus:
            tokens = self.pre_tokenize(code)
            word_freqs.update(tokens)

        print(f"  Unique words: {len(word_freqs):,}")
        print(f"  Total tokens: {sum(word_freqs.values()):,}")

        # Step 1: Add most common whole words to vocab
        base_size = min(2000, len(word_freqs))
        for word, _ in word_freqs.most_common(base_size):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        print(f"  Base vocab (common words): {len(self.vocab)}")

        # Step 2: Build character vocab from remaining words
        remaining = {w: f for w, f in word_freqs.items() if w not in self.vocab}

        # Add all individual characters first
        all_chars = set()
        for word in remaining:
            all_chars.update(word)
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        print(f"  After adding chars: {len(self.vocab)}")

        # Step 3: BPE merges
        splits = {word: list(word) for word in remaining}
        num_merges = self.vocab_size - len(self.vocab)

        print(f"  Learning {num_merges} BPE merges...")

        for i in range(num_merges):
            # Count pairs
            pair_freqs = Counter()
            for word, freq in remaining.items():
                split = splits[word]
                for j in range(len(split) - 1):
                    pair_freqs[(split[j], split[j + 1])] += freq

            if not pair_freqs:
                print(f"  No more pairs at merge {i}")
                break

            # Merge best pair
            best = max(pair_freqs, key=pair_freqs.get)
            merged = best[0] + best[1]

            # Update splits
            for word in splits:
                split = splits[word]
                new_split = []
                j = 0
                while j < len(split):
                    if (
                        j < len(split) - 1
                        and split[j] == best[0]
                        and split[j + 1] == best[1]
                    ):
                        new_split.append(merged)
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                splits[word] = new_split

            self.merges[best] = len(self.merges)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

            if (i + 1) % 1000 == 0:
                print(
                    f"    Merge {i+1}: '{best[0]}' + '{best[1]}' -> '{merged}' (freq: {pair_freqs[best]})"
                )

        self.idx_to_token = {i: t for t, i in self.vocab.items()}

        print(f"  Final vocab: {len(self.vocab)}")
        print(f"  Total merges: {len(self.merges)}")

        # Show some learned tokens
        print(f"\n  Sample merged tokens:")
        merged_tokens = [
            t for t in self.vocab if len(t) > 3 and t not in self.special_tokens
        ][:20]
        print(f"    {merged_tokens}")

    def _load_jsonl_corpus(self, data_dir: Path, max_examples: int) -> list[str]:
        """Load code from The Stack JSONL format."""
        corpus = []
        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    if len(corpus) >= max_examples:
                        print(f"  Loaded {len(corpus):,} examples from JSONL")
                        return corpus
                    item = json.loads(line)
                    corpus.append(item["code"])
        print(f"  Loaded {len(corpus):,} examples from JSONL")
        return corpus

    def _load_py_corpus(self, data_dir: Path, max_examples: int) -> list[str]:
        """Load code from .py files."""
        corpus = []
        skip = {".venv", "__pycache__", "site-packages", ".git"}
        for py_file in data_dir.rglob("*.py"):
            if any(s in py_file.parts for s in skip):
                continue
            if len(corpus) >= max_examples:
                break
            try:
                corpus.append(py_file.read_text(encoding="utf-8"))
            except:
                continue
        print(f"  Loaded {len(corpus):,} .py files")
        return corpus

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        words = self.pre_tokenize(text)

        ids = []
        for word in words:
            # Check if whole word in vocab
            if word in self.vocab:
                ids.append(self.vocab[word])
                continue

            # BPE encode
            chars = list(word)

            while len(chars) > 1:
                # Find mergeable pairs
                pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
                mergeable = [(p, self.merges[p]) for p in pairs if p in self.merges]

                if not mergeable:
                    break

                # Apply earliest learned merge
                best_pair = min(mergeable, key=lambda x: x[1])[0]

                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                        new_chars.append(chars[i] + chars[i + 1])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars

            # Convert to IDs
            for token in chars:
                ids.append(self.vocab.get(token, self.vocab["<UNK>"]))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for idx in ids:
            token = self.idx_to_token.get(idx, "<UNK>")
            if token not in self.special_tokens:
                tokens.append(token)

        # Smart joining - no space before punctuation
        result = []
        no_space_before = {",", ".", ":", ";", ")", "]", "}", "'", '"'}
        no_space_after = {"(", "[", "{", "'", '"'}

        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif token in no_space_before:
                result.append(token)
            elif result and result[-1] and result[-1][-1] in no_space_after:
                result.append(token)
            else:
                result.append(" " + token)

        return "".join(result)

    def save(self, path: str):
        """Save tokenizer to JSON."""
        data = {
            "vocab": self.vocab,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved tokenizer to {path}")

    def load(self, path: str):
        """Load tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.vocab_size = data.get("vocab_size", len(self.vocab))
        self.idx_to_token = {int(v): k for k, v in self.vocab.items()}

        # Parse merges
        self.merges = {}
        for k, v in data.get("merges", {}).items():
            if "|||" in k:
                parts = k.split("|||")
                self.merges[(parts[0], parts[1])] = v
            else:
                # Handle old format with eval
                try:
                    self.merges[eval(k)] = v
                except:
                    pass

        print(f"Loaded tokenizer: {len(self.vocab)} tokens, {len(self.merges)} merges")


if __name__ == "__main__":
    import sys

    tokenizer = CodeBPETokenizer(vocab_size=16000)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/the-stack"
    tokenizer.build_vocab(data_dir, max_examples=50000)

    # Test
    test_code = """def calculate_total(items):
    total = 0
    for item in items:
        total += item
    return total

if __name__ == "__main__":
    print(calculate_total([1, 2, 3]))
"""

    print("\n" + "=" * 60)
    print("Testing tokenizer")
    print("=" * 60)

    encoded = tokenizer.encode(test_code)
    print(f"\nEncoded ({len(encoded)} tokens): {encoded[:30]}...")

    # Show actual tokens
    tokens = [tokenizer.idx_to_token[i] for i in encoded]
    print(f"Tokens: {tokens[:30]}...")

    decoded = tokenizer.decode(encoded)
    print(f"\nDecoded:\n{decoded}")

    # Check common keywords
    print("\nKeyword encoding:")
    for kw in ["import", "def", "class", "return", "self", "__name__", "__main__"]:
        enc = tokenizer.encode(kw)
        toks = [tokenizer.idx_to_token[i] for i in enc]
        print(f"  '{kw}' -> {len(enc)} tokens: {toks}")

    # Save
    tokenizer.save("data/code_tokenizer_16k.json")
