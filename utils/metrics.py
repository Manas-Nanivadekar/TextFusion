import torch
import torch.nn as nn
import math
from typing import List, Dict
from collections import Counter
import numpy as np


def compute_perplexity(model, dataloader, device: str = "cpu") -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                tokens = batch[0].to(device)
            else:
                tokens = batch.to(device)

            batch_size, seq_len = tokens.shape

            t = torch.rand(batch_size, device=device) * 0.5

            loss = model.compute_loss(tokens, t)

            non_pad = (tokens != 0).sum().item()

            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    model.train()

    return perplexity


def compute_perplexity_autoregressive(model, dataloader, device: str = "cpu") -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=0)

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch.to(device)

            if hasattr(model, "network"):
                pass
            else:
                logits = model(tokens[:, :-1])
                targets = tokens[:, 1:]

                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)

                loss = criterion(logits_flat, targets_flat)

                non_pad = (targets != 0).sum().item()

                total_loss += loss.item()
                total_tokens += non_pad

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    model.train()

    return perplexity


def compute_sample_perplexity(samples: List[str], dataset) -> float:
    bigram_counts = Counter()
    unigram_counts = Counter()

    for sample in samples:
        tokens = sample.split()
        for i in range(len(tokens)):
            unigram_counts[tokens[i]] += 1
            if i > 0:
                bigram = (tokens[i - 1], tokens[i])
                bigram_counts[bigram] += 1

    total_log_prob = 0
    total_bigrams = 0

    for bigram, count in bigram_counts.items():
        prev_token, curr_token = bigram

        prob = count / unigram_counts[prev_token]

        total_log_prob += count * math.log(prob)
        total_bigrams += count

    avg_log_prob = total_log_prob / total_bigrams
    perplexity = math.exp(-avg_log_prob)

    return perplexity


def compute_diversity_metrics(samples: List[str]) -> Dict[str, float]:
    if not samples:
        return {}

    all_tokens = []
    for sample in samples:
        tokens = sample.lower().split()
        all_tokens.append(tokens)

    unigrams = set()
    bigrams = set()
    trigrams = set()

    total_unigrams = 0
    total_bigrams = 0
    total_trigrams = 0

    for tokens in all_tokens:
        for i, tok in enumerate(tokens):
            unigrams.add(tok)
            total_unigrams += 1

            if i > 0:
                bigrams.add((tokens[i - 1], tok))
                total_bigrams += 1

            if i > 1:
                trigrams.add((tokens[i - 2], tokens[i - 1], tok))
                total_trigrams += 1

    repetitions = 0
    total_words = 0

    for tokens in all_tokens:
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                repetitions += 1
            total_words += 1

    repetition_rate = repetitions / max(total_words, 1)

    self_bleu = compute_self_bleu(all_tokens)

    vocab_size = len(unigrams)

    return {
        "unique_unigrams": len(unigrams),
        "unique_bigrams": len(bigrams),
        "unique_trigrams": len(trigrams),
        "unigram_diversity": len(unigrams) / max(total_unigrams, 1),
        "bigram_diversity": len(bigrams) / max(total_bigrams, 1),
        "trigram_diversity": len(trigrams) / max(total_trigrams, 1),
        "repetition_rate": repetition_rate,
        "self_bleu": self_bleu,
        "vocab_size": vocab_size,
        "avg_length": np.mean([len(tokens) for tokens in all_tokens]),
    }


def compute_self_bleu(tokenized_samples: List[List[str]], n: int = 4) -> float:
    if len(tokenized_samples) < 2:
        return 0.0

    bleu_scores = []

    for i, sample in enumerate(tokenized_samples):
        references = [
            tokenized_samples[j] for j in range(len(tokenized_samples)) if j != i
        ]

        score = simple_bleu(sample, references)
        bleu_scores.append(score)

    return np.mean(bleu_scores)


def simple_bleu(candidate: List[str], references: List[List[str]]) -> float:
    candidate_ngrams = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}

    for i in range(len(candidate)):
        for n in range(1, 5):
            if i + n <= len(candidate):
                ngram = tuple(candidate[i : i + n])
                candidate_ngrams[n][ngram] += 1

    ref_ngrams = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}

    for ref in references:
        for i in range(len(ref)):
            for n in range(1, 5):
                if i + n <= len(ref):
                    ngram = tuple(ref[i : i + n])
                    ref_ngrams[n][ngram] += 1

    precisions = []

    for n in range(1, 5):
        matches = 0
        total = 0

        for ngram, count in candidate_ngrams[n].items():
            matches += min(count, ref_ngrams[n].get(ngram, 0))
            total += count

        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0.0)

    if all(p > 0 for p in precisions):
        bleu = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        bleu = 0.0

    return bleu


if __name__ == "__main__":
    print("Testing perplexity metrics...")

    from torch.utils.data import TensorDataset, DataLoader

    dummy_data = torch.randint(5, 100, (100, 32))
    dataset = TensorDataset(dummy_data)
    loader = DataLoader(dataset, batch_size=16)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Linear(32, 32)

        def compute_loss(self, tokens, t=None):
            return torch.tensor(2.5)

    model = DummyModel()

    ppl = compute_perplexity(model, loader)
    print(f"Dummy perplexity: {ppl:.2f}")

    samples = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
    ]

    sample_ppl = compute_sample_perplexity(samples, None)
    print(f"Sample perplexity: {sample_ppl:.2f}")

    print("\nTesting diversity metrics...")

    diverse_samples = [
        "the cat sat on the mat",
        "a dog ran through the park",
        "birds fly in the sky",
        "fish swim in the ocean",
    ]

    div_metrics = compute_diversity_metrics(diverse_samples)
    print(f"\nDiverse samples:")
    for key, val in div_metrics.items():
        print(f"  {key}: {val:.4f}")

    repetitive_samples = [
        "the the the the the",
        "cat cat cat cat cat",
        "the cat the cat the cat",
    ]

    rep_metrics = compute_diversity_metrics(repetitive_samples)
    print(f"\nRepetitive samples:")
    for key, val in rep_metrics.items():
        print(f"  {key}: {val:.4f}")

    print("\n Diversity metrics work!")
