import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from models.transformer_d3pm import TransformerD3PM
from models.d3pm import D3PM
from data.text_datasets import ShakespeareDataset
from utils.metrics import (
    compute_perplexity,
    compute_diversity_metrics,
    compute_sample_perplexity,
)


def evaluate_model(
    checkpoint_path: str = None, device: str = "cpu", n_samples: int = 100
):

    print("=" * 60)
    print("Evaluating D3PM Text Model")
    print("=" * 60)

    print("\nLoading dataset...")
    dataset = ShakespeareDataset(
        data_dir="data/shakespeare", max_seq_len=64, vocab_size=3000
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("\nBuilding model...")
    network = TransformerD3PM(
        vocab_size=dataset.vocab_size,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        max_seq_len=66,
        dropout=0.1,
    )

    model = D3PM(
        network=network,
        mask_token_id=dataset.mask_token_id,
        pad_token_id=dataset.pad_token_id,
        schedule_type="linear",
    ).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint found - evaluating untrained model")

    print("\n" + "=" * 60)
    print("1. Computing Perplexity...")
    print("=" * 60)

    perplexity = compute_perplexity(model, dataloader, device)
    print(f"Perplexity: {perplexity:.2f}")

    if perplexity < 50:
        print("  ✓ Excellent! Model understands the data well")
    elif perplexity < 100:
        print("  ✓ Good! Model has learned patterns")
    elif perplexity < 200:
        print("  ~ Okay. Model is learning but needs more training")
    else:
        print("  ✗ High perplexity. Model is still learning")

    print("\n" + "=" * 60)
    print("2. Generating Samples...")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        samples_tokens, _ = model.sample(
            n_samples=n_samples, seq_len=64, n_steps=50, device=device
        )

    samples_text = []
    for i in range(min(10, n_samples)):
        tokens = samples_tokens[i].cpu().tolist()
        text = dataset.decode(tokens)
        samples_text.append(text)

        if i < 5:
            print(f"\nSample {i+1}:")
            print(f"  {text[:100]}...")

    all_samples_text = [
        dataset.decode(samples_tokens[i].cpu().tolist()) for i in range(n_samples)
    ]

    print("\n" + "=" * 60)
    print("3. Diversity Metrics...")
    print("=" * 60)

    diversity = compute_diversity_metrics(all_samples_text)

    print(f"Unique unigrams: {diversity['unique_unigrams']}")
    print(f"Unique bigrams: {diversity['unique_bigrams']}")
    print(f"Unique trigrams: {diversity['unique_trigrams']}")
    print(f"Unigram diversity: {diversity['unigram_diversity']:.4f}")
    print(f"Bigram diversity: {diversity['bigram_diversity']:.4f}")
    print(f"Repetition rate: {diversity['repetition_rate']:.4f}")
    print(f"Self-BLEU: {diversity['self_bleu']:.4f}")
    print(f"Vocab coverage: {diversity['vocab_size']} words")
    print(f"Avg length: {diversity['avg_length']:.1f} tokens")

    if diversity["repetition_rate"] < 0.05:
        print("\n  ✓ Low repetition - good diversity!")
    elif diversity["repetition_rate"] < 0.15:
        print("\n  ~ Some repetition - could be better")
    else:
        print("\n  ✗ High repetition - model is stuck in loops")

    print("\n" + "=" * 60)
    print("4. Sample Quality (Perplexity)...")
    print("=" * 60)

    sample_ppl = compute_sample_perplexity(all_samples_text, dataset)
    print(f"Sample perplexity: {sample_ppl:.2f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model Perplexity: {perplexity:.2f}")
    print(f"Sample Diversity: {diversity['unigram_diversity']:.3f}")
    print(f"Sample Quality: {sample_ppl:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_samples", type=int, default=100)

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint, device=args.device, n_samples=args.n_samples
    )
