import torch
import argparse
from pathlib import Path

from models.transformer_d3pm import TransformerD3PM
from models.d3pm import D3PM
from data.code_tokenizer import CodeBPETokenizer
from data.code_datasets import PythonCodeDataset
from utils.code_metrics import (
    compute_syntax_validity,
    analyze_code_structure,
    evaluate_infilling_quality,
    check_syntax_valid,
)


def evaluate_code_model(
    checkpoint_path: str = None, device: str = "cpu", n_samples: int = 50
):
    """Evaluate code generation model"""

    print("=" * 60)
    print("Evaluating Code D3PM Model")
    print("=" * 60)

    print("\nLoading tokenizer...")
    tokenizer = CodeBPETokenizer()
    tokenizer.load("data/code_tokenizer.json")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    print("\nLoading test dataset...")
    dataset = PythonCodeDataset(
        data_source=".",
        tokenizer=tokenizer,
        max_seq_len=1024,
        max_files=5,
        use_cache=False,
    )
    print(f"  Test examples: {len(dataset)}")

    print("\nBuilding model...")
    network = TransformerD3PM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        max_seq_len=1024,
        dropout=0.1,
    )

    model = D3PM(
        network=network,
        mask_token_id=tokenizer.vocab["<MASK>"],
        pad_token_id=tokenizer.vocab["<PAD>"],
        schedule_type="linear",
    ).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No checkpoint - evaluating untrained model")

    print("\n" + "=" * 60)
    print("Generating Code Samples...")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        samples_tokens, _ = model.sample(
            n_samples=n_samples, seq_len=512, n_steps=50, device=device
        )

    samples_code = []
    for i in range(n_samples):
        tokens = samples_tokens[i].cpu().tolist()
        tokens = [t for t in tokens if t != tokenizer.vocab["<PAD>"]]
        code = tokenizer.decode(tokens)
        samples_code.append(code)

        if i < 3:
            print(f"\nSample {i+1}:")
            print("-" * 60)
            print(code[:300])
            print("...")
            print("-" * 60)

    print("\n" + "=" * 60)
    print("Syntax Validity Analysis")
    print("=" * 60)

    validity = compute_syntax_validity(samples_code)
    print(f"Valid samples: {validity['valid_count']}/{validity['total']}")
    print(f"Validity rate: {validity['validity_rate']:.2%}")

    if validity["syntax_errors"]:
        print(f"\nFirst syntax error:")
        err = validity["syntax_errors"][0]
        print(f"  Sample {err['sample_idx']}: {err['error']}")

    print("\n" + "=" * 60)
    print("Code Structure Analysis")
    print("=" * 60)

    valid_samples = [s for s in samples_code if check_syntax_valid(s)]

    if valid_samples:
        structures = [analyze_code_structure(s) for s in valid_samples[:10]]

        avg_funcs = sum(s["num_functions"] for s in structures) / len(structures)
        avg_classes = sum(s["num_classes"] for s in structures) / len(structures)
        avg_lines = sum(s["lines"] for s in structures) / len(structures)

        print(f"Average per sample (from {len(structures)} valid samples):")
        print(f"  Functions: {avg_funcs:.1f}")
        print(f"  Classes: {avg_classes:.1f}")
        print(f"  Lines: {avg_lines:.1f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Syntax Validity: {validity['validity_rate']:.1%}")
    print(f"Valid Samples: {validity['valid_count']}/{n_samples}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_samples", type=int, default=50)

    args = parser.parse_args()

    evaluate_code_model(
        checkpoint_path=args.checkpoint, device=args.device, n_samples=args.n_samples
    )
