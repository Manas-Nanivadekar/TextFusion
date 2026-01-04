# TextFusion

Research implementation of discrete and continuous diffusion models for text and code generation.

## Overview

TextFusion implements modern diffusion-based generative models adapted for discrete sequences (text, code) and continuous data (images, 2D distributions). Unlike autoregressive models that generate left-to-right, diffusion models can leverage bidirectional context and support parallel decoding.

**Key implementations:**
- **D3PM** (Discrete Denoising Diffusion Probabilistic Models) - Masking-based diffusion for text/code
- **DDPM** (Denoising Diffusion Probabilistic Models) - Gaussian noise diffusion for continuous data
- **Flow Matching** - Alternative to diffusion using continuous normalizing flows

## Why Diffusion for Text?

Traditional approach: Autoregressive models (GPT, BERT-style masked LM)
- Generate sequentially, no parallelism at inference
- Left-to-right bias, limited bidirectional reasoning
- Exposure bias during training

Diffusion approach (D3PM):
- Iterative refinement from random mask to coherent text
- Bidirectional context at every step
- Learned generation order (high-confidence tokens first)
- Supports infilling naturally

Trade-offs:
- **Pro**: Better for tasks requiring global coherence (code, structured text)
- **Pro**: Flexible generation order, conditional generation
- **Con**: Slower inference (requires multiple denoising steps)
- **Con**: Less mature than autoregressive models for general text

## Architecture

```
TextFusion/
├── models/
│   ├── d3pm.py              # D3PM discrete diffusion
│   ├── transformer_d3pm.py  # Time-conditioned transformer
│   ├── masking.py           # Masking schedules & utilities
│   ├── ddpm.py              # Continuous diffusion (2D/images)
│   └── flow_matching*.py    # Flow matching variants
├── training/
│   ├── trainer_d3pm.py      # D3PM training loop
│   └── schedulers.py        # Noise schedules (linear, cosine)
├── utils/
│   ├── metrics.py           # Perplexity, diversity metrics
│   └── code_metrics.py      # Code-specific evaluation
├── data/
│   └── code_datasets.py     # Python code dataset loaders
└── experiments/
    ├── train_code_d3pm.py   # Train D3PM on code
    └── evaluate_code.py     # Evaluate code generation
```

## Research Background

### D3PM: Discrete Diffusion
**Paper**: Austin et al. "Structured Denoising Diffusion Models in Discrete State Spaces" (NeurIPS 2021)
**Link**: https://arxiv.org/abs/2107.03006

Forward process: Progressively mask tokens with `[MASK]` token
Reverse process: Iteratively unmask based on model confidence

Key innovation: Confidence-based unmasking order emerges from training (e.g., function names before variable names in code)

### DDPM: Continuous Diffusion
**Paper**: Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
**Link**: https://arxiv.org/abs/2006.11239

Forward process: Add Gaussian noise according to schedule
Reverse process: Iteratively denoise using learned ε-prediction

Simplified training objective: `L = ||ε - ε_θ(√ᾱ_t x₀ + √(1-ᾱ_t) ε, t)||²`

### Improved Schedules
**Paper**: Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (ICML 2021)
**Link**: https://arxiv.org/abs/2102.09672

Cosine noise schedule: Smoother corruption, better for high-res data

## Installation

```bash
# Clone repository
git clone <repo-url>
cd TextFusion

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm tensorboard
pip install tokenizers  # For BPE tokenization
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Quick Start

### Train D3PM on Code

```python
from models.transformer_d3pm import TransformerD3PM
from models.d3pm import D3PM
from data.code_tokenizer import CodeBPETokenizer
from data.code_datasets import PythonCodeDataset
from training.trainer_d3pm import D3PMTrainer
import torch.optim as optim

# Load tokenizer
tokenizer = CodeBPETokenizer()
tokenizer.train_from_directory("path/to/python/code", vocab_size=5000)

# Create dataset
dataset = PythonCodeDataset(
    data_source="path/to/python/code",
    tokenizer=tokenizer,
    max_seq_len=512
)

# Build model
network = TransformerD3PM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    ff_dim=1024,
    max_seq_len=512
)

model = D3PM(
    network=network,
    mask_token_id=tokenizer.vocab["<MASK>"],
    pad_token_id=tokenizer.vocab["<PAD>"],
    schedule_type="linear"  # or "cosine", "sqrt"
)

# Train
trainer = D3PMTrainer(
    model=model,
    train_loader=DataLoader(dataset, batch_size=32),
    optimizer=optim.AdamW(model.parameters(), lr=1e-4),
    device="cuda"
)

trainer.train(epochs=50)
```

### Generate Code Samples

```python
# Load trained model
checkpoint = torch.load("outputs/checkpoints_d3pm/best.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Generate
samples, trajectory = model.sample(
    n_samples=5,
    seq_len=256,
    n_steps=50,  # More steps = higher quality
    return_trajectory=True,
    device="cuda"
)

# Decode
for i, sample in enumerate(samples):
    tokens = sample.cpu().tolist()
    code = tokenizer.decode(tokens)
    print(f"Sample {i+1}:\n{code}\n")
```

## Evaluation

### Code Generation Quality

```bash
python experiments/evaluate_code.py \
    --checkpoint outputs/checkpoints_d3pm/best.pt \
    --n_samples 100 \
    --device cuda
```

**Metrics:**
- **Syntax validity**: % of samples that parse successfully (minimum bar)
- **Structural analysis**: Distribution of functions, classes, loops
- **Diversity metrics**: Unique n-grams, Self-BLEU (detect mode collapse)
- **Perplexity**: Model's predictive quality (approximate for diffusion)

### Expected Results

Early training (syntax validity ~10-30%):
- Model generates token-level patterns but not structure
- High repetition, incomplete syntax

Mid training (syntax validity ~60-80%):
- Valid Python syntax, simple functions
- Some semantic errors, unusual patterns

Late training (syntax validity >80%):
- Coherent functions/classes with proper structure
- Still may have logical errors (requires semantic evaluation)

## Key Hyperparameters

### Model Architecture
- `embed_dim`: 256-512 for code, 768-1024 for general text
- `num_layers`: 6-12 (more layers = more capacity, slower)
- `num_heads`: 8-16 (must divide embed_dim)
- `ff_dim`: 4x embed_dim typically

### Training
- `schedule_type`: "linear" (simple), "cosine" (smoother, often better)
- `learning_rate`: 1e-4 to 3e-4 (AdamW)
- `batch_size`: 32-128 (larger better, limited by GPU memory)
- `max_seq_len`: 256-1024 (longer = more context, more memory)

### Sampling
- `n_steps`: 20-100 (more steps = higher quality but slower)
  - DDPM paper uses 1000, but 50 often sufficient
  - Quality plateaus around 100 steps
- `schedule_type`: Should match training schedule

## Design Decisions

### Why Pre-LayerNorm?
Post-norm (original Transformer) has gradient flow issues in deep networks. Pre-norm (GPT-2, GPT-3) is more stable. See Xiong et al. "On Layer Normalization in the Transformer Architecture" (2020).

### Why Gradient Clipping?
Diffusion models can have exploding gradients, especially early in training when predictions are poor. Clipping at norm=1.0 prevents instability.

### Why Not Importance Sampling for Timesteps?
We sample t uniformly in [0,1] for simplicity. Improved DDPM shows importance sampling can help (weight loss by SNR), but adds complexity and negligible benefit for discrete diffusion.

### Why Confidence-Based Unmasking?
Alternative: Random unmasking (like forward process in reverse). Confidence-based allows model to learn generation order, improving coherence. Empirically works better for structured data (code, formulas).

## Limitations

1. **Inference Speed**: 50 steps × forward pass slower than autoregressive single pass
   - Mitigation: Distillation, fewer steps, parallel sampling

2. **Likelihood Evaluation**: No tractable likelihood unlike autoregressive
   - Mitigation: Use proxy metrics (perplexity at t=0.5, syntax validity)

3. **Small Training Datasets**: Diffusion models benefit from large scale
   - Mitigation: Careful regularization, smaller models, data augmentation

4. **Discrete Space Challenges**: Masks less expressive than continuous noise
   - Ongoing research: Alternative discrete corruption processes

## Experiments

### Toy Experiments (Quick Iteration)
```bash
# 2D distributions (visualize learning)
python experiments/train_ddpm_2d.py

# MNIST (validate architecture)
python experiments/train_flow_matching_mnist.py
```

### Text Generation
```bash
# Shakespeare (small, fast)
python experiments/train_d3pm_shakespeare.py

# Evaluate quality
python experiments/evaluate_text.py --checkpoint <path>
```

### Code Generation
```bash
# Python code (main use case)
python experiments/train_code_d3pm.py

# Evaluate syntax validity, structure
python experiments/evaluate_code.py --checkpoint <path>
```

## Citation

If you use this code for research, please cite the original papers:

```bibtex
@inproceedings{austin2021structured,
  title={Structured Denoising Diffusion Models in Discrete State Spaces},
  author={Austin, Jacob and Johnson, Daniel D and Ho, Jonathan and Tarlow, Daniel and van den Berg, Rianne},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={NeurIPS},
  year={2020}
}
```

## Contributing

This is a research codebase. Contributions welcome for:
- New diffusion variants (absorbing states, multimodal corruption)
- Improved evaluation metrics (semantic code evaluation)
- Optimization (faster sampling, distillation)
- Alternative architectures (diffusion transformers, U-Nets for sequences)

Keep code clean, add docstrings following project style, include tests for new metrics.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- D3PM implementation inspired by [google-research/google-research](https://github.com/google-research/google-research/tree/master/d3pm)
- DDPM implementation follows [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- Transformer architecture based on PyTorch reference implementations

---

**Research Contact**: For questions about the diffusion formulation or experimental results, open an issue with the `research` label.

**Bug Reports**: For implementation bugs, open an issue with the `bug` label and include minimal reproduction code.
