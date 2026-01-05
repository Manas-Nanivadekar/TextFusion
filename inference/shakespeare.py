import argparse
import torch
import torch.nn.functional as F


def sample_tokens(logits: torch.Tensor, temperature: float, top_p: float, top_k: int):
    """Sample from logits with temperature, top-k, and nucleus filtering."""
    logits = logits / temperature

    if top_k > 0:
        topk_vals, _ = logits.topk(top_k, dim=-1)
        logits = logits.masked_fill(logits < topk_vals[..., -1:], float("-inf"))

    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        sorted_probs[(cumsum - sorted_probs) > top_p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        probs = probs.scatter(-1, sorted_idx, sorted_probs)

    flat = probs.view(-1, probs.size(-1)).clamp(min=1e-10)
    samples = torch.multinomial(flat / flat.sum(-1, keepdim=True), 1).view(
        probs.shape[:-1]
    )
    confidence = probs.gather(-1, samples.unsqueeze(-1)).squeeze(-1)

    return samples, confidence


def unmask_step(
    tokens,
    logits,
    k,
    mask_id,
    num_special=5,
    temperature=1.0,
    top_p=0.95,
    top_k=0,
    rep_penalty=1.0,
):
    """Unmask k most confident positions."""
    B, L, V = logits.shape
    logits = logits.clone()
    logits[..., :num_special] = float("-inf")

    if rep_penalty != 1.0:
        for i in range(B):
            seen = tokens[i][
                (tokens[i] != mask_id) & (tokens[i] >= num_special)
            ].unique()
            for t in seen:
                logits[i, :, t] /= (
                    rep_penalty if logits[i, :, t].mean() > 0 else (1 / rep_penalty)
                )

    predicted, confidence = sample_tokens(logits, temperature, top_p, top_k)
    confidence = confidence.masked_fill(tokens != mask_id, float("-inf"))

    result = tokens.clone()
    for i in range(B):
        masked_pos = (tokens[i] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_pos) == 0:
            continue
        _, top_idx = confidence[i, masked_pos].topk(min(k, len(masked_pos)))
        result[i, masked_pos[top_idx]] = predicted[i, masked_pos[top_idx]]

    return result


@torch.no_grad()
def generate(model, seq_len, n_samples, n_steps, mask_id=1, **kwargs):
    """Generate via iterative unmasking."""
    device = next(model.parameters()).device
    tokens = torch.full((n_samples, seq_len), mask_id, dtype=torch.long, device=device)

    for step in range(n_steps):
        t = 1.0 - step / n_steps
        k = max(int(seq_len * t) - int(seq_len * (1.0 - (step + 1) / n_steps)), 1)
        logits = model(tokens, torch.full((n_samples,), t, device=device))
        tokens = unmask_step(tokens, logits, k, mask_id, **kwargs)

    return tokens


def load_model(path, device):
    """Load model, infer architecture from weights."""
    from models.transformer_d3pm import TransformerD3PM

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k[8:]: v for k, v in state.items() if k.startswith("network.")} or state

    cfg = dict(
        vocab_size=state["token_embedding.weight"].shape[0],
        embed_dim=state["pos_embedding"].shape[2],
        max_seq_len=state["pos_embedding"].shape[1],
        ff_dim=state["transformer.layers.0.linear1.weight"].shape[0],
        num_layers=len({k.split(".")[2] for k in state if "transformer.layers." in k}),
        num_heads=8,
        dropout=0.0,
    )
    cfg["time_emb_dim"] = cfg["embed_dim"]

    model = TransformerD3PM(**cfg).to(device)
    model.load_state_dict(state)
    return model.eval(), cfg


def load_vocab():
    """Load vocab from dataset."""
    from data.text_datasets import ShakespeareDataset

    ds = ShakespeareDataset(
        data_dir="data/shakespeare", max_seq_len=64, vocab_size=3000
    )
    return {v: k for k, v in ds.vocab.items()}


def decode(tokens, vocab, num_special=5):
    """Decode tokens to text."""
    return [
        " ".join(vocab.get(t, f"[{t}]") for t in seq if t >= num_special)
        for seq in tokens.cpu().tolist()
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoints_d3pm/best.pt")
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.92)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--rep_penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    vocab = load_vocab()

    seq_len = min(args.seq_len, cfg["max_seq_len"] - 2)

    print(f"Device: {device} | Seq: {seq_len} | Steps: {args.n_steps}")
    print(f"temp={args.temperature} top_p={args.top_p} rep={args.rep_penalty}\n")

    tokens = generate(
        model,
        seq_len,
        args.n_samples,
        args.n_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        rep_penalty=args.rep_penalty,
    )

    for i, text in enumerate(decode(tokens, vocab), 1):
        print(f"[{i}] {text}\n")


if __name__ == "__main__":
    main()
