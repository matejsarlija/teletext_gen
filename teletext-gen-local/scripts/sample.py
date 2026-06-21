#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from teletext.renderer import render_page
from teletext.vocab import Vocabulary
from config import W, H, PAD_ID, START_ID, VOCAB_PATH


def sample_model(model: torch.nn.Module, num_pages: int,
                 temperature: float, top_k: int,
                 max_length: int, device: torch.device,
                 start_token_id: int = START_ID,
                 pad_token_id: int = PAD_ID) -> list:
    """Run autoregressive sampling."""
    model.eval()
    samples = []

    row_ids = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(1, -1)
    col_ids = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(1, -1)

    for i in range(num_pages):
        tokens = torch.full((1, max_length), pad_token_id, dtype=torch.long, device=device)
        tokens[0, 0] = start_token_id

        for pos in range(1, max_length):
            with torch.no_grad():
                logits = model(
                    tokens[:, :pos],
                    row_ids[:, :pos],
                    col_ids[:, :pos],
                )
            next_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[-1]] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens[0, pos] = next_token

        samples.append(tokens[0, :H * W].reshape(H, W).cpu().numpy())

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from trained generator")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to generator weights (.pt)")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of pages to generate (default: 16)")
    parser.add_argument("--out", type=str, default="samples",
                        help="Output directory (default: samples/)")
    parser.add_argument("--vocab", type=str, default=str(VOCAB_PATH),
                        help=f"Vocabulary path (default: {VOCAB_PATH})")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k filtering, 0 = disabled (default: 0)")
    parser.add_argument("--display", action="store_true",
                        help="Show results in a matplotlib grid")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"Vocabulary not found at {vocab_path}")
        return
    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded vocabulary: {vocab.observed_count} observed/total {vocab.size}")

    device = torch.device("cpu")
    from models.architectures import TeletextGenerator, GeneratorConfig

    config = GeneratorConfig(vocab_size=vocab.size)
    model = TeletextGenerator(config).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded weights from {args.weights}")

    print(f"Sampling {args.n} pages (temp={args.temperature}, top-k={args.top_k})...")
    samples = sample_model(model, args.n, args.temperature, args.top_k,
                           H * W, device)

    for i, grid in enumerate(samples):
        img = render_page(grid, vocab)
        img.save(out_dir / f"sample_{i:04d}.png")

    print(f"Saved {args.n} samples to {out_dir}/")

    if args.display:
        import matplotlib.pyplot as plt
        ncols = min(4, args.n)
        nrows = (args.n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i in range(args.n):
            axes[i].imshow(np.array(Image.open(out_dir / f"sample_{i:04d}.png")))
            axes[i].axis("off")
        for i in range(args.n, len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "grid.png", dpi=150)
        print(f"Grid preview saved to {out_dir}/grid.png")


if __name__ == "__main__":
    main()
