#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teletext.synthetic import generate_dataset
from teletext.vocab import Vocabulary
from config import SYNTHETIC_DIR, VOCAB_PATH, DEFAULT_NATIONAL_CHARSET, ACTIVE_CHARSETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic teletext pages")
    parser.add_argument("--n", type=int, default=50000,
                        help="Number of pages to generate (default: 50000)")
    parser.add_argument("--out", type=str, default=str(SYNTHETIC_DIR),
                        help=f"Output directory (default: {SYNTHETIC_DIR})")
    parser.add_argument("--vocab", type=str, default=str(VOCAB_PATH),
                        help=f"Vocabulary file (default: {VOCAB_PATH})")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render generated grids to PNG files (default: True)")
    parser.add_argument("--no-render", action="store_false", dest="render",
                        help="Only save token grids; skip PNG rendering")
    args = parser.parse_args()

    out_dir = Path(args.out)
    vocab_path = Path(args.vocab)

    if vocab_path.exists():
        vocab = Vocabulary.load(vocab_path)
        print(f"Loaded vocabulary with {vocab.observed_count} observed token types")
    else:
        vocab = Vocabulary.build_full(charsets=ACTIVE_CHARSETS)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
        print(f"Built full vocabulary ({vocab.observed_count} token types), saved to {vocab_path}")

    print(f"Generating {args.n} synthetic pages...")
    generate_dataset(args.n, vocab, out_dir, render=args.render)
    print(f"Done! Pages saved to {out_dir}/")


if __name__ == "__main__":
    main()
