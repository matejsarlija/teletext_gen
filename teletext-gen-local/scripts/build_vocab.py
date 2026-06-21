#!/usr/bin/env python3
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from teletext.vocab import Vocabulary
from teletext.sources import SOURCES
from teletext.charsets import decode_char_with_charset
from config import (
    TOKENS_DIR, SYNTHETIC_DIR, VOCAB_PATH, H, W,
    DEFAULT_NATIONAL_CHARSET,
)


def find_grid_files(base_dirs):
    """Recursively find all grid_*.npy files under given base directories.
    Returns list of (path, source_key) tuples.
    """
    files = []
    for base in base_dirs:
        if not base.exists():
            continue
        if base.name == 'synthetic':
            for p in base.glob("grid_*.npy"):
                files.append((p, 'synthetic'))
        elif base.name == 'tokens':
            for subdir in base.iterdir():
                if subdir.is_dir():
                    for p in subdir.glob("grid_*.npy"):
                        files.append((p, subdir.name))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabulary from token grids")
    parser.add_argument("--out", type=str, default=str(VOCAB_PATH),
                        help=f"Output vocabulary path (default: {VOCAB_PATH})")
    parser.add_argument("--full", action="store_true",
                        help="Include all possible token types, not just observed")
    args = parser.parse_args()

    grid_files = find_grid_files([TOKENS_DIR, SYNTHETIC_DIR])

    if not grid_files and not args.full:
        print("No token grids found. Use --full to build a complete vocabulary.")
        return

    if args.full or not grid_files:
        vocab = Vocabulary.build_full(charsets=[DEFAULT_NATIONAL_CHARSET])
        vocab.save(Path(args.out))
        print(f"Built full vocabulary: {vocab.observed_count} token types")
        print(f"Saved to {args.out}")
        return

    print(f"Scanning {len(grid_files)} token grids...")

    vocab = Vocabulary(charsets=[DEFAULT_NATIONAL_CHARSET])
    token_counter: Counter = Counter()
    source_stats: Counter = Counter()
    source_char_counts: defaultdict = defaultdict(Counter)

    for path, src_key in grid_files:
        grid = np.load(path)
        source_stats[src_key] += 1
        for token_id in grid.flatten():
            tid = int(token_id)
            token_counter[tid] += 1
            token = vocab.id_to_token(tid)
            if token is not None:
                vocab.mark_observed(token)

    total_cells = sum(token_counter.values())
    unique_ids = len(token_counter)

    print(f"\n=== Vocabulary Statistics ===")
    print(f"Grids scanned: {len(grid_files)}")
    print(f"Total cells: {total_cells:,}")
    print(f"Unique token IDs used: {unique_ids}")
    print(f"Observed (char_id, fg, bg) combos: {vocab.observed_count}")
    print(f"Total possible combos: {(127 + 1) * 8 * 8}")
    print(f"Coverage: {100.0 * vocab.observed_count / ((127 + 1) * 8 * 8):.1f}%")

    print(f"\nGrids per source:")
    for src_key, cnt in source_stats.most_common():
        print(f"  {src_key}: {cnt}")

    color_counts = Counter()
    char_type_counts = Counter()
    for tid, count in token_counter.items():
        token = vocab.id_to_token(tid)
        if token is not None:
            color_counts[(token.fg, token.bg)] += count
            if token.char_id <= 95:
                char_type_counts['ascii'] += count
            else:
                char_type_counts['mosaic'] += count
            if token.fg == token.bg:
                char_type_counts['invisible'] += count

    print(f"\nCharacter type distribution:")
    for key in ['ascii', 'mosaic', 'invisible']:
        cnt = char_type_counts.get(key, 0)
        pct = 100.0 * cnt / total_cells if total_cells else 0
        print(f"  {key}: {cnt:,} ({pct:.1f}%)")

    most_common = token_counter.most_common(10)
    print(f"\nTop 10 most common tokens:")
    for tid, count in most_common:
        token = vocab.id_to_token(tid)
        if token:
            if token.char_id == 0:
                ch = "space"
            elif token.char_id <= 95:
                ch = repr(vocab.decode_char(token.char_id))
            else:
                ch = f"mosaic[{token.char_id - 96}]"
            print(f"  ID {tid:4d} | char={ch:10s} | fg={token.fg} bg={token.bg} | count={count:,}")

    vocab.save(Path(args.out))
    print(f"\nVocabulary saved to {args.out}")


if __name__ == "__main__":
    main()
