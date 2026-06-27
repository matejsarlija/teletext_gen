#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from teletext.vocab import Vocabulary
from teletext.sources import SOURCES
from teletext.charsets import decode_char_with_charset
from config import (
    TOKENS_DIR, SYNTHETIC_DIR, VOCAB_PATH, H, W, PAGE_RANGES,
    page_to_range, SYNTHETIC_AUGMENTATION_THRESHOLD,
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


def _extract_page_number(path: Path, src_key: str) -> Optional[int]:
    m = re.search(r'grid_(\d+)_(\d+)', path.stem)
    if m:
        return int(m.group(1))
    m = re.search(r'grid_(\d+)', path.stem)
    if m:
        return int(m.group(1))
    return None


def _compute_distribution(grid_files: List[Tuple[Path, str]]) -> Dict:
    range_counts: Dict[str, int] = {name: 0 for name in PAGE_RANGES}
    range_sources: Dict[str, set] = {name: set() for name in PAGE_RANGES}
    range_real: Dict[str, int] = {name: 0 for name in PAGE_RANGES}
    range_synthetic: Dict[str, int] = {name: 0 for name in PAGE_RANGES}
    total_real = 0
    total_synthetic = 0
    unknown_count = 0

    for path, src_key in grid_files:
        page_number = _extract_page_number(path, src_key)
        if page_number is None:
            unknown_count += 1
            continue
        pr = page_to_range(page_number)
        if pr == 'unknown':
            unknown_count += 1
            continue

        range_counts[pr] = range_counts.get(pr, 0) + 1
        if src_key == 'synthetic':
            range_synthetic[pr] = range_synthetic.get(pr, 0) + 1
            total_synthetic += 1
        else:
            range_real[pr] = range_real.get(pr, 0) + 1
            range_sources[pr].add(src_key)
            total_real += 1

    return {
        'range_counts': dict(range_counts),
        'range_real': dict(range_real),
        'range_synthetic': dict(range_synthetic),
        'range_sources': {k: sorted(v) for k, v in range_sources.items()},
        'total_real': total_real,
        'total_synthetic': total_synthetic,
        'unknown_count': unknown_count,
    }


def print_distribution_report(dist: Dict) -> None:
    range_counts = dist['range_counts']
    range_real = dist['range_real']
    range_synthetic = dist['range_synthetic']
    range_sources = dist['range_sources']

    print(f"\n{'=' * 70}")
    print(f"Page Range Distribution Report")
    print(f"{'=' * 70}")
    print(f"{'Page Range':<15} {'Category':<20} {'Real':>10} {'Sources':<30}")
    print(f"{'-' * 15} {'-' * 20} {'-' * 10} {'-' * 30}")

    sparse_ranges = []
    for name, (lo, hi) in PAGE_RANGES.items():
        real = range_real.get(name, 0)
        synth = range_synthetic.get(name, 0)
        sources = range_sources.get(name, [])
        sources_str = ", ".join(sources) if sources else "-"
        flag = "  *** SPARSE" if real < SYNTHETIC_AUGMENTATION_THRESHOLD else ""
        if real < SYNTHETIC_AUGMENTATION_THRESHOLD:
            sparse_ranges.append(name)
        print(f"{lo:03d}-{hi:03d}     {name:<20} {real:>8,}  {sources_str:<30}{flag}")

    print(f"{'-' * 70}")
    print(f"{'Total real pages:':<36} {dist['total_real']:>8,}")
    print(f"{'Total synthetic pages:':<36} {dist['total_synthetic']:>8,}")
    print(f"{'Unknown:':<36} {dist['unknown_count']:>8,}")

    if sparse_ranges:
        print(f"\n{'!' * 70}")
        print(f"SPARSE RANGES (below threshold of {SYNTHETIC_AUGMENTATION_THRESHOLD}):")
        for name in sparse_ranges:
            lo, hi = PAGE_RANGES[name]
            real = range_real.get(name, 0)
            needed = max(0, SYNTHETIC_AUGMENTATION_THRESHOLD - real)
            print(f"  {lo:03d}-{hi:03d} ({name}) — {real} real, needs {needed} more pages")
        print(f"{'!' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabulary from token grids")
    parser.add_argument("--out", type=str, default=str(VOCAB_PATH),
                        help=f"Output vocabulary path (default: {VOCAB_PATH})")
    parser.add_argument("--full", action="store_true",
                        help="Include all possible token types, not just observed")
    parser.add_argument("--report-only", action="store_true",
                        help="Only print distribution report, skip vocab building")
    args = parser.parse_args()

    grid_files = find_grid_files([TOKENS_DIR, SYNTHETIC_DIR])

    if args.report_only:
        dist = _compute_distribution(grid_files)
        print_distribution_report(dist)
        report_path = TOKENS_DIR.parent / "distribution_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(dist, indent=2, ensure_ascii=False))
        print(f"\nReport saved to {report_path}")
        return

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

    dist = _compute_distribution(grid_files)
    print_distribution_report(dist)
    report_path = TOKENS_DIR.parent / "distribution_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dist, indent=2, ensure_ascii=False))
    print(f"\nDistribution report saved to {report_path}")

    vocab.save(Path(args.out))
    print(f"\nVocabulary saved to {args.out}")


if __name__ == "__main__":
    main()
