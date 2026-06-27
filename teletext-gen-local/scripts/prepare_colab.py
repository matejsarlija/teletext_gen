#!/usr/bin/env python3
import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from tqdm import tqdm

from teletext.renderer import render_page, image_to_grid
from teletext.vocab import Vocabulary
from teletext.synthetic import generate_page
import config as cfg


@dataclass
class PageRecord:
    token_grid: np.ndarray
    page_number: int
    page_range: str
    source: str
    is_synthetic: bool
    weight: float = 1.0


def load_vocab(vocab_path: Path) -> Vocabulary:
    if vocab_path.exists():
        return Vocabulary.load(vocab_path)
    print(f"Vocabulary not found at {vocab_path}. Building full vocabulary...")
    vocab = Vocabulary.build_full(charsets=cfg.ACTIVE_CHARSETS)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)
    return vocab


def count_real_pages(raw_dir: Path) -> Counter:
    counts: Counter = Counter()
    if not raw_dir.exists():
        return counts
    for source_dir in sorted(raw_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        for json_path in source_dir.glob("*.json"):
            try:
                meta = json.loads(json_path.read_text())
                pn = meta.get('page')
                if pn is not None and 100 <= pn <= 899:
                    pr = cfg.page_to_range(pn)
                    counts[pr] += 1
            except (json.JSONDecodeError, OSError):
                continue
    return counts


def load_real_pages(raw_dir: Path, vocab: Vocabulary) -> List[PageRecord]:
    records: List[PageRecord] = []
    if not raw_dir.exists():
        print(f"  Raw data directory not found: {raw_dir}")
        return records

    source_dirs = sorted(raw_dir.iterdir())
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            continue
        source_name = source_dir.name
        json_files = sorted(source_dir.glob("*.json"))
        if not json_files:
            continue
        print(f"  Loading {source_name} ({len(json_files)} pages)...")
        for json_path in tqdm(json_files, desc=f"  {source_name}", leave=False):
            try:
                meta = json.loads(json_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            page_number = meta.get('page')
            if page_number is None or not (100 <= page_number <= 899):
                continue
            png_path = json_path.with_suffix('.png')
            if not png_path.exists():
                continue
            try:
                img = Image.open(png_path).convert('RGB')
                grid = image_to_grid(img, vocab)
            except Exception:
                continue
            pr = cfg.page_to_range(page_number)
            records.append(PageRecord(
                token_grid=grid,
                page_number=page_number,
                page_range=pr,
                source=source_name,
                is_synthetic=False,
            ))
    return records


def load_synthetic_pages(synthetic_dir: Path, vocab: Optional[Vocabulary]) -> List[PageRecord]:
    records: List[PageRecord] = []
    if not synthetic_dir.exists():
        return records
    grid_files = sorted(synthetic_dir.glob("grid_*.npy"))
    if not grid_files:
        return records
    if vocab is None:
        print(f"  Found {len(grid_files)} existing synthetic pages")
        return [PageRecord(
            token_grid=np.zeros((25, 40), dtype=np.int64),
            page_number=0, page_range='unknown',
            source='synthetic', is_synthetic=True,
        ) for _ in grid_files]
    print(f"  Loading {len(grid_files)} existing synthetic pages...")
    for path in tqdm(grid_files, desc="  synthetic", leave=False):
        try:
            grid = np.load(path)
        except Exception:
            continue
        records.append(PageRecord(
            token_grid=grid,
            page_number=0,
            page_range='unknown',
            source='synthetic',
            is_synthetic=True,
        ))
    return records


def generate_augmentation(vocab: Vocabulary, needed_per_range: Dict[str, int],
                           out_dir: Path, render: bool = False) -> List[PageRecord]:
    records: List[PageRecord] = []
    for range_name, needed in needed_per_range.items():
        if needed <= 0:
            continue
        lo, hi = cfg.PAGE_RANGES[range_name]
        print(f"  Generating {needed} synthetic pages for {range_name} ({lo}-{hi})...")
        for _ in tqdm(range(needed), desc=f"  {range_name}", leave=False):
            page_number = random.randint(lo, hi)
            grid = generate_page(vocab, page_number=page_number, page_range=range_name)
            records.append(PageRecord(
                token_grid=grid,
                page_number=page_number,
                page_range=range_name,
                source='synthetic',
                is_synthetic=True,
            ))
    return records


def compute_weights(records: List[PageRecord],
                    max_oversample_factor: int = cfg.MAX_OVERSAMPLE_FACTOR) -> np.ndarray:
    range_counts: Counter = Counter()
    for r in records:
        range_counts[r.page_range] += 1
    if not range_counts:
        return np.ones(len(records), dtype=np.float32)
    max_count = max(range_counts.values())

    weights = np.ones(len(records), dtype=np.float32)
    for i, r in enumerate(records):
        count = range_counts.get(r.page_range, 1)
        if r.is_synthetic:
            weights[i] = 1.0
        else:
            factor = min(max_count / count, float(max_oversample_factor))
            weights[i] = factor
    return weights


def stratified_split(records: List[PageRecord],
                     val_ratio: float = 0.1) -> Tuple[List[PageRecord], List[PageRecord]]:
    range_groups: Dict[str, List[PageRecord]] = defaultdict(list)
    for r in records:
        range_groups[r.page_range].append(r)

    train: List[PageRecord] = []
    val: List[PageRecord] = []
    rng = random.Random(42)

    for pr, group in range_groups.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        if n_val >= len(group):
            n_val = len(group) // 2 if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def save_dataset(out_dir: Path, train: List[PageRecord], val: List[PageRecord],
                 vocab: Vocabulary, vocab_path: Path, render: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = np.stack([r.token_grid for r in train])
    val_tokens = np.stack([r.token_grid for r in val])
    np.save(out_dir / "train_tokens.npy", train_tokens)
    np.save(out_dir / "val_tokens.npy", val_tokens)

    train_weights = np.array([r.weight for r in train], dtype=np.float32)
    val_weights = np.array([r.weight for r in val], dtype=np.float32)
    np.save(out_dir / "train_weights.npy", train_weights)
    np.save(out_dir / "val_weights.npy", val_weights)

    def _metadata(records: List[PageRecord]) -> List[dict]:
        return [
            {
                'page_number': r.page_number,
                'page_range': r.page_range,
                'source': r.source,
                'is_synthetic': r.is_synthetic,
            }
            for r in records
        ]

    train_meta = _metadata(train)
    val_meta = _metadata(val)
    (out_dir / "train_metadata.json").write_text(
        json.dumps(train_meta, indent=2, ensure_ascii=False)
    )
    (out_dir / "val_metadata.json").write_text(
        json.dumps(val_meta, indent=2, ensure_ascii=False)
    )

    if render:
        print("  Rendering training images...")
        train_images = []
        for r in tqdm(train, desc="  train images", leave=False):
            img = render_page(r.token_grid, vocab)
            train_images.append(np.array(img))
        np.save(out_dir / "train_images.npy", np.stack(train_images))

        print("  Rendering validation images...")
        val_images = []
        for r in tqdm(val, desc="  val images", leave=False):
            img = render_page(r.token_grid, vocab)
            val_images.append(np.array(img))
        np.save(out_dir / "val_images.npy", np.stack(val_images))

    shutil.copy(str(vocab_path), str(out_dir / "vocab.json"))


def print_report(records: List[PageRecord],
                 real_counts: Dict[str, int], synth_counts: Dict[str, int],
                 train_size: int, val_size: int) -> None:
    weights = np.array([r.weight for r in records], dtype=np.float32)
    range_order = list(cfg.PAGE_RANGES.keys())
    total_real = sum(real_counts.values())
    total_synth = sum(synth_counts.values())

    range_weights: Dict[str, float] = {}
    for i, r in enumerate(records):
        if r.page_range not in range_weights:
            range_weights[r.page_range] = weights[i]

    max_weight = max(weights) if len(weights) > 0 else 1.0

    print(f"\n{'=' * 70}")
    print(f"Final dataset statistics:")
    print(f"{'=' * 70}")
    print(f"{'Range':<15} {'Category':<20} {'Real':>8} {'Synth':>8} {'Weight':>8}")
    print(f"{'-' * 15} {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8}")

    for name in range_order:
        real_c = real_counts.get(name, 0)
        synth_c = synth_counts.get(name, 0)
        w = range_weights.get(name, 1.0)
        cap_flag = " (capped)" if w >= cfg.MAX_OVERSAMPLE_FACTOR else ""
        lo, hi = cfg.PAGE_RANGES[name]
        print(f"{lo:03d}-{hi:03d}     {name:<20} {real_c:>8,} {synth_c:>8,} {w:.1f}x{cap_flag}")

    print(f"{'-' * 70}")
    print(f"{'Total real pages:':<36} {total_real:>8,}")
    print(f"{'Total synthetic pages:':<36} {total_synth:>8,}")
    print(f"{'Total training pages:':<36} {train_size:>8,}")
    print(f"{'Val pages:':<36} {val_size:>8,}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for Colab upload")
    parser.add_argument("--vocab", type=str, default=str(cfg.VOCAB_PATH),
                        help=f"Vocabulary path (default: {cfg.VOCAB_PATH})")
    parser.add_argument("--out", type=str, default=str(cfg.COLAB_UPLOAD_DIR),
                        help=f"Output directory (default: {cfg.COLAB_UPLOAD_DIR})")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render token grids to images (default: True)")
    parser.add_argument("--no-render", action="store_false", dest="render",
                        help="Skip image rendering (only save tokens + vocab)")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Skip synthetic augmentation")
    parser.add_argument("--no-oversample", action="store_true",
                        help="Skip oversampling (uniform weights)")
    parser.add_argument("--threshold", type=int, default=cfg.SYNTHETIC_AUGMENTATION_THRESHOLD,
                        help=f"Min real pages per range before augmentation (default: {cfg.SYNTHETIC_AUGMENTATION_THRESHOLD})")
    parser.add_argument("--target", type=int, default=cfg.TARGET_PAGES_PER_RANGE,
                        help=f"Target pages per range (default: {cfg.TARGET_PAGES_PER_RANGE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print distribution report without saving")
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    out_dir = Path(args.out)

    print("Counting real pages from metadata (fast)...")
    real_range_counts = count_real_pages(cfg.RAW_DIR)
    total_real = sum(real_range_counts.values())
    print(f"  Found {total_real:,} real pages across {len(real_range_counts)} ranges")

    range_order = list(cfg.PAGE_RANGES.keys())
    print(f"\n{'=' * 50}")
    print("Real page distribution:")
    print(f"{'=' * 50}")
    for name in range_order:
        lo, hi = cfg.PAGE_RANGES[name]
        real_c = real_range_counts.get(name, 0)
        flag = "  *** SPARSE" if real_c < args.threshold else ""
        print(f"  {lo:03d}-{hi:03d} {name:<20} {real_c:>8,}{flag}")

    if args.dry_run:
        synth_count = len(list(cfg.SYNTHETIC_DIR.glob("grid_*.npy"))) if cfg.SYNTHETIC_DIR.exists() else 0
        print(f"  Existing synthetic pages: {synth_count:,}")
        print(f"\n{'=' * 50}")
        print("DRY RUN — no files saved")
        print(f"\nWould generate augmentation for ranges below threshold ({args.threshold})")
        for name in range_order:
            real_c = real_range_counts.get(name, 0)
            if real_c < args.threshold:
                needed = max(0, args.target - real_c)
                print(f"  {name}: +{needed} synthetic pages")
        return

    print("\nLoading vocabulary...")
    vocab = load_vocab(vocab_path)
    print(f"  {vocab.observed_count} observed token types (size={vocab.size})")

    print("\nLoading existing synthetic pages (if any)...")
    synth_records = load_synthetic_pages(cfg.SYNTHETIC_DIR, vocab)
    print(f"  Loaded {len(synth_records)} existing synthetic pages")

    print("\nLoading real pages (tokenizing from raw images)...")
    real_records = load_real_pages(cfg.RAW_DIR, vocab)
    print(f"  Loaded {len(real_records)} real pages")

    if not args.no_synthetic:
        print(f"\nChecking for sparse ranges (threshold={args.threshold}, target={args.target})...")
        needed_per_range: Dict[str, int] = {}
        for name in range_order:
            real_c = real_range_counts.get(name, 0)
            if real_c < args.threshold:
                needed = max(0, args.target - real_c)
                if needed > 0:
                    needed_per_range[name] = needed

        if needed_per_range:
            print(f"  Generating synthetic augmentation for: {list(needed_per_range.keys())}")
            aug_records = generate_augmentation(vocab, needed_per_range, out_dir, render=args.render)
            print(f"  Generated {len(aug_records)} synthetic pages")
        else:
            aug_records = []
            print(f"  No sparse ranges found — augmentation not needed")
    else:
        aug_records = []
        print(f"  Synthetic augmentation skipped (--no-synthetic)")

    all_records = real_records + synth_records + aug_records

    if args.no_oversample:
        weights = np.ones(len(all_records), dtype=np.float32)
        print("  Oversampling skipped (--no-oversample)")
    else:
        weights = compute_weights(all_records)
        print(f"  Oversampling weights computed (capped at {cfg.MAX_OVERSAMPLE_FACTOR}x)")

    for r, w in zip(all_records, weights):
        r.weight = w

    print(f"\nSplitting train/val (ratio={args.val_split})...")
    train_records, val_records = stratified_split(all_records, args.val_split)
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}")

    print(f"\nSaving dataset to {out_dir}/...")
    save_dataset(out_dir, train_records, val_records, vocab, vocab_path, render=args.render)

    aug_range_counts: Counter = Counter()
    for r in aug_records:
        aug_range_counts[r.page_range] += 1

    print_report(all_records, dict(real_range_counts), dict(aug_range_counts),
                 len(train_records), len(val_records))

    files_to_zip = [
        "train_tokens.npy", "val_tokens.npy",
        "train_weights.npy", "val_weights.npy",
        "train_metadata.json", "val_metadata.json",
        "vocab.json",
    ]
    if args.render:
        files_to_zip.extend(["train_images.npy", "val_images.npy"])

    zip_path = out_dir / "colab_dataset.zip"
    total_mb = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in files_to_zip:
            fpath = out_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
                total_mb += fpath.stat().st_size
    total_mb /= 1024 * 1024
    zip_mb = zip_path.stat().st_size / (1024 * 1024) if zip_path.exists() else 0

    print(f"{'=' * 50}")
    print(f"Dataset Summary")
    print(f"{'=' * 50}")
    print(f"  Training samples:   {len(train_records):,}")
    print(f"  Validation samples: {len(val_records):,}")
    print(f"  Grid shape:         (25, 40)")
    print(f"  Vocab size:         {vocab.size}")
    print(f"  Active charsets:    {vocab.charsets}")
    print(f"  Uncompressed size:  {total_mb:.1f} MB")
    print(f"  Zip size:           {zip_mb:.1f} MB")
    print(f"  Zip location:       {zip_path}")
    print()
    print("Upload to Colab:")
    print(f"  from google.colab import files")
    print(f"  files.upload()")
    print(f"  !unzip -q colab_dataset.zip -d /content/data/")


if __name__ == "__main__":
    main()
