#!/usr/bin/env python3
import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from teletext.renderer import render_page
from teletext.vocab import Vocabulary
from teletext.sources import SOURCES
from config import TOKENS_DIR, SYNTHETIC_DIR, COLAB_UPLOAD_DIR, VOCAB_PATH, DEFAULT_NATIONAL_CHARSET


def find_grid_files(base_dirs):
    """Recursively find all grid_*.npy files under given base directories.
    Returns list of (path, source_key) tuples.
    """
    files = []
    for base in base_dirs:
        if not base.exists():
            continue
        if base.name == 'synthetic':
            for p in sorted(base.glob("grid_*.npy")):
                files.append((p, 'synthetic'))
        elif base.name == 'tokens':
            for subdir in sorted(base.iterdir()):
                if subdir.is_dir():
                    for p in sorted(subdir.glob("grid_*.npy")):
                        files.append((p, subdir.name))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for Colab upload")
    parser.add_argument("--vocab", type=str, default=str(VOCAB_PATH),
                        help=f"Vocabulary path (default: {VOCAB_PATH})")
    parser.add_argument("--out", type=str, default=str(COLAB_UPLOAD_DIR),
                        help=f"Output directory (default: {COLAB_UPLOAD_DIR})")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render token grids to images (default: True)")
    parser.add_argument("--no-render", action="store_false", dest="render",
                        help="Skip image rendering (only save tokens + vocab)")
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vocab_path.exists():
        print(f"Vocabulary not found at {vocab_path}. Building full vocabulary...")
        vocab = Vocabulary.build_full(charsets=[DEFAULT_NATIONAL_CHARSET])
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
    else:
        vocab = Vocabulary.load(vocab_path)

    print(f"Vocabulary: {vocab.observed_count} observed token types (size={vocab.size})")
    print(f"Active charsets: {vocab.charsets}")

    grid_files = find_grid_files([TOKENS_DIR, SYNTHETIC_DIR])

    if not grid_files:
        print("No token grids found. Generate some first.")
        return

    source_counts = Counter()
    print(f"Loading {len(grid_files)} grids...")
    all_grids = []
    source_ids = []

    for path, src_key in tqdm(grid_files, desc="Loading"):
        grid = np.load(path)
        all_grids.append(grid)
        source_counts[src_key] += 1
        source_id = 0 if src_key == 'synthetic' else (
            list(SOURCES.keys()).index(src_key) + 1 if src_key in SOURCES else 99
        )
        source_ids.append(source_id)

    all_grids = np.stack(all_grids)
    source_ids = np.array(source_ids)
    N = len(all_grids)
    val_size = max(1, int(N * args.val_split))

    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_grids = all_grids[train_indices]
    val_grids = all_grids[val_indices]

    print(f"\nGrids per source:")
    for src, cnt in source_counts.most_common():
        print(f"  {src}: {cnt}")

    print(f"\nTrain: {len(train_grids)}, Val: {len(val_grids)}")
    np.save(out_dir / "train_tokens.npy", train_grids)
    np.save(out_dir / "val_tokens.npy", val_grids)

    if args.render:
        def render_all(grids, desc):
            images = []
            for g in tqdm(grids, desc=desc):
                img = render_page(g, vocab)
                images.append(np.array(img))
            return np.stack(images)

        print("Rendering training images...")
        train_images = render_all(train_grids, "Train")
        np.save(out_dir / "train_images.npy", train_images)

        print("Rendering validation images...")
        val_images = render_all(val_grids, "Val")
        np.save(out_dir / "val_images.npy", val_images)

    shutil.copy(str(vocab_path), str(out_dir / "vocab.json"))

    zip_path = out_dir / "colab_dataset.zip"
    files_to_zip = ["train_tokens.npy", "val_tokens.npy", "vocab.json"]
    if args.render:
        files_to_zip.extend(["train_images.npy", "val_images.npy"])

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in files_to_zip:
            fpath = out_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)

    total_mb = 0
    for fname in files_to_zip:
        fpath = out_dir / fname
        if fpath.exists():
            total_mb += fpath.stat().st_size
    total_mb /= 1024 * 1024
    zip_mb = zip_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 50}")
    print(f"Dataset Summary")
    print(f"{'=' * 50}")
    print(f"  Training samples:   {len(train_grids):,}")
    print(f"  Validation samples: {len(val_grids):,}")
    print(f"  Grid shape:         {train_grids.shape[1:]} ({train_grids.shape[1]}x{train_grids.shape[2]})")
    if args.render:
        print(f"  Image shape:        {train_images.shape[1:]} ({train_images.shape[2]}x{train_images.shape[3]})")
    print(f"  Vocab size:         {vocab.size}")
    print(f"  Active charsets:    {vocab.charsets}")
    print(f"  Uncompressed size:  {total_mb:.1f} MB")
    print(f"  Zip size:           {zip_mb:.1f} MB")
    print(f"  Zip location:       {zip_path}")
    print()
    print("Upload to Colab:")
    print(f"  from google.colab import files")
    print(f"  files.upload()  # select the zip file")
    print(f"  !unzip -q colab_dataset.zip -d /content/data/")


if __name__ == "__main__":
    main()
