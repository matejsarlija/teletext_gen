import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import IMAGE_W, IMAGE_H
from scripts import prepare_colab
from teletext.synthetic import generate_page
from teletext.vocab import Vocabulary


def _create_dummy_raw_page(raw_source_dir: Path, page_number: int,
                           subpage: int = 1, source: str = "hrt") -> None:
    json_path = raw_source_dir / f"{page_number}_{subpage:02d}.json"
    png_path = raw_source_dir / f"{page_number}_{subpage:02d}.png"
    meta = {
        "source": source,
        "page": page_number,
        "subpage": subpage,
        "charset": "croatian",
    }
    json_path.write_text(json.dumps(meta))
    img = Image.new("RGB", (IMAGE_W, IMAGE_H), (0, 0, 0))
    img.save(png_path)


def test_prepare_colab_packages_token_dataset(tmp_path: Path, monkeypatch) -> None:
    import config as cfg

    raw_dir = tmp_path / "raw"
    synthetic_dir = tmp_path / "synthetic"
    out_dir = tmp_path / "colab_upload"
    vocab_path = tmp_path / "vocab.json"

    hrt_dir = raw_dir / "hrt"
    hrt_dir.mkdir(parents=True)
    synthetic_dir.mkdir()

    vocab = Vocabulary.build_full()
    vocab.save(vocab_path)
    for page_num in [100, 200, 300]:
        _create_dummy_raw_page(hrt_dir, page_num, source="hrt")
    for idx in range(3):
        np.save(synthetic_dir / f"grid_{idx:06d}.npy", generate_page(vocab))

    monkeypatch.setattr(cfg, "RAW_DIR", raw_dir)
    monkeypatch.setattr(cfg, "SYNTHETIC_DIR", synthetic_dir)
    monkeypatch.setattr(cfg, "COLAB_UPLOAD_DIR", out_dir)
    monkeypatch.setattr(cfg, "VOCAB_PATH", vocab_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_colab.py", "--out", str(out_dir), "--vocab", str(vocab_path),
         "--no-render", "--no-synthetic", "--no-oversample"],
    )

    prepare_colab.main()

    assert (out_dir / "train_tokens.npy").exists()
    assert (out_dir / "val_tokens.npy").exists()
    assert (out_dir / "train_weights.npy").exists()
    assert (out_dir / "val_weights.npy").exists()
    assert (out_dir / "train_metadata.json").exists()
    assert (out_dir / "val_metadata.json").exists()
    assert (out_dir / "vocab.json").exists()
    assert (out_dir / "colab_dataset.zip").exists()
    assert not (out_dir / "train_images.npy").exists()


def test_weights_stay_aligned_after_split(tmp_path: Path, monkeypatch) -> None:
    import config as cfg

    RAW_DIR = tmp_path / "raw"
    SYNTHETIC_DIR = tmp_path / "synthetic"
    out_dir = tmp_path / "colab_upload"
    vocab_path = tmp_path / "vocab.json"

    RAW_DIR.mkdir(parents=True)
    SYNTHETIC_DIR.mkdir()

    vocab = Vocabulary.build_full()
    vocab.save(vocab_path)

    hrt_dir = RAW_DIR / "hrt"
    hrt_dir.mkdir()
    for page_num in [100, 110, 200]:
        _create_dummy_raw_page(hrt_dir, page_num, source="hrt")

    monkeypatch.setattr(cfg, "RAW_DIR", RAW_DIR)
    monkeypatch.setattr(cfg, "SYNTHETIC_DIR", SYNTHETIC_DIR)
    monkeypatch.setattr(cfg, "COLAB_UPLOAD_DIR", out_dir)
    monkeypatch.setattr(cfg, "VOCAB_PATH", vocab_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_colab.py", "--out", str(out_dir), "--vocab", str(vocab_path),
         "--no-render", "--no-synthetic"],
    )

    prepare_colab.main()

    train_meta = json.loads((out_dir / "train_metadata.json").read_text())
    train_weights = np.load(out_dir / "train_weights.npy")
    val_meta = json.loads((out_dir / "val_metadata.json").read_text())
    val_weights = np.load(out_dir / "val_weights.npy")

    all_meta = train_meta + val_meta
    all_weights = np.concatenate([train_weights, val_weights])

    assert len(all_meta) == len(all_weights)
    for i, meta in enumerate(all_meta):
        if meta["page_range"] == "news":
            expected = 2.0
        else:
            expected = 1.0
        assert all_weights[i] == expected, (
            f"Row {i}: page_range={meta['page_range']}, "
            f"expected weight={expected}, got {all_weights[i]}"
        )


def test_synthetic_page_not_replaced_by_placeholder(tmp_path: Path, monkeypatch) -> None:
    import config as cfg

    RAW_DIR = tmp_path / "raw"
    SYNTHETIC_DIR = tmp_path / "synthetic"
    out_dir = tmp_path / "colab_upload"
    vocab_path = tmp_path / "vocab.json"

    RAW_DIR.mkdir(parents=True)
    SYNTHETIC_DIR.mkdir()

    vocab = Vocabulary.build_full()
    vocab.save(vocab_path)

    known_grid = np.full((25, 40), 3, dtype=np.int64)
    np.save(SYNTHETIC_DIR / "grid_000000.npy", known_grid)

    hrt_dir = RAW_DIR / "hrt"
    hrt_dir.mkdir()
    _create_dummy_raw_page(hrt_dir, 100, source="hrt")

    monkeypatch.setattr(cfg, "RAW_DIR", RAW_DIR)
    monkeypatch.setattr(cfg, "SYNTHETIC_DIR", SYNTHETIC_DIR)
    monkeypatch.setattr(cfg, "COLAB_UPLOAD_DIR", out_dir)
    monkeypatch.setattr(cfg, "VOCAB_PATH", vocab_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_colab.py", "--out", str(out_dir), "--vocab", str(vocab_path),
         "--no-render", "--no-oversample"],
    )

    prepare_colab.main()

    train_tokens = np.load(out_dir / "train_tokens.npy")
    val_tokens = np.load(out_dir / "val_tokens.npy")

    all_tokens = np.concatenate([train_tokens, val_tokens])
    assert any(np.array_equal(row, known_grid) for row in all_tokens), (
        "Synthetic grid not found in output — placeholders may have been used"
    )
