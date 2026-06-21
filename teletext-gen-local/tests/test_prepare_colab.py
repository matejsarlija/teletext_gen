import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import prepare_colab
from teletext.synthetic import generate_page
from teletext.vocab import Vocabulary


def test_prepare_colab_packages_token_dataset(tmp_path: Path, monkeypatch) -> None:
    synthetic_dir = tmp_path / "synthetic"
    tokens_dir = tmp_path / "tokens"
    out_dir = tmp_path / "colab_upload"
    vocab_path = tmp_path / "vocab.json"
    synthetic_dir.mkdir()
    tokens_dir.mkdir()

    vocab = Vocabulary.build_full()
    vocab.save(vocab_path)
    for idx in range(3):
        np.save(synthetic_dir / f"grid_{idx:06d}.npy", generate_page(vocab))

    monkeypatch.setattr(prepare_colab, "SYNTHETIC_DIR", synthetic_dir)
    monkeypatch.setattr(prepare_colab, "TOKENS_DIR", tokens_dir)
    monkeypatch.setattr(prepare_colab, "COLAB_UPLOAD_DIR", out_dir)
    monkeypatch.setattr(prepare_colab, "VOCAB_PATH", vocab_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_colab.py", "--out", str(out_dir), "--vocab", str(vocab_path), "--no-render"],
    )

    prepare_colab.main()

    assert (out_dir / "train_tokens.npy").exists()
    assert (out_dir / "val_tokens.npy").exists()
    assert (out_dir / "vocab.json").exists()
    assert (out_dir / "colab_dataset.zip").exists()
    assert not (out_dir / "train_images.npy").exists()
