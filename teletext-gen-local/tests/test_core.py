import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import H, IMAGE_H, IMAGE_W, W
from models.architectures import GeneratorConfig, TeletextGenerator
from teletext.renderer import render_page
from teletext.synthetic import generate_dataset, generate_page
from teletext.vocab import Token, Vocabulary


def test_vocab_roundtrip() -> None:
    vocab = Vocabulary.build_full()
    token = Token(char_id=1, fg=7, bg=0)

    token_id = vocab.token_to_id(token)

    assert vocab.id_to_token(token_id) == token


def test_synthetic_page_is_valid_grid() -> None:
    vocab = Vocabulary.build_full()

    grid = generate_page(vocab)

    assert grid.shape == (H, W)
    assert np.issubdtype(grid.dtype, np.integer)
    assert grid.min() >= 0
    assert grid.max() < vocab.size


def test_renderer_outputs_expected_size() -> None:
    vocab = Vocabulary.build_full()
    grid = generate_page(vocab)

    image = render_page(grid, vocab)

    assert image.size == (IMAGE_W, IMAGE_H)


def test_generator_forward_shape_on_cpu() -> None:
    vocab = Vocabulary.build_full()
    config = GeneratorConfig(
        vocab_size=vocab.size,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.0,
    )
    model = TeletextGenerator(config)
    model.eval()

    seq_len = 8
    tokens = torch.ones((1, seq_len), dtype=torch.long)
    row_ids = torch.arange(H).unsqueeze(1).expand(H, W).reshape(1, -1)[:, :seq_len]
    col_ids = torch.arange(W).unsqueeze(0).expand(H, W).reshape(1, -1)[:, :seq_len]

    with torch.no_grad():
        logits = model(tokens, row_ids, col_ids)

    assert logits.shape == (1, seq_len, vocab.size)


def test_generate_dataset_can_skip_rendering(tmp_path: Path) -> None:
    vocab = Vocabulary.build_full()

    generate_dataset(2, vocab, tmp_path, render=False)

    assert sorted(path.name for path in tmp_path.glob("grid_*.npy")) == [
        "grid_000000.npy",
        "grid_000001.npy",
    ]
    assert list(tmp_path.glob("page_*.png")) == []

