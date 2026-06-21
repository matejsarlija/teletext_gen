from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import H, W, FIRST_REAL_TOKEN_ID
from teletext.vocab import Vocabulary


def find_grid_files(paths: List[Path]) -> List[Path]:
    """Collect all .npy grid files from given directories, recursing into subdirs."""
    files: List[Path] = []
    for d in paths:
        if d.exists():
            files.extend(sorted(d.rglob("grid_*.npy")))
    return files


def load_grids(paths: List[Path]) -> np.ndarray:
    """Load multiple .npy grids into a single (N, H, W) array."""
    grids: List[np.ndarray] = []
    for p in paths:
        grids.append(np.load(p))
    if not grids:
        return np.empty((0, H, W), dtype=np.int64)
    return np.stack(grids)


def token_counts(grids: np.ndarray, vocab: Optional[Vocabulary] = None) -> np.ndarray:
    """Count token frequency across all grids."""
    flat = grids.flatten()
    max_id = int(flat.max())
    counts = np.bincount(flat.astype(np.int64), minlength=max_id + 1)
    return counts


def is_real_token(token_id: int) -> bool:
    return token_id >= FIRST_REAL_TOKEN_ID
