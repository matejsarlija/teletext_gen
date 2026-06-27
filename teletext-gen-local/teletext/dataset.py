import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

START_ID = 1
END_ID = 2


class BalancedTokenGridDataset(Dataset):
    def __init__(self, tokens_path: str, weights_path: str, metadata_path: str):
        super().__init__()
        self.tokens = np.load(tokens_path)
        self.weights = np.load(weights_path).astype(np.float32)
        self.metadata: List[dict] = json.loads(Path(metadata_path).read_text())

        if len(self.tokens) != len(self.weights) or len(self.tokens) != len(self.metadata):
            raise ValueError(
                f"Mismatched lengths: tokens={len(self.tokens)}, "
                f"weights={len(self.weights)}, metadata={len(self.metadata)}"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        grid = self.tokens[idx]
        seq = grid.flatten()
        tokens = np.concatenate([[START_ID], seq, [END_ID]])
        return torch.tensor(tokens, dtype=torch.long)

    def get_weights(self) -> torch.Tensor:
        return torch.tensor(self.weights, dtype=torch.float32)
