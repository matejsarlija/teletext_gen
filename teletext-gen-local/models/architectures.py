import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeneratorConfig:
    vocab_size: int = 8195
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_rows: int = 25
    max_cols: int = 40


class PositionalEncoding2D(nn.Module):
    """2D positional encoding: row embedding + column embedding concatenated."""

    def __init__(self, d_model: int, max_rows: int = 25, max_cols: int = 40):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding"
        half = d_model // 2
        self.row_emb = nn.Embedding(max_rows, half)
        self.col_emb = nn.Embedding(max_cols, half)

    def forward(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        row_emb = self.row_emb(row)
        col_emb = self.col_emb(col)
        return torch.cat([row_emb, col_emb], dim=-1)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block with causal self-attention."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class TeletextGenerator(nn.Module):
    """Decoder-only transformer with 2D positional encoding for teletext page generation."""

    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding2D(
            config.d_model, config.max_rows, config.max_cols
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.nhead, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, row_ids: torch.Tensor,
                col_ids: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_encoding(row_ids, col_ids)
        x = self.dropout(token_emb + pos_emb)

        causal_mask = torch.triu(
            torch.full((S, S), float('-inf'), device=x.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.norm(x)
        return self.output(x)

    @torch.no_grad()
    def generate(self, row_ids: torch.Tensor, col_ids: torch.Tensor,
                 max_len: int, temperature: float = 1.0, top_k: int = 0,
                 start_token_id: int = 1, end_token_id: int = 2,
                 pad_token_id: int = 0) -> torch.Tensor:
        """Autoregressive sampling with temperature and top-k filtering."""
        device = next(self.parameters()).device
        batch_size = row_ids.shape[0]
        tokens = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
        tokens[:, 0] = start_token_id

        for pos in range(1, max_len):
            logits = self(tokens[:, :pos], row_ids[:, :pos], col_ids[:, :pos])
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_values, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits[next_logits < top_values[:, -1:]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            tokens[:, pos] = next_token

            if end_token_id is not None and (next_token == end_token_id).any():
                break

        return tokens


class CellCNN(nn.Module):
    """Tiny CNN for single 13x16 teletext cell classification.
    Takes (B, 3, 16, 13) and produces 3 output heads.
    """

    def __init__(self, num_char_ids: int = 128, num_colors: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 128)
        self.char_head = nn.Linear(128, num_char_ids)
        self.fg_head = nn.Linear(128, num_colors)
        self.bg_head = nn.Linear(128, num_colors)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x).flatten(1)
        x = self.dropout(F.relu(self.fc(x)))
        return self.char_head(x), self.fg_head(x), self.bg_head(x)


class TeletextExtractor(nn.Module):
    """Applies CellCNN across all 1000 cells in a teletext page.
    Input: (B, 3, 400, 520). Output: per-cell logits for char_id, fg, bg.
    """

    def __init__(self, num_char_ids: int = 128, num_colors: int = 8):
        super().__init__()
        self.cell_cnn = CellCNN(num_char_ids, num_colors)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        cells = x.unfold(2, 16, 16).unfold(3, 13, 13)
        cells = cells.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, 16, 13)
        char_logits, fg_logits, bg_logits = self.cell_cnn(cells)
        char_logits = char_logits.reshape(B, 25, 40, -1)
        fg_logits = fg_logits.reshape(B, 25, 40, -1)
        bg_logits = bg_logits.reshape(B, 25, 40, -1)
        return char_logits, fg_logits, bg_logits
