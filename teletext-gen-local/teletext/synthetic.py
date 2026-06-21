import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import (
    W, H, NUM_COLORS,
    FIRST_MOSAIC_ID, LAST_MOSAIC_ID,
    SPACE_CHAR_ID,
)
from teletext.vocab import Token, Vocabulary

ASCII_WORDS = [
    "HELLO", "WORLD", "NEWS", "SPORT", "VRIJEME", "HRT", "TEKST",
    "DANAS", "SUTRA", "ZAGREB", "SPLIT", "RIJEKA", "OSIJEK",
    "INDEX", "NASLOV", "STRANICA", "IZBOR", "POVRATAK",
    "INFO", "KULTURA", "GLAZBA", "FILM", "TEATAR", "KNJIGA",
    "CROATIA", "EUROPA", "SVIJET", "POLITIKA", "GOSPODARSTVO",
    "ZNANOST", "TEHNOLOGIJA", "ZDRAVLJE", "KOSARKA",
    "NOGOMET", "TENIS", "VATERPOLO", "RUKOMET", "ODBOJKA",
    "SLOVENIJA", "BIH", "SRBIJA", "MADARSKA", "ITALIJA",
    "AUTOMOBILIZAM", "TURIZAM", "POZORISTE",
    "SERVIS", "MORE", "PLANINE", "NACIONALNI",
]

CROATIAN_WORDS = [
    "VRIJEME", "GLAZBA", "POZORIŠTE", "NACIONALNI",
    "GOSPODARSTVO", "ZNANOST", "TEHNOLOGIJA",
    "SJEĆANJE", "NACRT", "ZVJEŠĆE", "DJECA",
    "ŠIRINA", "ŽIVOT", "ZVJEZDA", "NJEŽNO",
    "PJESMA", "TRGOVINA", "PRIČE", "ZVONO",
    "JEDRO", "CVIJET", "LJUDI", "SUSJED",
    "MLIJEKO", "SMRČE", "VEČE", "NOSAČ",
    "VJEROVATI", "POZDRAV", "TISUĆA",
    "ČESTITATI", "IZLIJEČITI", "NAČELO",
    "PREDJEL", "SUDJELOVATI",
    "CVJETNICA", "LJETOVANJE",
    "RADNJE", "SJEDIŠTE", "CIJENA",
    "DJEVOJKA", "LJUDI",
]

CROATIAN_WORDS_LOWER = [w.lower() for w in CROATIAN_WORDS]
ALL_WORDS = ASCII_WORDS + CROATIAN_WORDS + CROATIAN_WORDS_LOWER
_WORD_WEIGHTS = [0.6] * len(ASCII_WORDS) + [1.0] * len(CROATIAN_WORDS) + [0.4] * len(CROATIAN_WORDS_LOWER)


def _random_word() -> str:
    return random.choices(ALL_WORDS, weights=_WORD_WEIGHTS, k=1)[0]


def _random_text_line(max_len: int = 40) -> str:
    words: List[str] = []
    total = 0
    while total < max_len:
        word = _random_word()
        if words and total + 1 + len(word) > max_len:
            word = word[:max_len - total - 1]
        if words:
            words.append(" ")
        words.append(word)
        total = sum(len(w) for w in words)
        if total >= max_len:
            break
    return "".join(words)[:max_len]


def _char_to_id(vocab: Vocabulary, ch: str) -> Optional[int]:
    cid = vocab.char_to_id(ch)
    if cid is not None:
        return cid
    byte_pos = ord(ch)
    if 32 <= byte_pos <= 126:
        return byte_pos - 32
    return None


def _new_token(vocab: Vocabulary, char_id: int, fg: int, bg: int) -> int:
    token = Token(char_id=char_id, fg=fg, bg=bg)
    return vocab.token_to_id(token)


def _encode_text(vocab: Vocabulary, text: str, fg: int, bg: int) -> np.ndarray:
    row = np.zeros(W, dtype=np.int64)
    for i, ch in enumerate(text):
        if i >= W:
            break
        cid = _char_to_id(vocab, ch)
        if cid is not None:
            row[i] = _new_token(vocab, cid, fg, bg)
    return row


def generate_page(vocab: Vocabulary) -> np.ndarray:
    """Generate a single structured synthetic teletext page (25x40 token IDs)."""
    grid = np.zeros((H, W), dtype=np.int64)

    header_bg = random.randint(1, NUM_COLORS - 1)
    page_num = str(random.randint(100, 899))
    offset = (W - len(page_num)) // 2
    for i, ch in enumerate(page_num):
        cid = _char_to_id(vocab, ch)
        if cid is not None:
            grid[0, offset + i] = _new_token(vocab, cid, 7, header_bg)
    for i in range(W):
        if grid[0, i] == 0:
            grid[0, i] = _new_token(vocab, SPACE_CHAR_ID, 7, header_bg)

    footer_bg = random.randint(1, NUM_COLORS - 1)
    footer_text = _random_text_line(36)
    footer_fg = random.randint(1, NUM_COLORS - 1)
    footer_row = _encode_text(vocab, footer_text, footer_fg, footer_bg)
    grid[24, :] = footer_row
    for i in range(len(footer_text), W):
        if grid[24, i] == 0:
            grid[24, i] = _new_token(vocab, SPACE_CHAR_ID, footer_fg, footer_bg)

    for row in range(1, H - 1):
        mode = random.choices(
            ["text", "graphics", "empty"],
            weights=[0.7, 0.2, 0.1],
        )[0]

        if mode == "empty":
            grid[row, :] = _new_token(vocab, SPACE_CHAR_ID, 7, 0)

        elif mode == "graphics":
            bg = random.randint(0, NUM_COLORS - 1)
            for col in range(W):
                char_id = random.randint(FIRST_MOSAIC_ID, LAST_MOSAIC_ID)
                fg = random.randint(0, NUM_COLORS - 1)
                grid[row, col] = _new_token(vocab, char_id, fg, bg)

        else:
            line = _random_text_line(W)
            fg = random.randint(1, NUM_COLORS - 1)
            bg = random.randint(0, NUM_COLORS - 1)
            text_row = _encode_text(vocab, line, fg, bg)
            grid[row, :len(line)] = text_row[:len(line)]
            for col in range(len(line), W):
                if grid[row, col] == 0:
                    grid[row, col] = _new_token(vocab, SPACE_CHAR_ID, fg, bg)

    return grid


def generate_dataset(n: int, vocab: Vocabulary, out_dir: Path, render: bool = True) -> None:
    """Generate N synthetic pages and save token grids, optionally with PNG renders."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_and_save = None
    if render:
        from teletext.renderer import render_and_save

    for i in range(n):
        grid = generate_page(vocab)
        np.save(out_dir / f"grid_{i:06d}.npy", grid)
        if render_and_save is not None:
            render_and_save(grid, vocab, out_dir / f"page_{i:06d}.png")
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{n} pages")
