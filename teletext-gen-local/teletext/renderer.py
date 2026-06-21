from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    W, H, CW, CH, IMAGE_W, IMAGE_H, PALETTE,
    FIRST_MOSAIC_ID, LAST_MOSAIC_ID,
)
from teletext.vocab import Token, Vocabulary
from teletext.charsets import decode_char_with_charset


def _find_font(font_path: Optional[str] = None, size: int = 10) -> ImageFont.FreeTypeFont:
    if font_path is not None:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            pass
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _draw_mosaic(draw: ImageDraw, char_id: int, fg_rgb: Tuple[int, int, int],
                 bg_rgb: Tuple[int, int, int], x0: int, y0: int) -> None:
    mosaic_idx = char_id - FIRST_MOSAIC_ID
    seg_h1 = CH // 3
    seg_h2 = CH - 2 * seg_h1
    seg_w1 = CW // 2
    seg_w2 = CW - seg_w1
    segments = [
        (0, 0, seg_w1, seg_h1),
        (seg_w1, 0, seg_w2, seg_h1),
        (0, seg_h1, seg_w1, seg_h1),
        (seg_w1, seg_h1, seg_w2, seg_h1),
        (0, seg_h1 * 2, seg_w1, seg_h2),
        (seg_w1, seg_h1 * 2, seg_w2, seg_h2),
    ]
    for i, (sx, sy, sw, sh) in enumerate(segments):
        if (mosaic_idx >> i) & 1:
            draw.rectangle([x0 + sx, y0 + sy, x0 + sx + sw - 1, y0 + sy + sh - 1], fill=fg_rgb)
        else:
            draw.rectangle([x0 + sx, y0 + sy, x0 + sx + sw - 1, y0 + sy + sh - 1], fill=bg_rgb)


def _char_in_font(char: str, font: ImageFont.FreeTypeFont) -> bool:
    try:
        mask = font.getmask(char)
        return True
    except (OSError, UnicodeEncodeError):
        return False


def render_page(token_grid: np.ndarray, vocab: Vocabulary,
                font_path: Optional[str] = None,
                charset_key: Optional[str] = None) -> Image.Image:
    """Render a 25x40 token grid to a 520x400 RGB PIL Image.

    If charset_key is provided, use that charset for decoding characters;
    otherwise use the vocab's default charset.
    """
    if token_grid.shape != (H, W):
        raise ValueError(f"Expected grid shape ({H}, {W}), got {token_grid.shape}")

    img = Image.new("RGB", (IMAGE_W, IMAGE_H))
    draw = ImageDraw.Draw(img)
    font = _find_font(font_path, size=10)

    for row in range(H):
        for col in range(W):
            token_id = int(token_grid[row, col])
            token = vocab.id_to_token(token_id)
            if token is None:
                continue

            x0 = col * CW
            y0 = row * CH
            bg_rgb = PALETTE[token.bg]
            fg_rgb = PALETTE[token.fg]

            draw.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], fill=bg_rgb)

            if FIRST_MOSAIC_ID <= token.char_id <= LAST_MOSAIC_ID:
                _draw_mosaic(draw, token.char_id, fg_rgb, bg_rgb, x0, y0)
            else:
                if charset_key is not None:
                    char = decode_char_with_charset(token.char_id, charset_key)
                else:
                    char = vocab.decode_char(token.char_id)
                if not char:
                    continue
                if not _char_in_font(char, font):
                    char = '\uFFFD'
                bbox = draw.textbbox((0, 0), char, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x0 + (CW - tw) // 2
                ty = y0 + (CH - th) // 2 - bbox[1]
                draw.text((tx, ty), char, fill=fg_rgb, font=font)

    return img


def render_and_save(token_grid: np.ndarray, vocab: Vocabulary,
                    out_path: Path, font_path: Optional[str] = None,
                    charset_key: Optional[str] = None) -> None:
    """Render a grid and save to disk."""
    img = render_page(token_grid, vocab, font_path, charset_key)
    img.save(out_path)


def _render_worker(args: Tuple) -> None:
    grid_bytes, vocab_dict, charsets_list, charset_key, out_path_str, font_path = args
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(H, W)
    v = Vocabulary(charsets=charsets_list)
    for tid, (cid, fg, bg) in vocab_dict.items():
        t = Token(cid, fg, bg)
        v.mark_observed(t)
    render_and_save(grid, v, Path(out_path_str), font_path, charset_key)


def render_batch(grids: List[np.ndarray], vocab: Vocabulary,
                 out_dir: Path, font_path: Optional[str] = None,
                 num_workers: int = 4,
                 charset_key: Optional[str] = None) -> None:
    """Render multiple grids in parallel using multiprocessing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_dict: Dict[int, Tuple[int, int, int]] = {}
    for cid in range(LAST_MOSAIC_ID + 1):
        for fg in range(8):
            for bg in range(8):
                token = Token(cid, fg, bg)
                token_id = vocab.token_to_id(token)
                vocab_dict[token_id] = (cid, fg, bg)

    args = []
    for i, grid in enumerate(grids):
        args.append((
            grid.astype(np.int64).tobytes(),
            vocab_dict,
            vocab.charsets,
            charset_key,
            str(out_dir / f"page_{i:05d}.png"),
            font_path,
        ))

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        pool.map(_render_worker, args)
