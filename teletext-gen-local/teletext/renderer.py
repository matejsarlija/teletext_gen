import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    W, H, CW, CH, IMAGE_W, IMAGE_H, PALETTE, NUM_COLORS,
    FIRST_MOSAIC_ID, LAST_MOSAIC_ID, SPACE_CHAR_ID,
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


_COLOR_TOLERANCE = 40


def _color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
    return abs(int(c1[0]) - int(c2[0])) + abs(int(c1[1]) - int(c2[1])) + abs(int(c1[2]) - int(c2[2]))


def _closest_palette_index(rgb: Tuple[int, int, int]) -> int:
    best_idx = 0
    best_dist = 999999
    for idx, prgb in PALETTE.items():
        d = _color_distance(rgb, prgb)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def _mosaic_masks() -> List[np.ndarray]:
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
    masks = []
    for mosaic_idx in range(32):
        mask = np.zeros((CH, CW), dtype=np.bool_)
        for i, (sx, sy, sw, sh) in enumerate(segments):
            if (mosaic_idx >> i) & 1:
                mask[sy:sy + sh, sx:sx + sw] = True
        masks.append(mask)
    return masks


_MOSAIC_MASKS = None


def _detect_mosaic(ink_mask: np.ndarray) -> Optional[int]:
    global _MOSAIC_MASKS
    if _MOSAIC_MASKS is None:
        _MOSAIC_MASKS = _mosaic_masks()
    for mosaic_idx, template in enumerate(_MOSAIC_MASKS):
        match = np.all(ink_mask == template)
        if match:
            return mosaic_idx + FIRST_MOSAIC_ID
    return None


@functools.lru_cache(maxsize=1)
def _char_templates(font_size: int = 10) -> Dict[int, np.ndarray]:
    font = _find_font(size=font_size)
    templates: Dict[int, np.ndarray] = {}
    for char_id in range(LAST_MOSAIC_ID + 1):
        if char_id == 0:
            templates[0] = np.zeros((CH, CW), dtype=np.bool_)
            continue
        img = Image.new("RGB", (CW, CH), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        char = chr(char_id + 32)
        if not _char_in_font(char, font):
            templates[char_id] = np.zeros((CH, CW), dtype=np.bool_)
            continue
        bbox = draw.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (CW - tw) // 2
        ty = (CH - th) // 2 - bbox[1]
        draw.text((tx, ty), char, fill=(255, 255, 255), font=font)
        arr = np.array(img)
        mask = (arr[:, :, 0] > 127) | (arr[:, :, 1] > 127) | (arr[:, :, 2] > 127)
        templates[char_id] = mask
    return templates


def _char_templates_batch():
    templates = _char_templates()
    ids = np.array(sorted(templates.keys()), dtype=np.int32)
    masks = np.stack([templates[cid] for cid in ids])
    ink_counts = masks.reshape(len(ids), -1).sum(axis=1).astype(np.float32)
    return masks.reshape(len(ids), -1).astype(np.float32), ids, ink_counts


def _cell_fg_rgb(cell: np.ndarray, ink_mask: np.ndarray, bg_rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    ink_pixels = cell[ink_mask]
    if len(ink_pixels) == 0:
        return (255, 255, 255)
    avg = tuple(int(np.mean(ink_pixels[:, i])) for i in range(3))
    return min(set(tuple(p) for p in ink_pixels), key=lambda p: _color_distance(p, avg))


def image_to_grid(image: Image.Image, vocab: Vocabulary) -> np.ndarray:
    if image.size != (IMAGE_W, IMAGE_H):
        image = image.resize((IMAGE_W, IMAGE_H), Image.NEAREST)

    arr = np.array(image, dtype=np.uint8)
    grid = np.zeros((H, W), dtype=np.int64)

    mosaic_masks = _mosaic_masks()
    t_masks, t_ids, t_ink_counts = _char_templates_batch()

    batch_rows = []
    batch_cols = []
    batch_ink_masks = []
    batch_fg = []
    batch_bg = []

    for row in range(H):
        y0 = row * CH
        for col in range(W):
            x0 = col * CW
            cell = arr[y0:y0 + CH, x0:x0 + CW]

            corners = [cell[0, 0], cell[0, -1], cell[-1, 0], cell[-1, -1]]
            avg = tuple(int(np.mean([c[i] for c in corners])) for i in range(3))
            bg_rgb = min(set(tuple(c) for c in corners), key=lambda c: _color_distance(c, avg))
            bg = _closest_palette_index(bg_rgb)

            diff = np.abs(cell.astype(np.int32) - np.array(bg_rgb, dtype=np.int32))
            ink_mask = np.any(diff > _COLOR_TOLERANCE, axis=2)
            ink_count = np.sum(ink_mask)
            total = CH * CW

            if ink_count < 3:
                grid[row, col] = vocab.token_to_id(Token(0, bg, bg))
                continue

            for mosaic_idx, tmpl in enumerate(mosaic_masks):
                if ink_count == np.sum(tmpl) and np.array_equal(ink_mask, tmpl):
                    fg_rgb = _cell_fg_rgb(cell, ink_mask, bg_rgb)
                    fg = _closest_palette_index(fg_rgb)
                    grid[row, col] = vocab.token_to_id(Token(mosaic_idx + FIRST_MOSAIC_ID, fg, bg))
                    break
            else:
                fg_rgb = _cell_fg_rgb(cell, ink_mask, bg_rgb)
                fg = _closest_palette_index(fg_rgb)
                batch_rows.append(row)
                batch_cols.append(col)
                batch_ink_masks.append(ink_mask)
                batch_fg.append(fg)
                batch_bg.append(bg)

    if batch_ink_masks:
        masks = np.stack(batch_ink_masks).astype(np.float32)
        N = len(masks)
        m_flat = masks.reshape(N, -1)
        m_ink_counts = m_flat.sum(axis=1)

        intersection = m_flat @ t_masks.T
        union = m_ink_counts[:, None] + t_ink_counts[None, :] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        best_idx = np.argmax(iou, axis=1)
        best_scores = iou[np.arange(N), best_idx]
        best_ids = t_ids[best_idx]
        best_ids[best_scores < 0.3] = 0

        for i in range(N):
            row = batch_rows[i]
            col = batch_cols[i]
            char_id = int(best_ids[i])
            fg = int(batch_fg[i])
            bg = int(batch_bg[i])
            grid[row, col] = vocab.token_to_id(Token(char_id, fg, bg))

    return grid
