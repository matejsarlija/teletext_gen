import json
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np

from config import (
    PAD_ID, START_ID, END_ID, FIRST_REAL_TOKEN_ID,
    NUM_COLORS, LAST_MOSAIC_ID, SPACE_CHAR_ID,
    FIRST_PRINTABLE_CHAR_ID, LAST_PRINTABLE_CHAR_ID,
    DEFAULT_NATIONAL_CHARSET,
)
from teletext.charsets import (
    NATIONAL_CHARSETS,
    REVERSE_CHARSETS,
    decode_char_with_charset,
)

Token = namedtuple('Token', ['char_id', 'fg', 'bg'])


class Vocabulary:
    """Maps (char_id, fg, bg) tokens to/from integer IDs using deterministic encoding.
    Token ID = char_id * NUM_COLORS^2 + fg * NUM_COLORS + bg + FIRST_REAL_TOKEN_ID.
    Special IDs: PAD=0, START=1, END=2. All 8192 real token IDs are always valid.

    Supports multiple national character sets for display.
    """

    def __init__(self, charsets: Optional[Union[str, List[str]]] = None) -> None:
        self._observed: Set[Token] = set()
        if charsets is None:
            charsets = [DEFAULT_NATIONAL_CHARSET]
        elif isinstance(charsets, str):
            charsets = [charsets]
        self._charsets: List[str] = charsets

    @property
    def size(self) -> int:
        return FIRST_REAL_TOKEN_ID + (LAST_MOSAIC_ID + 1) * NUM_COLORS * NUM_COLORS

    @property
    def national_charset(self) -> str:
        return self._charsets[0] if self._charsets else DEFAULT_NATIONAL_CHARSET

    @property
    def charsets(self) -> List[str]:
        return list(self._charsets)

    def token_to_id(self, token: Token) -> int:
        return (
            token.char_id * NUM_COLORS * NUM_COLORS
            + token.fg * NUM_COLORS
            + token.bg
            + FIRST_REAL_TOKEN_ID
        )

    def id_to_token(self, token_id: int) -> Optional[Token]:
        if token_id < FIRST_REAL_TOKEN_ID:
            return None
        tid = token_id - FIRST_REAL_TOKEN_ID
        char_id = tid // (NUM_COLORS * NUM_COLORS)
        remainder = tid % (NUM_COLORS * NUM_COLORS)
        fg = remainder // NUM_COLORS
        bg = remainder % NUM_COLORS
        if char_id > LAST_MOSAIC_ID or fg >= NUM_COLORS or bg >= NUM_COLORS:
            return None
        return Token(char_id=int(char_id), fg=int(fg), bg=int(bg))

    def decode_char(self, char_id: int) -> str:
        """Convert char_id to display char using the default charset."""
        return decode_char_with_charset(char_id, self._charsets[0])

    def decode_char_with(self, char_id: int, charset_key: str) -> str:
        """Convert char_id to display char using a specific charset."""
        return decode_char_with_charset(char_id, charset_key)

    def char_to_id(self, char: str) -> Optional[int]:
        """Convert display char to char_id using the default charset."""
        cs = self._charsets[0]
        subs = NATIONAL_CHARSETS.get(cs, {})
        for rev_byte, rev_char in subs.items():
            if rev_char == char:
                return rev_byte - 32
        byte_pos = ord(char)
        if 32 <= byte_pos <= 126:
            return byte_pos - 32
        return None

    def mark_observed(self, token: Token, charset: Optional[str] = None) -> None:
        self._observed.add(token)

    @property
    def observed_count(self) -> int:
        return len(self._observed)

    def observed_ids(self) -> List[int]:
        return sorted(self.token_to_id(t) for t in self._observed)

    def save(self, path: Path) -> None:
        data = {
            'charsets': self._charsets,
            'national_charset': self._charsets[0],
            'tokens': [],
        }
        for token in sorted(self._observed):
            data['tokens'].append({
                'char_id': token.char_id,
                'fg': token.fg,
                'bg': token.bg,
            })
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> 'Vocabulary':
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            charsets = [DEFAULT_NATIONAL_CHARSET]
            tokens_data = raw
        elif isinstance(raw, dict) and 'tokens' not in raw:
            charsets = [raw.get('national_charset', DEFAULT_NATIONAL_CHARSET)]
            tokens_data = [raw]
        else:
            charsets = raw.get('charsets', [raw.get('national_charset', DEFAULT_NATIONAL_CHARSET)])
            tokens_data = raw.get('tokens', [])
        vocab = cls(charsets=charsets)
        for entry in tokens_data:
            token = Token(char_id=entry['char_id'], fg=entry['fg'], bg=entry['bg'])
            vocab.mark_observed(token)
        return vocab

    def build_from_grids(self, paths: List[Path]) -> None:
        """Scan .npy grid files and mark all observed tokens."""
        for path in paths:
            grid = np.load(path)
            for token_id in grid.flatten():
                tid = int(token_id)
                token = self.id_to_token(tid)
                if token is not None:
                    self.mark_observed(token)

    @classmethod
    def build_full(cls, charsets: Optional[Union[str, List[str]]] = None) -> 'Vocabulary':
        """Create a vocabulary with all possible (char_id, fg, bg) combos."""
        vocab = cls(charsets=charsets)
        for char_id in range(LAST_MOSAIC_ID + 1):
            for fg in range(NUM_COLORS):
                for bg in range(NUM_COLORS):
                    vocab.mark_observed(Token(char_id, fg, bg))
        return vocab
