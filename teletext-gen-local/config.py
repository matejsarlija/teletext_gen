from pathlib import Path
from typing import Dict, List, Tuple

W: int = 40
H: int = 25
CW: int = 13
CH: int = 16
IMAGE_W: int = W * CW
IMAGE_H: int = H * CH

PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (255, 255, 0),
    4: (0, 0, 255),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (255, 255, 255),
}
NUM_COLORS: int = len(PALETTE)

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TOKENS_DIR = DATA_DIR / "tokens"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
COLAB_UPLOAD_DIR = DATA_DIR / "colab_upload"
VOCAB_PATH = DATA_DIR / "vocab.json"

DEFAULT_DELAY: float = 1.5
MAX_RETRIES: int = 3

PAD_ID: int = 0
START_ID: int = 1
END_ID: int = 2
FIRST_REAL_TOKEN_ID: int = 3

SPACE_CHAR_ID: int = 0
FIRST_PRINTABLE_CHAR_ID: int = 1
LAST_PRINTABLE_CHAR_ID: int = 95
FIRST_MOSAIC_ID: int = 96
LAST_MOSAIC_ID: int = 127

DEFAULT_NATIONAL_CHARSET: str = 'croatian'
DEFAULT_SOURCES: List[str] = ['hrt', 'rtvslo', 'rtvfbih']
ACTIVE_CHARSETS: List[str] = ['croatian', 'slovenian', 'bosnian']
DEFAULT_SCRAPE_DELAY: float = 1.5

PAGE_RANGES: Dict[str, Tuple[int, int]] = {
    'index':         (100, 199),
    'news':          (200, 299),
    'sports':        (300, 399),
    'info':          (400, 499),
    'entertainment': (500, 599),
    'subtitles':     (600, 699),
    'services':      (700, 799),
    'promotions':    (800, 899),
}

SYNTHETIC_AUGMENTATION_THRESHOLD: int = 200
TARGET_PAGES_PER_RANGE: int = 2000
MAX_OVERSAMPLE_FACTOR: int = 10


def page_to_range(page_number: int) -> str:
    for name, (lo, hi) in PAGE_RANGES.items():
        if lo <= page_number <= hi:
            return name
    return 'unknown'
