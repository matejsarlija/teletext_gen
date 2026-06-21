import base64
import json
import time
import random
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

from config import MAX_RETRIES
from teletext.sources import TeletextSource, SOURCES


def _scrape_base64_gif(source: TeletextSource, page: int, subpage: int,
                       url: str) -> Optional[Image.Image]:
    """Extract base64-encoded GIF from HTML page."""
    resp = requests.get(url, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    img_tag = soup.find('img')
    if img_tag is None:
        return None
    src = img_tag.get('src', '')
    if not src.startswith('data:image'):
        return None
    b64_data = src.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_data)
    return Image.open(BytesIO(img_bytes)).convert('RGB')


def _scrape_direct_url(source: TeletextSource, page: int, subpage: int,
                       url: str) -> Optional[Image.Image]:
    """Download image directly from URL."""
    resp = requests.get(url, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert('RGB')


def _scrape_json_api(source: TeletextSource, page: int, subpage: int,
                     url: str) -> Optional[Image.Image]:
    """Fetch JSON API response, extract image URL, download."""
    resp = requests.get(url, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError:
        return None

    raw_path = Path(f"/tmp/teletext_inspect_{source.name}.json")
    raw_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    img_url = None
    if isinstance(data, dict):
        for key in ('image', 'img', 'imageUrl', 'image_url', 'url', 'src', 'data'):
            val = data.get(key)
            if isinstance(val, str) and (val.startswith('http') or val.startswith('data:')):
                img_url = val
                break
        if img_url is None:
            for val in data.values():
                if isinstance(val, str) and (val.startswith('http') or val.startswith('data:')):
                    img_url = val
                    break

    if img_url is None:
        return None

    if img_url.startswith('data:image'):
        b64_data = img_url.split(',', 1)[1]
        img_bytes = base64.b64decode(b64_data)
        return Image.open(BytesIO(img_bytes)).convert('RGB')
    else:
        img_resp = requests.get(img_url, timeout=15)
        img_resp.raise_for_status()
        return Image.open(BytesIO(img_resp.content)).convert('RGB')


def _build_url(source: TeletextSource, page: int, subpage: int) -> str:
    """Build the URL for a given page+subpage using the source's base_url template."""
    return source.base_url.format(page=page, subpage=subpage)


def scrape_source(source: TeletextSource, out_dir: Path, delay: float = 1.5,
                  page_range_override: Optional[tuple] = None) -> None:
    """Scrape a single teletext source.

    Downloads images and saves as PNG + JSON metadata sidecar.
    Supports resume: already-downloaded pages are skipped.
    Uses the source's image_format to select the right extraction strategy.
    """
    out_dir = Path(out_dir)
    source_dir = out_dir / source.name
    source_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for p in source_dir.glob("*.png"):
        seen.add(p.stem)

    pr = page_range_override or source.page_range
    sr = source.subpage_range
    total = (pr[1] - pr[0] + 1) * (sr[1] - sr[0] + 1)

    strategy_map = {
        'base64_gif': _scrape_base64_gif,
        'direct_url': _scrape_direct_url,
        'json_api': _scrape_json_api,
    }
    strategy = strategy_map.get(source.image_format)
    if strategy is None:
        print(f"  Unknown image_format '{source.image_format}' for {source.name}, skipping")
        return

    pbar = tqdm(total=total, desc=f"{source.name}", unit="page")
    for page in range(pr[0], pr[1] + 1):
        for subpage in range(sr[0], sr[1] + 1):
            stem = f"{page}_{subpage:02d}"
            if stem in seen:
                pbar.update(1)
                continue

            url = _build_url(source, page, subpage)
            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    img = strategy(source, page, subpage, url)
                    if img is not None:
                        img.save(source_dir / f"{stem}.png")
                        meta = {
                            'source': source.name,
                            'language': source.language,
                            'charset': source.charset,
                            'page': page,
                            'subpage': subpage,
                            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                            'url': url,
                        }
                        (source_dir / f"{stem}.json").write_text(
                            json.dumps(meta, indent=2)
                        )
                        success = True
                    break
                except (requests.RequestException, OSError, ValueError):
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(delay * 2)
                    else:
                        pass

            jitter = delay + random.uniform(-0.3, 0.3)
            time.sleep(max(0.1, jitter))
            pbar.update(1)
    pbar.close()


def scrape_all_sources(source_keys: List[str], out_dir: Path, delay: float = 1.5,
                       page_range_override: Optional[tuple] = None) -> None:
    """Scrape multiple teletext sources sequentially."""
    for key in source_keys:
        source = SOURCES.get(key)
        if source is None:
            print(f"Unknown source '{key}', skipping")
            continue
        print(f"\nScraping {source.display_name}...")
        scrape_source(source, out_dir, delay, page_range_override)


def inspect_source(source_key: str) -> None:
    """Fetch one page from a source and print raw HTML/JSON for inspection."""
    source = SOURCES.get(source_key)
    if source is None:
        print(f"Unknown source '{source_key}'")
        return
    page = source.page_range[0]
    subpage = source.subpage_range[0]
    url = _build_url(source, page, subpage)
    print(f"Fetching {url}...")
    try:
        resp = requests.get(url, timeout=15)
        print(f"HTTP {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('content-type', 'unknown')}")
        print(f"Content length: {len(resp.content)} bytes")
        if source.image_format == 'json_api':
            try:
                print(json.dumps(resp.json(), indent=2)[:2000])
            except ValueError:
                print(resp.text[:2000])
        else:
            print(resp.text[:2000])
    except requests.RequestException as e:
        print(f"Error: {e}")
