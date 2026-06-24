#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teletext.scraper import scrape_source, scrape_all_sources, inspect_source
from teletext.sources import SOURCES
from config import RAW_DIR, DEFAULT_SCRAPE_DELAY, DEFAULT_SOURCES


def list_sources() -> None:
    print(f"{'Key':<12} {'Display Name':<35} {'Lang':<6} {'Charset':<12} {'Format':<12} {'Pages':<10}")
    print("-" * 90)
    for key, src in sorted(SOURCES.items()):
        pr = f"{src.page_range[0]}-{src.page_range[1]}"
        print(f"{key:<12} {src.display_name:<35} {src.language:<6} {src.charset:<12} {src.image_format:<12} {pr:<10}")
    print(f"\n{len(SOURCES)} sources registered")


def parse_range(s: str) -> tuple:
    parts = s.split('-')
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape teletext sources")
    parser.add_argument("--sources", type=str, nargs="+",
                        help="Source keys to scrape, or 'all' for everything")
    parser.add_argument("--out", type=str, default=str(RAW_DIR),
                        help=f"Output directory (default: {RAW_DIR})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel sources to scrape (default: min(4, num_sources))")
    parser.add_argument("--delay", type=float, default=DEFAULT_SCRAPE_DELAY,
                        help=f"Seconds between requests (default: {DEFAULT_SCRAPE_DELAY})")
    parser.add_argument("--pages", type=str, default=None,
                        help="Page range override e.g. 100-200 for testing")
    parser.add_argument("--list", action="store_true",
                        help="List all available sources and exit")
    parser.add_argument("--inspect", type=str, default=None,
                        help="Inspect a single source's page structure and exit")
    args = parser.parse_args()

    if args.list:
        list_sources()
        return

    if args.inspect:
        inspect_source(args.inspect)
        return

    if not args.sources:
        parser.print_help()
        return

    page_range_override = parse_range(args.pages) if args.pages else None

    if 'all' in args.sources:
        source_keys = list(SOURCES.keys())
    else:
        source_keys = args.sources

    out_dir = Path(args.out)
    scrape_all_sources(source_keys, out_dir, args.delay, page_range_override,
                       max_workers=args.workers)
    print("\nDone!")


if __name__ == "__main__":
    main()
