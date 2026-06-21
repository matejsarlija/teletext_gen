# teletext-gen-local

Local/CPU portion of a teletext generative AI project. Handles data preparation, scraping, rendering, and inference. Training happens separately on Colab.

## Setup

```bash
cd teletext-gen-local
pip install -r requirements.txt
```

## Workflow

Run commands from this directory (`teletext-gen-local/`). The local workflow is
CPU-only: scraping, synthetic data generation, vocabulary building, packaging,
and inference do not require CUDA. GPU training is intentionally out of scope
for local scripts and should run on Colab.

```bash
# 1. Scrape teletext sources
python scripts/scrape.py --sources hrt --pages 100-899 --out data/raw/

# 2. Generate synthetic training data
python scripts/generate_synthetic.py --n 50000 --out data/synthetic/ --no-render

# 3. Build vocabulary from all available token grids
python scripts/build_vocab.py

# 4. Prepare and zip datasets for Colab upload
python scripts/prepare_colab.py

# 5. After training on Colab, run inference
python scripts/sample.py --weights path/to/generator.pt --n 16 --out samples/
```

## Tests

```bash
pytest
```

The test suite checks local library behavior: vocabulary roundtrips, synthetic
grid validity, rendering dimensions, model forward tensor shapes, and Colab
dataset packaging with tiny temporary fixtures. It does not run a local training
path.

## Project Structure

```
teletext-gen-local/
├── config.py              # Grid dimensions, palette, paths, scraping config
├── teletext/
│   ├── vocab.py           # Vocabulary (deterministic token encoding)
│   ├── renderer.py        # Token grid -> PIL image rendering
│   ├── synthetic.py       # Structured random page generation
│   ├── scraper.py         # HRT teletext website scraper
│   └── utils.py           # Helper utilities
├── models/
│   └── architectures.py   # Self-contained model definitions (no local imports)
├── scripts/               # CLI entry points
├── tests/                 # Local unit tests; no training workflow
├── data/                  # Data directories
└── notebooks/             # Exploration notebooks
```

## Design

- Grid: 40x25 cells, each 13x16px → 520x400 image
- EBU teletext 8-color palette
- Token = (char_id, fg_color, bg_color) — 128 char_ids × 8 × 8 = 8192 possible tokens
- Deterministic token encoding: `id = char_id * 64 + fg * 8 + bg + 3`
- All data as .npy files for speed and simplicity
- CPU-only — no CUDA assumptions in local code
