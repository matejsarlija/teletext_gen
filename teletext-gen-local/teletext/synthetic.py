import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    W, H, NUM_COLORS, PAGE_RANGES, page_to_range,
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

NEWS_WORDS = [
    "IZBORI", "VLADA", "SABOR", "PREDSJEDNIK", "PREMJER",
    "MINISTAR", "OPORBA", "ZAKON", "PRORAČUN", "REFORMA",
    "KRIZA", "RAT", "MIR", "SPORAZUM", "SUMNJA",
    "ISTRAGA", "SUD", "TUŽITELJSTVO", "KORUPCIJA", "AFERA",
    "GOSPODARSTVO", "GOSPODARSKI", "RAST", "PAD", "INFLACIJA",
    "TEČAJ", "DUG", "KREDIT", "BANKA", "ULAGANJE",
    "ZDRAVLJE", "BOLNICA", "LIJEK", "LIJEČNIK", "CIJEPLJENJE",
    "OBRAZOVANJE", "ŠKOLA", "SVEUČILIŠTE", "STUDENT", "NASTAVA",
    "EUROPA", "EU", "BRUXELLES", "STRASBOURG", "SVIJET",
    "AMERIKA", "AZIJA", "BLISKI", "ISTOK", "RUSIJA",
    "UKRAJINA", "KINA", "SJEDINJENE", "DRŽAVE", "NJEMAČKA",
    "ANALIZA", "IZVJEŠĆE", "PROGNOZA", "AKTUALNO", "DNEVNO",
]

SPORTS_WORDS = [
    "NOGOMET", "RUKOMET", "KOSARKA", "ODBOJKA", "TENIS",
    "VATERPOLO", "RUKOMET", "ATLETIKA", "PLIVANJE", "BICIKLIZAM",
    "SKIJANJE", "SKOKOVI", "HRVANJE", "BOKS", "DŽUDO",
    "PRVENSTVO", "LIGA", "PRVA", "DRUGA", "TRČA",
    "UTAKMICA", "REZULTAT", "POBJEDA", "PORAZ", "NERIJEŠENO",
    "PRVI", "DRUGI", "TREČI", "ČETVRTI", "PETI",
    "GOL", "POGODAK", "STRIJELAC", "ASISTENCIJA", "KARTON",
    "PLASMAN", "KVALIFIKACIJE", "PRVENSTVO", "NATJECANJE", "TURNIR",
    "TRENER", "IGRAČ", "MOMČAD", "REPREZENTACIJA", "KLUB",
    "BODOVI", "TABLICA", "LJESTVICA", "RASPORED", "TERMIN",
]

INFO_WORDS = [
    "VRIJEME", "PROGNOZA", "TEMPERATURA", "OBORINE", "SUNCE",
    "OBLAČNO", "KIŠA", "SNIJEG", "MAGLA", "VJETAR",
    "MORE", "VALOVI", "PLIMA", "OSEKA", "BAROMETAR",
    "PUTOVANJE", "PROMET", "CESTOVNI", "ŽELJEZNIČKI", "ZRAČNI",
    "LETOVI", "POLAZAK", "DOLAZAK", "TERMINAL", "KARTA",
    "TEČAJNA", "LISTA", "VALUTA", "EUR", "USD",
    "TEČAJ", "KUPOVNI", "PRODAJNI", "SREDNJI", "BANKA",
    "HOROSKOP", "OVAN", "BIK", "BLIZANCI", "RAK",
    "LAV", "DJEVICA", "VAGA", "ŠKORPION", "STRIJELAC",
    "JARAC", "VODENJAK", "RIBE", "LOTO", "SREĆKA",
]

ENTERTAINMENT_WORDS = [
    "TELEVIZIJA", "PROGRAM", "RASPORED", "SERIJA", "FILM",
    "DOKUMENTARAC", "EMISIJA", "VODITELJ", "GLUMAC", "REDATELJ",
    "KINO", "PREDSTAVA", "KONCERT", "FESTIVAL", "GLAZBA",
    "POP", "ROCK", "JAZZ", "KLASIKA", "NARODNA",
    "KULTURA", "UMJETNOST", "IZLOŽBA", "MUZEJ", "KNJIŽNICA",
    "KNJIGA", "AUTOR", "NAKLADA", "BESTSELLER", "RECENZIJA",
    "RADIO", "SAT", "JUTARNJI", "POPODNEVNI", "VEČERNJI",
    "ZABAVA", "IGRA", "KVZ", "NATJECATELJ", "POBJEDNIK",
]

PROMOTION_WORDS = [
    "AKCIJA", "POPUST", "RASPRODAJA", "PONUDA", "JEFTINO",
    "NOVO", "OTVORENJE", "PREDSTAVLJANJE", "DOBRODOŠLI", "POSJETITE",
    "KUPITE", "ISKORISTITE", "BESPLATNO", "GARANCIJA", "KVALITETA",
    "USLUGA", "PROIZVOD", "CJENA", "POGODBA", "USLUGA",
    "Reklama", "OGLAS", "SPONZOR", "PARTNER", "PROMOCIJA",
]

SERVICE_WORDS = [
    "SERVIS", "TEHNIČKI", "PODRŠKA", "SUSTAV", "KORISNIK",
    "POMOĆ", "UPDATE", "VERZIJA", "KONFIGURACIJA", "INSTALACIJA",
    "SIGURNOST", "ZAŠTITA", "PRIJAVA", "KOD", "LOZINKA",
    "TEST", "KONTROLA", "PRACENJE", "STATUS", "DOJAVA",
    "UREDAJ", "POVEZIVANJE", "MREŽA", "SERVER", "BAZA",
]

SUBTITLE_WORDS = [
    "...", "...", "...", "TEKST", "PRIJEVOD",
    "GOVOR", "DIJALOG", "OPIS", "NAPOMENA",
]

ALL_RANGE_WORDS = {
    'index': ALL_WORDS,
    'news': NEWS_WORDS + CROATIAN_WORDS,
    'sports': SPORTS_WORDS,
    'info': INFO_WORDS,
    'entertainment': ENTERTAINMENT_WORDS,
    'subtitles': SUBTITLE_WORDS,
    'services': SERVICE_WORDS,
    'promotions': PROMOTION_WORDS,
}


def _random_word() -> str:
    return random.choices(ALL_WORDS, weights=_WORD_WEIGHTS, k=1)[0]


def _random_text_line(max_len: int = 40, word_list: Optional[List[str]] = None) -> str:
    words: List[str] = []
    total = 0
    pool = word_list or ALL_WORDS
    while total < max_len:
        word = random.choice(pool)
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


def _separator_row(vocab: Vocabulary, char: str, fg: int, bg: int) -> np.ndarray:
    line = char * W
    return _encode_text(vocab, line, fg, bg)


def _set_header(grid: np.ndarray, vocab: Vocabulary, page_number: int,
                label: str, bg: int, fg: int = 7) -> None:
    header_text = f" {page_number} {label} "
    mid = W // 2
    start = mid - len(header_text) // 2
    for i, ch in enumerate(header_text):
        pos = start + i
        if 0 <= pos < W:
            cid = _char_to_id(vocab, ch)
            if cid is not None:
                grid[0, pos] = _new_token(vocab, cid, fg, bg)
    for i in range(W):
        if grid[0, i] == 0:
            grid[0, i] = _new_token(vocab, SPACE_CHAR_ID, fg, bg)


def _set_footer(grid: np.ndarray, vocab: Vocabulary, page_number: int) -> None:
    bg = random.randint(1, NUM_COLORS - 1)
    fg = random.randint(1, NUM_COLORS - 1)
    footer_text = _random_text_line(36)
    footer_row = _encode_text(vocab, footer_text, fg, bg)
    grid[24, :] = footer_row
    for i in range(len(footer_text), W):
        if grid[24, i] == 0:
            grid[24, i] = _new_token(vocab, SPACE_CHAR_ID, fg, bg)


def _generate_index_grid(vocab: Vocabulary, page_number: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int64)
    header_bg = random.randint(1, NUM_COLORS - 1)
    _set_header(grid, vocab, page_number, "INDEX", header_bg, 7)

    num_items = random.randint(4, 10)
    item_rows = []
    available = list(range(2, H - 2))
    random.shuffle(available)
    item_rows = sorted(available[:num_items])

    sep_char = random.choice(["-", "=", "\u2500", "\u2550"])

    for ri, row_idx in enumerate(item_rows):
        item_fg = random.randint(1, NUM_COLORS - 1)
        item_bg = 0
        item_page = random.choice([p for p in range(100, 900) if p != page_number])
        label = _random_text_line(random.randint(10, 28))
        item_text = f"  {label}"
        page_str = f"{item_page}"
        if len(item_text) + len(page_str) + 4 < W:
            row = _encode_text(vocab, item_text, item_fg, item_bg)
            grid[row_idx, :len(item_text)] = row[:len(item_text)]
            page_start = W - len(page_str) - 2
            page_row = _encode_text(vocab, page_str, 3, item_bg)
            grid[row_idx, page_start:page_start + len(page_str)] = page_row[:len(page_str)]
            for i in range(W):
                if grid[row_idx, i] == 0:
                    grid[row_idx, i] = _new_token(vocab, SPACE_CHAR_ID, item_fg, item_bg)

        if ri < len(item_rows) - 1:
            sep_row_idx = item_rows[ri] + 1
            if sep_row_idx < item_rows[ri + 1] and sep_row_idx < H - 1:
                sep_fg = random.randint(1, NUM_COLORS - 1)
                grid[sep_row_idx, :] = _separator_row(vocab, sep_char, sep_fg, 0)

    _set_footer(grid, vocab, page_number)
    return grid


def _generate_news_grid(vocab: Vocabulary, page_number: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int64)
    header_bg = random.choice([1, 4, 0])
    header_fg = 7 if header_bg != 7 else 1
    _set_header(grid, vocab, page_number, "VIJESTI", header_bg, header_fg)

    date_fg = 3
    date_bg = 0
    date_text = "DANAS, 12. LIPNJA 2026."
    date_row = _encode_text(vocab, date_text, date_fg, date_bg)
    grid[1, :len(date_text)] = date_row[:len(date_text)]

    num_sections = random.randint(2, 4)
    body_start = 3
    for sec in range(num_sections):
        if body_start >= H - 2:
            break
        heading_fg = 3
        heading = _random_text_line(random.randint(20, 36), word_list=NEWS_WORDS)
        heading_row = _encode_text(vocab, heading.upper(), heading_fg, 0)
        grid[body_start, :len(heading)] = heading_row[:len(heading)]
        body_start += 1

        num_lines = random.randint(2, 5)
        for _ in range(num_lines):
            if body_start >= H - 2:
                break
            text_fg = random.choices([7, 6, 2], weights=[0.7, 0.2, 0.1])[0]
            line = _random_text_line(random.randint(30, 39), word_list=NEWS_WORDS)
            text_row = _encode_text(vocab, line, text_fg, 0)
            grid[body_start, :len(line)] = text_row[:len(line)]
            body_start += 1

        body_start += 1

    _set_footer(grid, vocab, page_number)
    return grid


def _generate_sports_grid(vocab: Vocabulary, page_number: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int64)
    header_bg = random.choice([2, 3, 6])
    _set_header(grid, vocab, page_number, "SPORT", header_bg, 0 if header_bg in (2, 3) else 7)

    league_text = _random_text_line(random.randint(20, 36), word_list=SPORTS_WORDS)
    league_row = _encode_text(vocab, league_text.upper(), 3, 0)
    grid[2, :len(league_text)] = league_row[:len(league_text)]

    num_games = min(random.randint(5, 12), H - 5)
    for i in range(num_games):
        row_idx = 3 + i
        if row_idx >= H - 1:
            break
        team1 = _random_text_line(random.randint(8, 16), word_list=SPORTS_WORDS)
        team2 = _random_text_line(random.randint(8, 16), word_list=SPORTS_WORDS)
        score = f"{random.randint(0, 5)}:{random.randint(0, 5)}"
        line = f"  {team1:<20} {score:>5}  {team2:<20}"
        line = line[:W]
        text_fg = 7
        score_fg = 3
        row = _encode_text(vocab, line, text_fg, 0)
        grid[row_idx, :len(line)] = row[:len(line)]
        score_start = line.find(score)
        if score_start >= 0:
            score_row = _encode_text(vocab, score, score_fg, 0)
            for j, ch in enumerate(score):
                cid = _char_to_id(vocab, ch)
                if cid is not None and score_start + j < W:
                    grid[row_idx, score_start + j] = _new_token(vocab, cid, score_fg, 0)

    _set_footer(grid, vocab, page_number)
    return grid


def _generate_info_grid(vocab: Vocabulary, page_number: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int64)
    header_bg = random.choice([4, 6, 1])
    _set_header(grid, vocab, page_number, "INFO", header_bg, 7)

    sections = [("VRIJEME", 2), ("PUTOVANJA", 8), ("TEČAJ", 14)]
    for title, start_row in sections:
        if start_row >= H - 2:
            break
        title_row = _encode_text(vocab, title, 3, 0)
        grid[start_row, :len(title)] = title_row[:len(title)]

        if title == "VRIJEME":
            cities = ["ZAGREB", "SPLIT", "RIJEKA", "OSIJEK", "DUBROVNIK"]
            temps = [f"{random.randint(-5, 38)}\u00b0" for _ in cities]
            for i, (city, temp) in enumerate(zip(cities, temps)):
                r = start_row + 1 + i
                if r >= H - 2:
                    break
                line = f"  {city:<20} {temp:>6}"
                row = _encode_text(vocab, line, random.choice([7, 6, 2]), 0)
                grid[r, :len(line)] = row[:len(line)]
        elif title == "TEČAJ":
            rates = [("EUR", random.uniform(7.0, 8.0)), ("USD", random.uniform(6.0, 7.5)),
                     ("GBP", random.uniform(8.5, 9.5)), ("CHF", random.uniform(7.0, 8.0))]
            for i, (cur, rate) in enumerate(rates):
                r = start_row + 1 + i
                if r >= H - 2:
                    break
                line = f"  {cur:<5} {rate:.4f}"
                row = _encode_text(vocab, line, 7, 0)
                grid[r, :len(line)] = row[:len(line)]

    _set_footer(grid, vocab, page_number)
    return grid


def _generate_entertainment_grid(vocab: Vocabulary, page_number: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int64)
    header_bg = random.choice([5, 6, 3])
    _set_header(grid, vocab, page_number, "ZABAVA", header_bg, 0)

    body_start = 2
    schedule = []
    for h in range(6, 23, 1):
        hour = f"{h:02d}:00"
        show = _random_text_line(random.randint(15, 30), word_list=ENTERTAINMENT_WORDS)
        schedule.append((hour, show))

    for i, (time_str, show) in enumerate(schedule):
        row_idx = body_start + i
        if row_idx >= H - 2:
            break
        time_fg = 3
        show_fg = random.choice([7, 6, 2, 5])
        line = f"{time_str}  {show}"
        if len(line) > W:
            line = line[:W]
        time_part = f"{time_str}  "
        p1 = _encode_text(vocab, time_part, time_fg, 0)
        grid[row_idx, :len(time_part)] = p1[:len(time_part)]
        show_part = line[len(time_part):]
        p2 = _encode_text(vocab, show_part, show_fg, 0)
        grid[row_idx, len(time_part):len(time_part) + len(show_part)] = p2[:len(show_part)]

    _set_footer(grid, vocab, page_number)
    return grid


def _generate_subtitles_grid(vocab: Vocabulary, page_number: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int64)

    page_str = str(page_number)
    offset = (W - len(page_str)) // 2
    for i, ch in enumerate(page_str):
        cid = _char_to_id(vocab, ch)
        if cid is not None:
            grid[0, offset + i] = _new_token(vocab, cid, 7, 0)
    for i in range(W):
        if grid[0, i] == 0:
            grid[0, i] = _new_token(vocab, SPACE_CHAR_ID, 7, 0)

    num_lines = random.randint(1, 4)
    line_starts = sorted(random.sample(range(8, 20), num_lines))
    for r in line_starts:
        text = _random_text_line(random.randint(12, 36))
        fg = 7
        bg = 0
        text_start = (W - len(text)) // 2
        row = _encode_text(vocab, text, fg, bg)
        grid[r, text_start:text_start + len(text)] = row[:len(text)]
        for i in range(W):
            if grid[r, i] == 0:
                grid[r, i] = _new_token(vocab, SPACE_CHAR_ID, fg, bg)

    return grid


def _generate_services_grid(vocab: Vocabulary, page_number: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int64)
    header_bg = random.randint(1, NUM_COLORS - 1)
    _set_header(grid, vocab, page_number, "SERVIS", header_bg, 7)

    entries = [
        ("STATUS SUSTAVA", "OPERATIVAN"),
        ("VERZIJA", f"{random.randint(1, 9)}.{random.randint(0, 9)}.{random.randint(0, 99)}"),
        ("ZADNJA PRIJAVA", f"{random.randint(1, 28)}.{random.randint(1, 12)}.2026"),
        ("KORISNICI", str(random.randint(10, 999))),
        ("UPTIME", f"{random.randint(0, 99)}h {random.randint(0, 59)}m"),
    ]
    for i, (key, val) in enumerate(entries):
        r = 2 + i * 3
        if r >= H - 2:
            break
        label_fg = 3
        label_row = _encode_text(vocab, f"  {key}:", label_fg, 0)
        grid[r, :len(key) + 4] = label_row[:len(key) + 4]
        val_fg = 7
        val_row = _encode_text(vocab, f"  {val}", val_fg, 0)
        grid[r + 1, :len(val) + 2] = val_row[:len(val) + 2]

    _set_footer(grid, vocab, page_number)
    return grid


def _generate_promotions_grid(vocab: Vocabulary, page_number: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int64)
    header_bg = random.choices([1, 5, 3, 6], weights=[0.4, 0.3, 0.2, 0.1])[0]
    _set_header(grid, vocab, page_number, "PONUDA", header_bg, 0 if header_bg in (3, 6) else 7)

    slogan = _random_text_line(random.randint(10, 30), word_list=PROMOTION_WORDS)
    slogan_start = (W - len(slogan)) // 2
    slogan_row = _encode_text(vocab, slogan, random.randint(1, 7), 0)
    grid[10, slogan_start:slogan_start + len(slogan)] = slogan_row[:len(slogan)]

    for r in [12, 13, 14]:
        mosaic_bg = random.randint(0, NUM_COLORS - 1)
        for col in range(W):
            char_id = random.randint(FIRST_MOSAIC_ID, LAST_MOSAIC_ID)
            fg = random.randint(0, NUM_COLORS - 1)
            grid[r, col] = _new_token(vocab, char_id, fg, mosaic_bg)

    price = f"{random.randint(1, 999)}.{random.randint(0, 99):02d}"
    price_text = f"  CJJENA: {price} EUR  "
    price_start = (W - len(price_text)) // 2
    price_row = _encode_text(vocab, price_text, 3, 1)
    grid[16, price_start:price_start + len(price_text)] = price_row[:len(price_text)]

    _set_footer(grid, vocab, page_number)
    return grid


_RANGE_GENERATORS = {
    'index': _generate_index_grid,
    'news': _generate_news_grid,
    'sports': _generate_sports_grid,
    'info': _generate_info_grid,
    'entertainment': _generate_entertainment_grid,
    'subtitles': _generate_subtitles_grid,
    'services': _generate_services_grid,
    'promotions': _generate_promotions_grid,
}


def generate_page(vocab: Vocabulary, page_number: Optional[int] = None,
                  page_range: Optional[str] = None) -> np.ndarray:
    if page_number is None:
        page_number = random.randint(100, 899)
    if page_range is None:
        page_range = page_to_range(page_number)

    generator = _RANGE_GENERATORS.get(page_range, _generate_news_grid)
    return generator(vocab, page_number)


def generate_dataset(n: int, vocab: Vocabulary, out_dir: Path, render: bool = True,
                     page_range: Optional[str] = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_and_save = None
    if render:
        from teletext.renderer import render_and_save

    for i in range(n):
        grid = generate_page(vocab, page_range=page_range)
        np.save(out_dir / f"grid_{i:06d}.npy", grid)
        if render_and_save is not None:
            render_and_save(grid, vocab, out_dir / f"page_{i:06d}.png")
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{n} pages")


def generate_dataset_balanced(target_per_range: int, vocab: Vocabulary,
                               out_dir: Path, render: bool = True) -> Dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_and_save = None
    if render:
        from teletext.renderer import render_and_save

    generated_counts: Dict[str, int] = {}
    total = 0
    for range_name, (lo, hi) in PAGE_RANGES.items():
        count = 0
        for _ in range(target_per_range):
            page_number = random.randint(lo, hi)
            grid = generate_page(vocab, page_number=page_number, page_range=range_name)
            np.save(out_dir / f"grid_{total:06d}.npy", grid)
            if render_and_save is not None:
                render_and_save(grid, vocab, out_dir / f"page_{total:06d}.png")
            total += 1
            count += 1
        generated_counts[range_name] = count

    return generated_counts
