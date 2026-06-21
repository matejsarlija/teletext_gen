NATIONAL_CHARSETS = {
    'ascii': {},
    'croatian': {
        0x23: 'š', 0x24: 'đ', 0x40: 'Š',
        0x5B: 'Ć', 0x5C: 'Č', 0x5D: 'Ž',
        0x60: 'ž', 0x7B: 'ć', 0x7C: 'č',
    },
    'slovenian': {
        0x23: 'š', 0x24: 'đ', 0x40: 'Š',
        0x5B: 'Ć', 0x5C: 'Č', 0x5D: 'Ž',
        0x60: 'ž', 0x7B: 'ć', 0x7C: 'č',
    },
    'bosnian': {
        0x23: 'š', 0x24: 'đ', 0x40: 'Š',
        0x5B: 'Ć', 0x5C: 'Č', 0x5D: 'Ž',
        0x60: 'ž', 0x7B: 'ć', 0x7C: 'č',
    },
    'czech': {
        0x23: '#', 0x40: 'č', 0x5B: 'ť',
        0x5C: 'ž', 0x5D: 'ý', 0x5E: 'í',
        0x60: 'é', 0x7B: 'á', 0x7C: 'ě',
        0x7D: 'ú', 0x7E: 'š',
    },
    'dutch': {
        0x23: 'é', 0x40: 'ê', 0x5B: 'ä',
        0x5C: 'ö', 0x5D: 'ü', 0x5E: '^',
        0x60: '`', 0x7B: 'ë', 0x7C: 'ï',
        0x7D: 'ú', 0x7E: 'û',
    },
    'swedish': {
        0x23: '#', 0x40: 'É', 0x5B: 'Ä',
        0x5C: 'Ö', 0x5D: 'Å', 0x5E: 'Ü',
        0x60: 'é', 0x7B: 'ä', 0x7C: 'ö',
        0x7D: 'å', 0x7E: 'ü',
    },
    'portuguese': {
        0x23: 'ç', 0x40: 'Á', 0x5B: 'Ã',
        0x5C: 'Â', 0x5D: 'Ê', 0x5E: 'Ú',
        0x60: 'á', 0x7B: 'ã', 0x7C: 'â',
        0x7D: 'ê', 0x7E: 'ú',
    },
    'german': {
        0x23: '#', 0x40: 'Ä', 0x5B: 'Ö',
        0x5C: 'Ü', 0x5D: '^', 0x5E: '_',
        0x60: '`', 0x7B: 'ä', 0x7C: 'ö',
        0x7D: 'ü', 0x7E: 'ß',
    },
}


def build_reverse_charset(charset):
    return {ch: byte_pos for byte_pos, ch in charset.items()}


REVERSE_CHARSETS = {
    name: build_reverse_charset(cs)
    for name, cs in NATIONAL_CHARSETS.items()
}


def decode_char_with_charset(char_id: int, charset_key: str) -> str:
    """Convert char_id to display char using a specific national charset."""
    from config import SPACE_CHAR_ID, LAST_PRINTABLE_CHAR_ID
    if char_id == SPACE_CHAR_ID:
        return ' '
    if 1 <= char_id <= LAST_PRINTABLE_CHAR_ID:
        byte_val = char_id + 32
        subs = NATIONAL_CHARSETS.get(charset_key, {})
        return subs.get(byte_val, chr(byte_val))
    return ''


def char_to_id_with_charset(char: str, charset_key: str) -> int:
    """Convert display char to char_id using a specific national charset.
    Checks substitutions first, then falls back to plain ASCII mapping.
    """
    subs = NATIONAL_CHARSETS.get(charset_key, {})
    for rev_byte, rev_char in subs.items():
        if rev_char == char:
            return rev_byte - 32
    byte_pos = ord(char)
    if 32 <= byte_pos <= 126:
        return byte_pos - 32
    return None
