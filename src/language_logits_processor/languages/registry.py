"""
Language-to-Unicode-range registry.

Each language maps to a list of (start, end) codepoint ranges (inclusive).
A character belongs to a language if it falls in any of that language's ranges.
Characters in COMMON_RANGES are always allowed regardless of language selection.
"""

from typing import NamedTuple


class CharRange(NamedTuple):
    start: int
    end: int  # inclusive


COMMON_RANGES: list[CharRange] = [
    # ASCII digits
    CharRange(0x0030, 0x0039),
    # ASCII punctuation & symbols
    CharRange(0x0020, 0x002F),
    CharRange(0x003A, 0x0040),
    CharRange(0x005B, 0x0060),
    CharRange(0x007B, 0x007E),
    # Whitespace / control
    CharRange(0x0009, 0x000D),
    # General Punctuation
    CharRange(0x2000, 0x206F),
    # CJK Symbols and Punctuation
    CharRange(0x3000, 0x303F),
    # Fullwidth ASCII variants
    CharRange(0xFF01, 0xFF60),
    # Halfwidth CJK punctuation
    CharRange(0xFF61, 0xFF65),
    # Superscripts and Subscripts
    CharRange(0x2070, 0x209F),
    # Currency Symbols
    CharRange(0x20A0, 0x20CF),
    # Number Forms
    CharRange(0x2150, 0x218F),
    # Arrows
    CharRange(0x2190, 0x21FF),
    # Mathematical Operators
    CharRange(0x2200, 0x22FF),
    # Box Drawing
    CharRange(0x2500, 0x257F),
    # Geometric Shapes
    CharRange(0x25A0, 0x25FF),
    # Miscellaneous Symbols
    CharRange(0x2600, 0x26FF),
    # Dingbats
    CharRange(0x2700, 0x27BF),
    # Emoji
    CharRange(0x1F300, 0x1F9FF),
    CharRange(0x1FA00, 0x1FA6F),
    CharRange(0x1FA70, 0x1FAFF),
    # Latin-1 Supplement punctuation
    CharRange(0x00A0, 0x00BF),
    # Enclosed Alphanumerics
    CharRange(0x2460, 0x24FF),
]


LANGUAGES: dict[str, list[CharRange]] = {
    # Korean
    "ko": [
        CharRange(0xAC00, 0xD7AF),  # Hangul Syllables
        CharRange(0x1100, 0x11FF),  # Hangul Jamo
        CharRange(0x3130, 0x318F),  # Hangul Compatibility Jamo
        CharRange(0xA960, 0xA97F),  # Hangul Jamo Extended-A
        CharRange(0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    ],
    # English
    "en": [
        CharRange(0x0041, 0x005A),  # A-Z
        CharRange(0x0061, 0x007A),  # a-z
        CharRange(0x00C0, 0x00FF),  # Latin-1 Supplement letters
        CharRange(0x0100, 0x017F),  # Latin Extended-A
        CharRange(0x0180, 0x024F),  # Latin Extended-B
    ],
    # Japanese
    "ja": [
        CharRange(0x3040, 0x309F),  # Hiragana
        CharRange(0x30A0, 0x30FF),  # Katakana
        CharRange(0x31F0, 0x31FF),  # Katakana Phonetic Extensions
        CharRange(0xFF66, 0xFF9F),  # Halfwidth Katakana
        CharRange(0x4E00, 0x9FFF),  # CJK Unified Ideographs (kanji)
        CharRange(0x3400, 0x4DBF),  # CJK Extension A
    ],
    # Chinese
    "zh": [
        CharRange(0x4E00, 0x9FFF),  # CJK Unified Ideographs
        CharRange(0x3400, 0x4DBF),  # CJK Extension A
        CharRange(0x20000, 0x2A6DF),  # CJK Extension B
        CharRange(0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        CharRange(0x3100, 0x312F),  # Bopomofo
        CharRange(0x31A0, 0x31BF),  # Bopomofo Extended
    ],
    # Thai
    "th": [
        CharRange(0x0E00, 0x0E7F),  # Thai
    ],
    # Vietnamese
    "vi": [
        CharRange(0x0041, 0x005A),  # A-Z
        CharRange(0x0061, 0x007A),  # a-z
        CharRange(0x00C0, 0x00FF),  # Latin-1 Supplement
        CharRange(0x0100, 0x024F),  # Latin Extended-A/B
        CharRange(0x1E00, 0x1EFF),  # Latin Extended Additional
        CharRange(0x0300, 0x036F),  # Combining Diacritical Marks
    ],
    # Arabic
    "ar": [
        CharRange(0x0600, 0x06FF),  # Arabic
        CharRange(0x0750, 0x077F),  # Arabic Supplement
        CharRange(0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        CharRange(0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ],
    # Russian
    "ru": [
        CharRange(0x0400, 0x04FF),  # Cyrillic
        CharRange(0x0500, 0x052F),  # Cyrillic Supplement
    ],
    # German
    "de": [
        CharRange(0x0041, 0x005A),  # A-Z
        CharRange(0x0061, 0x007A),  # a-z
        CharRange(0x00C0, 0x00FF),  # Latin-1 Supplement
        CharRange(0x0100, 0x017F),  # Latin Extended-A
    ],
    # French
    "fr": [
        CharRange(0x0041, 0x005A),  # A-Z
        CharRange(0x0061, 0x007A),  # a-z
        CharRange(0x00C0, 0x00FF),  # Latin-1 Supplement
        CharRange(0x0100, 0x017F),  # Latin Extended-A
    ],
    # Spanish
    "es": [
        CharRange(0x0041, 0x005A),  # A-Z
        CharRange(0x0061, 0x007A),  # a-z
        CharRange(0x00C0, 0x00FF),  # Latin-1 Supplement
        CharRange(0x0100, 0x017F),  # Latin Extended-A
    ],
    # Portuguese
    "pt": [
        CharRange(0x0041, 0x005A),  # A-Z
        CharRange(0x0061, 0x007A),  # a-z
        CharRange(0x00C0, 0x00FF),  # Latin-1 Supplement
        CharRange(0x0100, 0x017F),  # Latin Extended-A
    ],
}
