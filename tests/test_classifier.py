"""
Tests for token classifier, Unicode ranges, and masking logic.
Runs without vLLM or GPU — only needs torch and transformers.
"""

import pytest
import torch
from language_logits_processor.languages.registry import LANGUAGES, COMMON_RANGES, CharRange
from language_logits_processor.classifier import (
    _in_ranges,
    _is_common,
    _char_languages,
    _token_languages,
)
from language_logits_processor.masking import (
    _cp_range_for_lead,
    _cp_range_for_lead_second,
    _overlaps_any,
    ByteGuardRules,
)


# ── Unicode range tests ─────────────────────────────────────────────────


class TestCharRanges:
    """Verify that representative characters land in the right language."""

    @pytest.mark.parametrize("char,expected_lang", [
        ("가", "ko"),
        ("힣", "ko"),
        ("ㄱ", "ko"),
        ("ㅎ", "ko"),
        ("ㅣ", "ko"),
    ])
    def test_korean_chars(self, char, expected_lang):
        langs = _char_languages(ord(char))
        assert expected_lang in langs

    @pytest.mark.parametrize("char,expected_lang", [
        ("A", "en"),
        ("z", "en"),
        ("é", "en"),
        ("ñ", "en"),  # also es, fr, etc.
    ])
    def test_english_chars(self, char, expected_lang):
        langs = _char_languages(ord(char))
        assert expected_lang in langs

    @pytest.mark.parametrize("char,expected_lang", [
        ("あ", "ja"),
        ("ア", "ja"),
        ("漢", "ja"),  # CJK — shared with zh
    ])
    def test_japanese_chars(self, char, expected_lang):
        langs = _char_languages(ord(char))
        assert expected_lang in langs

    @pytest.mark.parametrize("char,expected_lang", [
        ("中", "zh"),
        ("国", "zh"),
    ])
    def test_chinese_chars(self, char, expected_lang):
        langs = _char_languages(ord(char))
        assert expected_lang in langs

    def test_cjk_shared_between_ja_zh_not_ko(self):
        cp = ord("漢")
        langs = _char_languages(cp)
        assert "ja" in langs
        assert "zh" in langs
        assert "ko" not in langs

    @pytest.mark.parametrize("char", [
        "ค",  # Thai
        "ล",  # Thai
        "ก",  # Thai
    ])
    def test_thai_not_in_ko_en(self, char):
        langs = _char_languages(ord(char))
        assert "ko" not in langs
        assert "en" not in langs
        assert "th" in langs

    @pytest.mark.parametrize("char", [
        "д",  # Cyrillic
        "ж",  # Cyrillic
    ])
    def test_cyrillic_not_in_ko_en(self, char):
        langs = _char_languages(ord(char))
        assert "ko" not in langs
        assert "en" not in langs
        assert "ru" in langs

    @pytest.mark.parametrize("char", [
        "ع",  # Arabic
        "ب",  # Arabic
    ])
    def test_arabic_not_in_ko_en(self, char):
        langs = _char_languages(ord(char))
        assert "ko" not in langs
        assert "en" not in langs
        assert "ar" in langs


class TestCommonChars:
    """Characters that should be allowed regardless of language selection."""

    @pytest.mark.parametrize("char", [
        " ", ".", ",", "!", "?", ":", ";",
        "0", "5", "9",
        "(", ")", "[", "]", "{", "}",
        "\n", "\t",
        "→", "★", "—",
        "、", "。", "「", "」",  # CJK punctuation
    ])
    def test_common_chars(self, char):
        assert _is_common(ord(char)), f"'{char}' (U+{ord(char):04X}) should be common"

    @pytest.mark.parametrize("char", [
        "A", "z", "가", "あ", "ア", "中", "ค",
    ])
    def test_non_common_chars(self, char):
        assert not _is_common(ord(char)), f"'{char}' should not be common"


# ── Token-level classification tests ────────────────────────────────────


class TestTokenLanguages:

    def test_pure_korean_token(self):
        langs = _token_languages("안녕하세요")
        assert "ko" in langs
        assert "th" not in langs

    def test_pure_english_token(self):
        langs = _token_languages("Hello")
        assert "en" in langs
        assert "ko" not in langs

    def test_pure_common_token(self):
        langs = _token_languages("123")
        assert "__common__" in langs

    def test_punctuation_only(self):
        langs = _token_languages("...")
        assert "__common__" in langs

    def test_space_only(self):
        langs = _token_languages(" ")
        assert "__common__" in langs

    def test_mixed_ko_en(self):
        """Token with both Korean and English — both languages should match."""
        langs = _token_languages("Hello안녕")
        assert "ko" in langs
        assert "en" in langs

    def test_mixed_ko_common(self):
        """Korean + punctuation — Korean should match, common is ignored."""
        langs = _token_languages("안녕!")
        assert "ko" in langs
        assert "__common__" not in langs  # has non-common chars

    def test_thai_token_blocked_for_ko_en(self):
        """Thai-only token should not match ko or en."""
        langs = _token_languages("คล")
        assert "ko" not in langs
        assert "en" not in langs
        assert "th" in langs

    def test_lenient_mixed_with_unwanted(self):
        """
        Lenient mode: if ANY char is in a requested language, allow.
        Token 'aค' has 'a' (en) and 'ค' (th).
        Should match en AND th.
        """
        langs = _token_languages("aค")
        assert "en" in langs
        assert "th" in langs

    def test_empty_token(self):
        langs = _token_languages("")
        assert "__common__" in langs

    def test_emoji_is_common(self):
        langs = _token_languages("😊")
        assert "__common__" in langs

    def test_cjk_token_not_in_ko(self):
        """CJK ideographs are NOT in ko (removed)."""
        langs = _token_languages("漢字")
        assert "ko" not in langs
        assert "ja" in langs
        assert "zh" in langs


# ── Filtering logic simulation ──────────────────────────────────────────


class TestFilteringLogic:
    """
    Simulate what the processor does: given output_languages,
    which tokens pass?
    """

    def _would_allow(self, text: str, output_languages: list[str]) -> bool:
        """Simulate the lenient allow logic."""
        token_langs = _token_languages(text)
        if "__common__" in token_langs:
            return True
        return bool(token_langs & set(output_languages))

    def test_ko_en_allows_korean(self):
        assert self._would_allow("안녕", ["ko", "en"])

    def test_ko_en_allows_english(self):
        assert self._would_allow("Hello", ["ko", "en"])

    def test_ko_en_allows_numbers(self):
        assert self._would_allow("123", ["ko", "en"])

    def test_ko_en_allows_punctuation(self):
        assert self._would_allow("...", ["ko", "en"])

    def test_ko_en_blocks_thai(self):
        assert not self._would_allow("คล", ["ko", "en"])

    def test_ko_en_blocks_cyrillic(self):
        assert not self._would_allow("Привет", ["ko", "en"])

    def test_ko_en_blocks_arabic(self):
        assert not self._would_allow("مرحبا", ["ko", "en"])

    def test_ko_en_blocks_hanja(self):
        """CJK ideographs are NOT in ko — blocked for ko+en."""
        assert not self._would_allow("漢字", ["ko", "en"])

    def test_ja_allows_hanja(self):
        """CJK ideographs ARE in ja."""
        assert self._would_allow("漢字", ["ja"])

    def test_ko_ja_allows_hanja(self):
        """ko+ja combo allows CJK via ja."""
        assert self._would_allow("漢字", ["ko", "ja"])

    def test_en_only_blocks_korean(self):
        assert not self._would_allow("안녕", ["en"])

    def test_ko_only_blocks_english(self):
        assert not self._would_allow("Hello", ["ko"])

    def test_ko_only_allows_korean(self):
        assert self._would_allow("안녕하세요", ["ko"])

    def test_th_allows_thai(self):
        assert self._would_allow("สวัสดี", ["th"])

    def test_no_filter_without_output_languages(self):
        """Without output_languages, everything passes (processor not active)."""
        pass

    def test_the_actual_bug_case(self):
        """
        The bug: '헷คล렸나' — Thai chars mixed into Korean.
        With ko+en filter, pure Thai tokens would be blocked.
        """
        assert not self._would_allow("คล", ["ko", "en"])
        assert self._would_allow("헷", ["ko", "en"])
        assert self._would_allow("렸나", ["ko", "en"])


# ── Byte guard: UTF-8 range computation tests ──────────────────────────


class TestByteGuardRanges:
    """Test the UTF-8 codepoint range helper functions."""

    def test_lead_2byte_cyrillic(self):
        lo, hi = _cp_range_for_lead(0xD0)
        assert lo == 0x0400
        assert hi == 0x043F

    def test_lead_3byte_cjk(self):
        lo, hi = _cp_range_for_lead(0xE4)
        assert lo == 0x4000
        assert hi == 0x4FFF

    def test_lead_3byte_e3(self):
        lo, hi = _cp_range_for_lead(0xE3)
        assert lo == 0x3000
        assert hi == 0x3FFF

    def test_lead_4byte(self):
        lo, hi = _cp_range_for_lead(0xF0)
        assert lo == 0x10000
        assert hi == 0x3FFFF

    def test_lead_invalid_ascii(self):
        assert _cp_range_for_lead(0x41) is None

    def test_lead_invalid_continuation(self):
        assert _cp_range_for_lead(0x80) is None

    def test_lead_second_2byte(self):
        lo, hi = _cp_range_for_lead_second(0xD0, 0x90)
        assert lo == hi  # single codepoint
        assert lo == 0x0410  # Cyrillic А

    def test_lead_second_3byte_hiragana(self):
        lo, hi = _cp_range_for_lead_second(0xE3, 0x81)
        assert lo == 0x3040
        assert hi == 0x307F

    def test_lead_second_3byte_hangul_compat(self):
        lo, hi = _cp_range_for_lead_second(0xE3, 0x84)
        assert lo == 0x3100
        assert hi == 0x313F

    def test_lead_second_4byte_emoji(self):
        lo, hi = _cp_range_for_lead_second(0xF0, 0x9F)
        assert lo == 0x1F000
        assert hi == 0x1FFFF

    def test_lead_second_4byte_cjk_ext(self):
        lo, hi = _cp_range_for_lead_second(0xF0, 0xA0)
        assert lo == 0x20000
        assert hi == 0x20FFF

    def test_lead_second_invalid(self):
        assert _cp_range_for_lead_second(0xE3, 0x40) is None

    def test_overlaps_any_hit(self):
        ranges = [CharRange(0x3040, 0x309F)]
        assert _overlaps_any(0x3040, 0x307F, ranges)

    def test_overlaps_any_miss(self):
        ranges = [CharRange(0x3040, 0x309F)]
        assert not _overlaps_any(0x4000, 0x4FFF, ranges)

    def test_overlaps_any_partial(self):
        ranges = [CharRange(0x3130, 0x318F)]
        assert _overlaps_any(0x3100, 0x313F, ranges)


# ── Byte guard: rule computation and application ───────────────────────


def _make_guard(languages: list[str]) -> ByteGuardRules:
    """Build a ByteGuardRules with byte_token_ids = range(256), vocab=256."""
    allowed: list[CharRange] = list(COMMON_RANGES)
    for lang in languages:
        allowed.extend(LANGUAGES[lang])
    byte_ids = list(range(256))
    return ByteGuardRules(allowed, byte_ids, 256, torch.device("cpu"))


class TestByteGuardRules:
    """Test byte guard rule computation for various language combos."""

    def test_ko_en_bans_cjk_lead_bytes(self):
        guard = _make_guard(["ko", "en"])
        for lead in range(0xE4, 0xEA):
            assert lead in guard.static_ban_ids, f"0x{lead:02X} should be banned"

    def test_ko_en_bans_thai_lead(self):
        guard = _make_guard(["ko", "en"])
        assert 0xE0 in guard.static_ban_ids

    def test_ko_en_bans_cyrillic_leads(self):
        guard = _make_guard(["ko", "en"])
        for lead in range(0xD0, 0xD4):
            assert lead in guard.static_ban_ids, f"0x{lead:02X} should be banned"

    def test_ko_en_allows_hangul_leads(self):
        guard = _make_guard(["ko", "en"])
        for lead in [0xEA, 0xEB, 0xEC, 0xED]:
            assert lead not in guard.static_ban_ids, f"0x{lead:02X} should be allowed"

    def test_ko_en_allows_latin_leads(self):
        guard = _make_guard(["ko", "en"])
        assert 0xC3 not in guard.static_ban_ids

    def test_ko_en_has_dynamic_rules(self):
        """E3 has both Hiragana (banned) and Hangul Compat (allowed)."""
        guard = _make_guard(["ko", "en"])
        assert guard.has_dynamic_rules

    def test_th_allows_thai_lead(self):
        guard = _make_guard(["th"])
        assert 0xE0 not in guard.static_ban_ids

    def test_ja_allows_cjk_leads(self):
        guard = _make_guard(["ja"])
        for lead in range(0xE4, 0xEA):
            assert lead not in guard.static_ban_ids

    def test_ru_allows_cyrillic_leads(self):
        guard = _make_guard(["ru"])
        for lead in [0xD0, 0xD1]:
            assert lead not in guard.static_ban_ids


class TestByteGuardApply:
    """Test byte guard dynamic rule application."""

    def test_blocks_hiragana_second_byte_after_e3(self):
        guard = _make_guard(["ko", "en"])
        logits = torch.zeros(1, 256)
        rows = torch.tensor([0])
        last_tokens = torch.tensor([0xE3])
        guard.apply(logits, rows, last_tokens)
        assert logits[0, 0x81].item() == float("-inf")  # Hiragana

    def test_allows_hangul_compat_second_byte_after_e3(self):
        guard = _make_guard(["ko", "en"])
        logits = torch.zeros(1, 256)
        rows = torch.tensor([0])
        last_tokens = torch.tensor([0xE3])
        guard.apply(logits, rows, last_tokens)
        assert logits[0, 0x84].item() == 0.0  # Hangul Compat Jamo area

    def test_no_effect_when_last_is_not_lead(self):
        guard = _make_guard(["ko", "en"])
        logits = torch.zeros(1, 256)
        rows = torch.tensor([0])
        last_tokens = torch.tensor([0x41])  # ASCII 'A', not a lead byte
        guard.apply(logits, rows, last_tokens)
        assert (logits == 0.0).all()

    def test_batch_selective(self):
        """Only rows in row_indices are affected."""
        guard = _make_guard(["ko", "en"])
        logits = torch.zeros(3, 256)
        rows = torch.tensor([1])  # only row 1
        last_tokens = torch.tensor([0xE3])
        guard.apply(logits, rows, last_tokens)
        assert (logits[0] == 0.0).all()  # untouched
        assert logits[1, 0x81].item() == float("-inf")  # affected
        assert (logits[2] == 0.0).all()  # untouched

    def test_f0_blocks_cjk_ext_allows_emoji(self):
        guard = _make_guard(["ko", "en"])
        logits = torch.zeros(1, 256)
        rows = torch.tensor([0])
        last_tokens = torch.tensor([0xF0])
        guard.apply(logits, rows, last_tokens)
        assert logits[0, 0x9F].item() == 0.0  # emoji area allowed
        assert logits[0, 0xA0].item() == float("-inf")  # CJK ext B banned
