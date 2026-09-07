# SPDX-License-Identifier: Apache-2.0
"""
Token classifier — maps each token in a vocabulary to the set of languages
whose Unicode ranges it touches.

No vLLM dependency. Requires only torch and a HuggingFace tokenizer.
"""

import logging
from typing import TYPE_CHECKING

import torch

from language_logits_processor.languages import COMMON_RANGES, LANGUAGES, CharRange

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def detect_byte_tokens(tokenizer: "PreTrainedTokenizerBase") -> list[int] | None:
    """
    Detect byte-fallback tokens (<0x00> through <0xFF>).
    Returns list of 256 token IDs where index = byte value, or None
    if the tokenizer does not have byte-fallback tokens.
    """
    try:
        byte_ids = []
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for b in range(256):
            token = f"<0x{b:02X}>"
            tid = tokenizer.convert_tokens_to_ids(token)
            if tid is None or tid == unk_id:
                return None
            byte_ids.append(tid)
        return byte_ids
    except Exception:
        return None


def _in_ranges(codepoint: int, ranges: list[CharRange]) -> bool:
    return any(r.start <= codepoint <= r.end for r in ranges)


def _is_common(codepoint: int) -> bool:
    return _in_ranges(codepoint, COMMON_RANGES)


def _char_languages(codepoint: int) -> set[str]:
    langs = set()
    for lang, ranges in LANGUAGES.items():
        if _in_ranges(codepoint, ranges):
            langs.add(lang)
    return langs


def _token_languages(text: str) -> set[str]:
    """
    Lenient: a token belongs to every language that ANY of its non-common
    characters match. Pure-common tokens get "__common__".
    """
    langs: set[str] = set()
    has_non_common = False
    for ch in text:
        cp = ord(ch)
        if _is_common(cp):
            continue
        has_non_common = True
        langs.update(_char_languages(cp))

    if not has_non_common:
        langs.add("__common__")
    return langs


def classify_vocabulary(
    tokenizer: "PreTrainedTokenizerBase",
    vocab_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Scan the full vocabulary once and return:
      lang_masks:   {lang_code: bool tensor (vocab_size,)}
      common_mask:  bool tensor — True for pure-common tokens
      special_mask: bool tensor — True for special tokens (BOS/EOS/PAD/UNK)
    """
    special_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)

    lang_codes = list(LANGUAGES.keys())
    lang_hits: dict[str, list[bool]] = {lang: [] for lang in lang_codes}
    common_hits: list[bool] = []
    special_hits: list[bool] = []

    unknown_count = 0
    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            text = ""

        if token_id in special_ids or not text.strip():
            special_hits.append(True)
            common_hits.append(False)
            for lang in lang_codes:
                lang_hits[lang].append(False)
            continue

        special_hits.append(False)
        token_langs = _token_languages(text)
        is_common = "__common__" in token_langs
        common_hits.append(is_common)

        is_any_lang = False
        for lang in lang_codes:
            hit = lang in token_langs
            lang_hits[lang].append(hit)
            if hit:
                is_any_lang = True

        if not is_common and not is_any_lang:
            unknown_count += 1

    if unknown_count > 0:
        logger.info(
            "Language filter: %d/%d tokens matched no language and are not "
            "common — they will be blocked when output_languages is set.",
            unknown_count, vocab_size,
        )

    lang_masks = {
        lang: torch.tensor(lang_hits[lang], dtype=torch.bool, device=device)
        for lang in lang_codes
    }
    common_mask = torch.tensor(common_hits, dtype=torch.bool, device=device)
    special_mask = torch.tensor(special_hits, dtype=torch.bool, device=device)

    logger.info(
        "Language filter vocabulary classified: vocab_size=%d, "
        "ko=%d, en=%d, common=%d, special=%d",
        vocab_size,
        lang_masks.get("ko", torch.zeros(1)).sum().item(),
        lang_masks.get("en", torch.zeros(1)).sum().item(),
        common_mask.sum().item(),
        special_mask.sum().item(),
    )

    return lang_masks, common_mask, special_mask
