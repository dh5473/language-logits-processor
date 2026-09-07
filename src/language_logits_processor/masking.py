"""
Mask cache, byte-fallback guard, and combo mask.

No vLLM dependency. Requires only torch and a HuggingFace tokenizer.

Two layers of filtering:
  1. Static mask — precomputed per language combo, bans all tokens whose
     decoded text is outside allowed Unicode ranges.
  2. Byte-fallback guard — prevents the model from assembling banned
     characters one byte at a time via byte-fallback tokens (<0x00>..<0xFF>).
     Derived automatically from the allowed Unicode ranges.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from language_logits_processor.classifier import classify_vocabulary, detect_byte_tokens
from language_logits_processor.languages import COMMON_RANGES, LANGUAGES, CharRange

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

NEG_INF = float("-inf")


# ── UTF-8 codepoint range helpers ──────────────────────────────────────


def _cp_range_for_lead(lead: int) -> tuple[int, int] | None:
    """Codepoint range reachable from a UTF-8 lead byte."""
    if 0xC2 <= lead <= 0xDF:
        lo = (lead & 0x1F) << 6
        return (lo, lo | 0x3F)
    elif 0xE0 <= lead <= 0xEF:
        lo = (lead & 0x0F) << 12
        hi = lo | 0x0FFF
        if lead == 0xE0:
            lo = 0x0800
        if lead == 0xED:
            hi = 0xD7FF
        return (lo, hi)
    elif 0xF0 <= lead <= 0xF4:
        lo = (lead & 0x07) << 18
        hi = min(lo | 0x3FFFF, 0x10FFFF)
        if lead == 0xF0:
            lo = 0x10000
        return (lo, hi)
    return None


def _cp_range_for_lead_second(lead: int, second: int) -> tuple[int, int] | None:
    """Codepoint range reachable from a UTF-8 lead + second byte pair."""
    if not (0x80 <= second <= 0xBF):
        return None
    if lead == 0xE0 and second < 0xA0:
        return None
    if lead == 0xED and second > 0x9F:
        return None
    if lead == 0xF0 and second < 0x90:
        return None
    if lead == 0xF4 and second > 0x8F:
        return None
    if 0xC2 <= lead <= 0xDF:
        cp = ((lead & 0x1F) << 6) | (second & 0x3F)
        return (cp, cp)
    elif 0xE0 <= lead <= 0xEF:
        lo = ((lead & 0x0F) << 12) | ((second & 0x3F) << 6)
        return (lo, lo | 0x3F)
    elif 0xF0 <= lead <= 0xF4:
        lo = ((lead & 0x07) << 18) | ((second & 0x3F) << 12)
        return (lo, min(lo | 0x0FFF, 0x10FFFF))
    return None


def _overlaps_any(cp_lo: int, cp_hi: int, ranges: list[CharRange]) -> bool:
    """Check if [cp_lo, cp_hi] overlaps with any CharRange."""
    return any(r.start <= cp_hi and cp_lo <= r.end for r in ranges)


# ── Byte-fallback guard ────────────────────────────────────────────────


class ByteGuardRules:
    """
    Byte-fallback guard rules for a specific set of allowed Unicode ranges.

    Computed automatically from the allowed ranges:
      - static_ban_ids: byte token IDs for lead bytes that ONLY start
        banned codepoints — folded into the static mask.
      - Dynamic second-byte rules: for lead bytes that start BOTH allowed
        and banned codepoints, conditionally ban specific second bytes
        based on the previous generated token.
    """

    def __init__(
        self,
        allowed_ranges: list[CharRange],
        byte_token_ids: list[int],
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.static_ban_ids: list[int] = []
        self._lead_lut = torch.full(
            (vocab_size,), -1, dtype=torch.long, device=device,
        )
        self._second_byte_rules: list[tuple[int, torch.Tensor]] = []

        self._regulated_token_ids: set[int] = set()

        for lead in range(0xC2, 0xF5):
            cp_range = _cp_range_for_lead(lead)
            if cp_range is None:
                continue

            if not _overlaps_any(cp_range[0], cp_range[1], allowed_ranges):
                self.static_ban_ids.append(byte_token_ids[lead])
            else:
                banned_seconds: list[int] = []
                for second in range(0x80, 0xC0):
                    cp_range2 = _cp_range_for_lead_second(lead, second)
                    if cp_range2 is not None and not _overlaps_any(
                        cp_range2[0], cp_range2[1], allowed_ranges,
                    ):
                        banned_seconds.append(second)
                if banned_seconds:
                    token_id = byte_token_ids[lead]
                    self._lead_lut[token_id] = lead
                    self._regulated_token_ids.add(token_id)
                    mask = torch.zeros(
                        vocab_size, dtype=torch.bool, device=device,
                    )
                    for s in banned_seconds:
                        mask[byte_token_ids[s]] = True
                    self._second_byte_rules.append((lead, mask))

    @property
    def has_dynamic_rules(self) -> bool:
        return len(self._second_byte_rules) > 0

    def apply(
        self,
        logits: torch.Tensor,
        row_indices: torch.Tensor,
        last_token_ids: torch.Tensor,
    ) -> None:
        """
        Apply dynamic byte-guard rules to specific rows of the logits tensor.

        Args:
            logits: [B, V] float tensor (modified in-place)
            row_indices: [N] long tensor — which rows to apply to
            last_token_ids: [N] long tensor — last generated token per row
        """
        if not self._second_byte_rules:
            return
        prev_lead = self._lead_lut[last_token_ids]
        for lead_val, banned_mask in self._second_byte_rules:
            affected = prev_lead == lead_val
            if affected.any():
                affected_rows = row_indices[affected]
                logits[affected_rows] = logits[affected_rows].masked_fill(
                    banned_mask, NEG_INF,
                )


# ── Combo mask ─────────────────────────────────────────────────────────


class ComboMask:
    """Static float mask + optional byte guard for a language combo."""

    __slots__ = ("mask", "byte_guard")

    def __init__(
        self,
        mask: torch.Tensor,
        byte_guard: ByteGuardRules | None,
    ) -> None:
        self.mask = mask
        self.byte_guard = byte_guard


# ── Mask cache ─────────────────────────────────────────────────────────


class MaskCache:
    """
    Precomputes per-language bool masks at init time, then serves
    ComboMask instances for arbitrary language combos on demand.

    Combined masks are cached — repeated requests for the same
    language combo (e.g. ("en", "ko")) reuse the same tensors.
    """

    SUPPORTED_LANGUAGES = frozenset(LANGUAGES.keys())

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self._device = device
        self._vocab_size = vocab_size

        lang_masks, common_mask, special_mask = classify_vocabulary(
            tokenizer, vocab_size, device,
        )
        self._lang_masks = lang_masks

        self._byte_token_ids = detect_byte_tokens(tokenizer)

        byte_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if self._byte_token_ids is not None:
            for bid in self._byte_token_ids:
                byte_mask[bid] = True
            logger.info(
                "Language filter: detected %d byte-fallback tokens",
                len(self._byte_token_ids),
            )

        self._always_allowed = common_mask | special_mask | byte_mask
        self._cache: dict[tuple[str, ...], ComboMask] = {}

    def get(self, languages: tuple[str, ...]) -> ComboMask:
        """
        Return a ComboMask for the given language combo.

        The result is cached and the static mask must not be modified
        in place (byte guard modifies logits directly, not the mask).
        """
        if languages in self._cache:
            return self._cache[languages]

        allowed = self._always_allowed.clone()
        for lang in languages:
            allowed |= self._lang_masks[lang]

        static_mask = torch.where(
            allowed,
            torch.tensor(0.0, device=self._device),
            torch.tensor(NEG_INF, device=self._device),
        )

        byte_guard = None
        if self._byte_token_ids is not None:
            all_ranges: list[CharRange] = list(COMMON_RANGES)
            for lang in languages:
                all_ranges.extend(LANGUAGES[lang])
            byte_guard = ByteGuardRules(
                all_ranges,
                self._byte_token_ids,
                self._vocab_size,
                self._device,
            )
            for byte_id in byte_guard.static_ban_ids:
                static_mask[byte_id] = NEG_INF

        result = ComboMask(mask=static_mask, byte_guard=byte_guard)
        self._cache[languages] = result
        logger.info("Language filter: built mask for %s", languages)
        return result
