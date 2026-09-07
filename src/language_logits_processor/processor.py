"""
vLLM V1 batch-level LogitsProcessor.

This is the only file that imports from vLLM.
Register via:
    vllm serve ... --logits_processors language_logits_processor:LanguageLogitsProcessor

Performance note:
    apply() runs every decode step. All tensors used in apply() are
    pre-allocated on GPU in update_state() — no CPU→GPU transfers,
    no Python allocations per step. The only CPU work in apply() is
    reading output_ids[-1] for the byte guard, which triggers a
    GPU transfer only when a regulated lead byte was just generated
    (rare in normal text).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

from language_logits_processor.languages import LANGUAGES
from language_logits_processor.masking import ComboMask, MaskCache

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = frozenset(LANGUAGES.keys())


@dataclass
class _RequestState:
    combo: tuple[str, ...]
    output_ids: list[int]


@dataclass
class _ComboGPUState:
    combo_mask: ComboMask
    rows: torch.Tensor
    indices: list[int]
    last_tok_cpu: torch.Tensor
    last_tok_gpu: torch.Tensor
    regulated_ids: set[int] = field(default_factory=set)


class LanguageLogitsProcessor(LogitsProcessor):
    """
    Restricts token generation to specified languages.

    Per-request activation via vllm_xargs:
        {"output_languages": ["ko", "en"]}

    Requests without output_languages pass through unfiltered.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        langs: Any | None = params.extra_args and params.extra_args.get(
            "output_languages"
        )
        if langs is None:
            return
        if not isinstance(langs, list) or len(langs) == 0:
            raise ValueError(
                "output_languages must be a non-empty list of language codes"
            )
        for lang in langs:
            if lang not in SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Unsupported language '{lang}'. "
                    f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
                )

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        vocab_size = vllm_config.model_config.get_vocab_size()

        self._device = device
        self._is_pin_memory = is_pin_memory
        self.mask_cache = MaskCache(tokenizer, vocab_size, device)
        self._requests: dict[int, _RequestState] = {}
        self._active = False
        self._combo_state: list[_ComboGPUState] = []

        logger.info(
            "LanguageLogitsProcessor initialized: vocab_size=%d, "
            "supported_languages=%s",
            vocab_size,
            sorted(SUPPORTED_LANGUAGES),
        )

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        changed = False

        for idx in batch_update.removed:
            if self._requests.pop(idx, None) is not None:
                changed = True

        for idx, params, _prompt_ids, output_ids in batch_update.added:
            langs: Any | None = params.extra_args and params.extra_args.get(
                "output_languages"
            )
            if langs is None:
                continue
            combo = tuple(sorted(langs))
            self._requests[idx] = _RequestState(
                combo=combo, output_ids=output_ids,
            )
            changed = True

        if batch_update.moved:
            moved_states: dict[int, _RequestState] = {}
            for from_idx, to_idx, _direction in batch_update.moved:
                state = self._requests.pop(from_idx, None)
                if state is not None:
                    moved_states[to_idx] = state
                    changed = True
            self._requests.update(moved_states)

        if changed:
            self._rebuild_gpu_state()

    def _rebuild_gpu_state(self) -> None:
        """Pre-allocate all GPU tensors. Called only on batch composition changes."""
        if not self._requests:
            self._active = False
            self._combo_state = []
            return

        self._active = True
        combo_groups: dict[tuple[str, ...], list[int]] = {}
        for idx, state in self._requests.items():
            combo_groups.setdefault(state.combo, []).append(idx)

        self._combo_state = []
        for combo, indices in combo_groups.items():
            cm = self.mask_cache.get(combo)
            n = len(indices)

            rows = torch.tensor(
                indices, dtype=torch.long, device=self._device,
            )

            lt_cpu = torch.zeros(n, dtype=torch.long)
            if self._is_pin_memory:
                lt_cpu = lt_cpu.pin_memory()
            lt_gpu = torch.zeros(n, dtype=torch.long, device=self._device)

            regulated: set[int] = set()
            if cm.byte_guard is not None and cm.byte_guard.has_dynamic_rules:
                regulated = cm.byte_guard._regulated_token_ids

            self._combo_state.append(_ComboGPUState(
                combo_mask=cm,
                rows=rows,
                indices=indices,
                last_tok_cpu=lt_cpu,
                last_tok_gpu=lt_gpu,
                regulated_ids=regulated,
            ))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._active:
            return logits

        for gs in self._combo_state:
            logits[gs.rows] += gs.combo_mask.mask

            bg = gs.combo_mask.byte_guard
            if bg is None or not bg.has_dynamic_rules:
                continue

            need_guard = False
            for i, idx in enumerate(gs.indices):
                oids = self._requests[idx].output_ids
                if oids:
                    last = oids[-1]
                    gs.last_tok_cpu[i] = last
                    if last in gs.regulated_ids:
                        need_guard = True
                else:
                    gs.last_tok_cpu[i] = 0

            if need_guard:
                gs.last_tok_gpu.copy_(gs.last_tok_cpu, non_blocking=True)
                bg.apply(logits, gs.rows, gs.last_tok_gpu)

        return logits
