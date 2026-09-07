# SPDX-License-Identifier: Apache-2.0

def __getattr__(name: str):
    if name == "LanguageLogitsProcessor":
        from language_logits_processor.processor import LanguageLogitsProcessor
        return LanguageLogitsProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LanguageLogitsProcessor"]
