# language-logits-processor

Language-aware logits processor for vLLM. Restricts model output to specified
languages by masking tokens outside the allowed Unicode ranges.

Solves the problem of multilingual models (Gemma, Llama, Qwen, etc.) randomly
inserting characters from wrong scripts — e.g. Thai characters appearing in
Korean output.

## How it works

Two-layer filtering:

1. **Static mask**: scans the tokenizer vocabulary once at startup via
   `cached_tokenizer_from_config` and classifies every token by Unicode range.
   Builds per-language boolean mask tensors on GPU. Tokens outside allowed
   languages get `-inf` logits.
2. **Byte-fallback guard**: prevents the model from assembling banned characters
   one byte at a time via byte-fallback tokens (`<0x00>`..`<0xFF>`). Guard rules
   are automatically derived from the allowed Unicode ranges — no manual
   configuration needed.
3. **Token classification (lenient)**: a token is allowed for a language if
   **any** character in the token belongs to that language's ranges. Common
   characters (digits, punctuation, whitespace, emoji) are always allowed.

## Quick start

```bash
pip install /path/to/language-logits-processor

vllm serve my-model \
    --logits_processors language_logits_processor:LanguageLogitsProcessor
```

## Usage

Pass `output_languages` in `vllm_xargs`:

```python
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "vllm_xargs": {"output_languages": ["ko", "en"]}
    },
)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "messages": [{"role": "user", "content": "Hello"}],
    "vllm_xargs": {"output_languages": ["ko", "en"]}
  }'
```

Requests without `output_languages` are not filtered.

## Supported languages

| Code | Language | Key Unicode ranges |
|------|----------|--------------------|
| `ko` | Korean | Hangul Syllables, Jamo |
| `en` | English | Basic Latin, Latin-1 Supplement, Extended Latin |
| `ja` | Japanese | Hiragana, Katakana, CJK Ideographs (kanji) |
| `zh` | Chinese | CJK Unified Ideographs, Extensions, Bopomofo |
| `th` | Thai | Thai block |
| `vi` | Vietnamese | Latin + Vietnamese diacritics |
| `ar` | Arabic | Arabic, Presentation Forms |
| `ru` | Russian | Cyrillic |
| `de` | German | Latin ranges |
| `fr` | French | Latin ranges |
| `es` | Spanish | Latin ranges |
| `pt` | Portuguese | Latin ranges |

Common characters (digits, punctuation, whitespace, CJK punctuation, emoji,
mathematical symbols, etc.) are always allowed regardless of language selection.

CJK ideographs are NOT in `ko`. Use `["ko", "ja"]` or `["ko", "zh"]`
if you need them.

## Adding a language

Add ranges to `src/language_logits_processor/languages/registry.py`:

```python
# Hindi
LANGUAGES["hi"] = [
    CharRange(0x0900, 0x097F),  # Devanagari
]
```

Byte-fallback guard rules are automatically computed from the ranges.

## Architecture

```
LogitsProcessor (processor.py — only vLLM-dependent file)
    |
    +-- __init__(vllm_config, device, is_pin_memory)
    |     +-- cached_tokenizer_from_config()
    |     +-- MaskCache(tokenizer, vocab_size, device)
    |           +-- classify_vocabulary() -> per-language bool masks on GPU
    |           +-- detect_byte_tokens() -> byte-fallback token IDs
    |           +-- ByteGuardRules() -> UTF-8 lead/second byte rules
    |
    +-- update_state(batch_update)
    |     +-- tracks per-request language combos
    |     +-- processes removes -> adds -> moves
    |     +-- _rebuild_gpu_state() pre-allocates GPU tensors (only on batch changes)
    |
    +-- apply(logits) — operates on [B, V] tensor, no CPU->GPU transfers
          |
          +-- no filtered requests -> immediate return (zero overhead)
          |
          +-- filtered requests (using pre-allocated GPU tensors)
                +-- Layer 1: logits[rows] += static_mask (single GPU vector op)
                +-- Layer 2: byte guard — runs only when a regulated lead byte
                     was just generated (rare in normal text)
```

### File dependency boundaries

```
processor.py          <- vLLM dependency (only file)
masking.py            <- torch only (MaskCache + ByteGuardRules + ComboMask)
classifier.py         <- torch only (token classification + byte token detection)
languages/registry.py <- pure data (Unicode ranges)
```

## Performance

- **Latency overhead: 0%** — no CPU-to-GPU transfers in `apply()`, GPU-only ops
- **Init**: vocabulary scan ~12s for 262K vocab (one-time at startup)
- **Per-step**: single vector addition using pre-allocated GPU tensors — negligible
- **Memory**: one bool tensor per language + cached float masks for used combos
- **Mixed batch**: unfiltered requests return immediately from `apply()` — zero impact

## Requirements

- vLLM >= 0.26.0 (V1 LogitsProcessor interface)
- Python >= 3.10
- torch, transformers

## License

Apache-2.0
