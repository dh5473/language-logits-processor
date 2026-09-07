# language-logits-processor — 설계 결정 기록

## 목적

Gemma4 등 다국어 모델이 생성 중 의도하지 않은 언어 문자를 혼입하는 문제를
vLLM custom logits processor로 해결한다.

예: 한국어 생성 중 `제가 좀 헷คล렸나 봐요` — 태국어 문자 혼입.

---

## 결정 1: LogitsProcessor — batch-level (확정, 변경)

**선택**: `LogitsProcessor` (batch-level 직접 구현)

**기각**: `AdapterLogitsProcessor` (per-request wrapper)

**경과**: 초기에 AdapterLogitsProcessor를 선택했으나, vLLM 공식 문서에서
"performance가 부족하면 LogitsProcessor로 재구현하라"고 명시하고 있어 전환함.

**이유**:
- AdapterLogitsProcessor.apply()는 내부적으로 Python 루프로 요청을 하나씩 처리
- 마스크 적용(`logits[rows] += mask`)은 배치 전체에 벡터 연산 한 번으로 가능
- update_state()에서 per-request 언어 조합을 추적, apply()에서 같은 조합끼리 묶어 적용
- 다중 언어 조합 지원 유지 — 배치 내 A요청 `["ko","en"]`, B요청 `["ja"]` 공존 가능

---

## 결정 2: 토큰 분류 — 느슨(lenient) 모드 (확정)

**규칙**: 토큰 안의 문자 중 **하나라도** 요청 언어에 포함되면 허용.

**이유**:
- 혼합 스크립트 토큰(극히 드묾)을 차단하면 순수 토큰 조합으로
  같은 텍스트를 생성할 수 있지만, 생성 품질이 떨어질 위험
- 실제 tokenizer vocab에서 cross-script 혼합 토큰은 매우 드묾
- common 문자(숫자, 구두점, 공백, 이모지)는 언어 무관 항상 허용

---

## 결정 3: per-request 파라미터 경로 — vllm_xargs (확정)

**사용법**:
```json
{
  "model": "gemma-4-26b-a4b-it",
  "messages": [...],
  "vllm_xargs": {"output_languages": ["ko", "en"]}
}
```

**이유**: vLLM의 커스텀 파라미터 통로가 `vllm_xargs` → `SamplingParams.extra_args`로
고정됨. `extra_body`에 직접 넣으면 인식 안 됨.

**기본 동작**: `output_languages`가 없으면 필터 없이 통과.

---

## 결정 4: 토크나이저 로딩 — cached_tokenizer_from_config (확정)

```python
from vllm.tokenizers import cached_tokenizer_from_config
tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
```

**기각**: `AutoTokenizer.from_pretrained(model_name)`

**이유**:
- `--tokenizer`가 `--model`과 다를 수 있음
- `tokenizer_mode`, `trust_remote_code` 자동 반영
- `lru_cache`로 엔진과 동일 인스턴스 재사용
- vLLM 메인테이너 공식 답변 (GitHub #29409)

---

## 결정 5: 등록 방식 — --logits_processors CLI (확정)

```bash
vllm serve ... --logits_processors language_logits_processor:LanguageLogitsProcessor
```

**기각**: entry point 자동 발견 (opt-out 불가능 — 설치만으로 항상 로드됨)

---

## 결정 6: 파일 구조 — vLLM 의존성 파일 분리 (확정, 갱신)

```
language-logits-processor/
├── src/language_logits_processor/
│   ├── processor.py       ← vLLM 의존 (유일)
│   ├── masking.py         ← torch만 (마스크 캐시 + byte guard)
│   ├── classifier.py      ← torch만 (토큰→언어 분류 + byte 토큰 탐지)
│   └── languages/
│       └── registry.py    ← 순수 데이터 (Unicode 범위)
```

**이유**: vLLM import는 processor.py에만 격리.
테스트는 vLLM 없이 classifier + masking만으로 가능 (111건).

---

## 결정 7: is_argmax_invariant = False (확정)

마스크가 최고 logit 토큰(예: 태국어)을 -inf로 바꿀 수 있으므로
argmax가 변할 수 있다. False 선언 필수.

---

## 결정 8: ko에서 한자(CJK) 제거 (확정)

**변경**: ko 범위에서 CJK Unified Ideographs(U+4E00-U+9FFF),
CJK Extension A(U+3400-U+4DBF) 제거.

**이유**:
- 한국어 챗 서비스에서 한자 출력은 불필요
- 한자를 ko에 포함하면 중국어 전용 토큰도 통과하게 됨
- 모델이 `北京` 대신 `베이징`으로 출력 — 원하는 동작
- 한자가 필요한 경우 `["ko", "ja"]` 또는 `["ko", "zh"]` 조합으로 요청

---

## 결정 9: 2계층 필터링 — byte-fallback 가드 (확정)

**구조**:
1. **정적 마스크**: 디코딩된 텍스트 기반으로 토큰 분류. 허용 언어 밖 토큰을 -inf.
2. **byte-fallback 가드**: byte 토큰(`<0x00>`~`<0xFF>`)으로 차단된 문자를
   byte 단위로 조립하는 우회 경로를 차단.

**byte guard 동작**:
- 허용된 Unicode 범위에서 UTF-8 인코딩 규칙을 역산하여 자동 계산
- 정적 차단: 차단 영역만 시작하는 선두 바이트를 static mask에 편입
  (ko+en: 0xE0=태국어, 0xE4-0xE9=CJK, 0xD0-0xD3=키릴 등)
- 동적 차단: 허용/차단 혼재 선두 바이트(0xE3, 0xEF, 0xF0 등)는
  직전 토큰 기반으로 두 번째 바이트를 조건부 차단
- UTF-8 overlong encoding도 처리 (0xE0 최소 second=0xA0 등)

**기각**: byte 토큰 전면 차단 — 희귀 한글 음절(뷁, 쒫)이 byte로 조립되므로 불가.

**장점**: 언어 조합에서 자동 산출되므로 언어 추가 시 별도 작업 불필요.

---

## 결정 10: 런타임 vocab 스캔 (확정)

**선택**: 서버 기동 시 tokenizer vocab 전체를 decode+분류하여 마스크 생성.

**기각**: 오프라인 빌드(`build_script_mask.py` → JSON)

**이유**:
- 모델/토크나이저 변경 시 재빌드 불필요 — 자동 적응
- 다중 언어 조합을 위한 per-language 마스크 생성에 자연스러움
- 배포 절차 단순 — 패키지 설치만으로 동작
- 기동 시 수초 추가지만 모델 로딩(수십 초~분) 대비 무시할 수준

**서빙 성능**: 기동 이후 apply()에서 사용하는 마스크 텐서는
오프라인 방식과 동일한 shape/dtype/연산. per-step 비용 차이 없음.

---

## 결정 11: super().__init__() 호출 금지 (확정)

vLLM 0.28에서 `LogitsProcessor.__init__`이 `NotImplementedError`를 raise한다.
서브클래스에서 `super().__init__()` 호출하면 안 됨.

**발견**: RunPod end-to-end 테스트에서 크래시로 확인 (2026-09-07).

---

## 검증 완료 (2026-09-07, RunPod L40S + vLLM 0.28.0)

### Phase 2: Gemma4 vocab 전수 분류 (tokenizer만, 추론 없음)

모델: `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`, vocab 262,144

| 분류 | 토큰 수 |
|------|--------:|
| ko | 4,678 |
| en | 158,450 |
| ja | 24,172 |
| zh | 20,614 |
| th | 2,177 |
| ru | 13,398 |
| ar | 8,457 |
| common | 6,856 |
| special | 374 |
| unknown (필터 시 차단) | 43,265 |
| byte-fallback | 256 |
| **ko+en banned** | **91,786 (35.0%)** |

대겸 분석 참고: ban 90,031 (34.3%). 차이 1,755는 ko에서 한자 제거 + 분류 방식 차이.

Spot check 전부 정상:
- 안녕하세요, Hello, ㅋㅋㅋ, 123, 뷁, 베이징 → allowed
- 谢谢, สวัสดี, こんにちは, 韓國, Привет, مرحبا → BLOCKED

Byte guard (ko+en): 정적 차단 34 lead bytes, 동적 8 rules.

### Phase 3: End-to-end 서빙 테스트

- `--logits_processors language_logits_processor:LanguageLogitsProcessor` 정상 로드
- T1 baseline (필터 없음): 정상 한국어 응답
- T2 ko+en 필터: 정상 한국어, 외국 문자 없음
- T3 버그 케이스 x5 (temp=1.0): **5/5 CLEAN** — 태국어/CJK/키릴 혼입 0건
- T4 ja+en 필터: 일본어+한자+영어 정상 출력
- `vllm_xargs` 경로 동작 확인
- per-request 다중 언어 조합 동작 확인

---

## 결정 12: apply() 최적화 — GPU 텐서 사전 할당 (확정)

**문제**: 최적화 전 apply()에서 매 디코드 스텝마다 CPU→GPU 텐서 할당+전송 발생.
latency overhead +22%.

**원인**:
- `torch.tensor(indices, device=cuda)` — 매 스텝 GPU 텐서 할당 + CPU→GPU 전송
- `torch.tensor(last_tokens, device=cuda)` — byte guard용 동일
- Python dict(combo_groups) 매 스텝 재구성

**해결**:
1. `update_state()`에서 GPU 텐서(`rows`, `last_tok_gpu`) 사전 할당 — 배치 변경 시에만 재생성
2. byte guard CPU 버퍼를 pinned memory로 사전 할당 → `non_blocking=True` copy
3. byte guard 실행 전 CPU 측 `regulated_token_ids` set 확인 → lead byte 아니면 GPU 전송 자체 스킵
4. `changed` 플래그로 배치 변경 없으면 `_rebuild_gpu_state()` 스킵

**결과**: overhead +22% → **0%** (측정 오차 범위 내)

---

## vLLM 0.26 호환성 확인 (2026-09-07)

0.26.0 소스 확인 결과, 코드 변경 없이 호환:
- `LogitsProcessor` 인터페이스 (init/apply/update_state/is_argmax_invariant/validate_params) — 동일
- `BatchUpdate` 필드 (batch_size, removed, added, moved) — 동일
- `__init__`의 `NotImplementedError` — 동일 (super() 호출 금지 동일)
- `cached_tokenizer_from_config` import 경로 — 동일
- `SamplingParams.extra_args` + `vllm_xargs` — 동일
- `--logits_processors` CLI — 동일

---

## 미결

- [ ] 프로덕션 배포 (Elice B200, vLLM 0.26)
- [ ] 공통(common) 범위 미세 조정 — 실제 차단되는 토큰 중 문제 있는 것 검토

---

## 참고 자료

- [vLLM Custom Logits Processors](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)
- [vLLM Logits Processors Design](https://docs.vllm.ai/en/stable/design/logits_processors/)
- [예제: batch-level](https://github.com/vllm-project/vllm/blob/main/examples/features/logits_processor/custom.py)
- [예제: per-request adapter](https://github.com/vllm-project/vllm/blob/main/examples/features/logits_processor/custom_req.py)
- [예제: adapter with init](https://github.com/vllm-project/vllm/blob/main/examples/features/logits_processor/custom_req_init.py)
- [토크나이저 로딩 방법](https://github.com/vllm-project/vllm/issues/29409)
- [대겸 분석: Gemma4 스크립트 마스크](docs/gemma4_script_mask/script_constrained_sampling_gemma4.md)
