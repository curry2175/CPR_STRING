# v038 Six-Agent Discussion Pilot

v038은 Discussion Lab의 화면, API payload, public Node/Edge/Issue 구조, `schema_version: 0.27.0`을 유지하면서 내부 분석만 6-Agent graph-native pipeline으로 바꿉니다.

## Agents

1. **Compiler** — 원문을 중요한 atomic Node와 candidate Edge로 변환합니다. 오류를 판정하지 않습니다.
2. **Evidence** — Node와 정확한 원문 span의 일치를 검사합니다.
3. **Logic** — Edge가 논리적으로 성립하는지 검사하고 필요한 누락 전제를 후보로 제안합니다.
4. **Assumption** — 누락 전제를 정의·상식·문맥상 지지·불확실·핵심 미지원·순환 가정으로 분류합니다.
5. **Target** — 결론이 문단의 질문, 목표, 비교 또는 endpoint에 답하는지 검사합니다.
6. **Judge** — 같은 graph target에 대한 불확실하거나 충돌하는 finding만 조정합니다.

Python deterministic auditor는 Agent 수에 포함하지 않습니다.

## Missing-premise 처리

Logic Agent가 누락 전제를 발견해도 즉시 Red로 확정하지 않습니다.

- `accepted_definition`, `accepted_background`, `explicitly_supported`, `supported_by_context` → 단순 missing-premise finding 제거 가능
- `plausible_but_unverified` → 조건부·불확실 finding으로 유지
- `unsupported_critical_assumption`, `overly_strong_assumption` → 부당한 가정 issue 유지
- `circular_assumption` → 순환논리 issue
- `irrelevant_assumption` → Logic 제안 기각

## Long-document safeguards

### Per-chunk Compiler budget

기본값:

- 최대 Node: 32
- 최대 Edge: 72
- absolute schema hard cap: Node 64, Edge 160

Node가 예산보다 많을 때 보존 우선순위:

1. 질문 또는 핵심 목적
2. 최종 결론
3. 결론으로 직접 이어지는 중간 주장
4. 가장 중요한 근거
5. 반대 근거와 limitation
6. 핵심 연구설계·분석·대상집단·노출 정의

반복 배경, 예시, 같은 수치의 재진술, 결론 경로와 무관한 세부사항은 생략합니다.

### Document-level budget

여러 chunk를 병합한 뒤 기본적으로:

- 최대 Node: 120
- 최대 Edge: 260

을 적용합니다. 단순 앞부분 자르기가 아니라 역할, edge degree, conclusion ancestor path, 모델의 importance score를 합쳐 중요한 Node를 보존합니다.

## Duplicate Node control

중복 제거는 두 번 수행됩니다.

1. Compiler prompt: 반복·의역·재진술을 하나의 canonical Node로 만들도록 강제
2. Python postprocessor: 원문, normalized meaning, subject–predicate–object, polarity, assertion type, 숫자를 비교해 local 및 cross-chunk 중복을 병합

반대 polarity, 다른 숫자, 다른 assertion type은 자동 병합하지 않습니다. Edge와 Issue의 Node reference는 canonical Node로 remap됩니다.

## Environment variables

```env
DISCUSSION_ARCHITECTURE=graph_native_6agents_pilot
DISCUSSION_JUDGE_ENABLED=1
DISCUSSION_MAX_NODES_PER_CHUNK=32
DISCUSSION_MAX_EDGES_PER_CHUNK=72
DISCUSSION_MAX_DOCUMENT_NODES=120
DISCUSSION_MAX_DOCUMENT_EDGES=260
DISCUSSION_NODE_DEDUP_THRESHOLD=0.96
```

`DISCUSSION_NODE_DEDUP_THRESHOLD`는 0.85–1.0 범위이며 높을수록 더 엄격하게 같은 Node만 병합합니다.

## Compatibility rollback

```env
DISCUSSION_ARCHITECTURE=graph_native_5agents_pilot
```

또는:

```env
DISCUSSION_ARCHITECTURE=graph_native_multi_agent
```

기존 single-pass:

```env
DISCUSSION_ARCHITECTURE=legacy_single_pass
```
