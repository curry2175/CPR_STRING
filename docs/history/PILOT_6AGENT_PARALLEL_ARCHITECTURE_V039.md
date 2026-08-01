# v039 Six-Agent Parallel Discussion Architecture

v039는 v038과 같은 6개 Agent를 사용합니다.

1. Compiler
2. Evidence
3. Logic
4. Assumption
5. Target
6. Judge

Python Global Auditor와 output adapter는 Agent 수에 포함하지 않습니다.

## 실행 단계

### 1. Compiler

원문을 capped, deduplicated candidate graph로 변환합니다. 오류를 판정하지 않습니다.

기본 제한:

```text
chunk당 Node 32 / Edge 72
문서 전체 Node 120 / Edge 260
```

### 2. Evidence · Logic · Target 병렬 실행

세 Agent는 Compiler가 만든 동일한 read-only Shared Graph를 동시에 검토합니다.

```text
                 ┌─ Evidence: Node ↔ source span
Compiler Graph ──┼─ Logic: candidate Edge validity
                 └─ Target: conclusion ↔ objective/endpoint
```

API 호출 수는 그대로 3회지만 순차 대기 대신 동시에 진행하므로 wall-clock latency를 줄입니다.

### 3. 조건부 Assumption

Logic이 명시적인 missing-premise 후보를 출력한 경우에만 호출합니다. 단순한 불확실성이나 약한 inference만으로는 호출하지 않습니다.

```text
Logic missing_node_proposal 없음 → 호출 0회
Edge에 연결된 missing_node_proposal 있음 → Assumption 호출
```

Assumption은 누락 전제를 정의적·상식적 배경지식, 문맥상 지지, plausible but unverified, critical unsupported, overly strong, circular 등으로 분류합니다.

### 4. 조건부 Judge

Judge 조건은 너무 엄격한 hard conflict 전용도 아니고, 모든 저신뢰 finding에 호출되는 것도 아닙니다.

호출 가능 조건:

- medium/high finding이 아직 `uncertain`
- 같은 Node/Edge에 여러 상이한 판정이 겹침
- Edge type 변경 또는 claim qualification 제안
- conclusion에 직접 영향을 주는 medium/high finding의 confidence가 기준 이하

기본 confidence 기준은 `0.65`이며 `.env`에서 조절할 수 있습니다.

```env
DISCUSSION_JUDGE_CONFIDENCE_THRESHOLD=0.65
```

## 변경하지 않은 것

- Discussion Lab 화면
- API payload와 endpoint
- public Node/Edge/Issue schema
- `schema_version: 0.27.0`
- 긴 문서 chunking
- Node 제한과 중복 병합
- ProcessBench, CLUTRR, RAGTruth runner
- RAGQA 전용 Source Graph / Response Graph / cross-graph alignment
