# 6-Agent Discussion Lab Architecture

## 목적

원문 문제를 단순히 Knowledge Graph로 바꾸는 것이 아니라, **LLM이 생성한 주장과 reasoning output을 Node/Edge로 변환**하고 검증 가능하게 만드는 것이 핵심입니다.

## 실행 흐름

```text
User input / Source evidence / LLM response
             │
             ▼
Balanced Compiler
  ├─ Source Graph
  └─ Response Graph
             │
             ▼
Evidence ─┐
Logic    ─┼─ parallel specialist review
Target   ─┘
             │
             ├─ conditional Assumption review
             └─ conditional Judge review
             │
             ▼
Validated graph + issues + patches + span projection
```

## Agent 역할

### Compiler

- 완전한 proposition 단위 Node 생성
- atomicity와 sentence coverage 검사
- source/response graph를 별도로 구성
- local repair로 누락·fragment 문제를 보완

### Evidence Agent

- Node가 실제 source evidence와 일치하는지 확인
- source mismatch, hallucinated content, modality/scope distortion 탐지

### Logic Agent

- Edge가 논리적으로 타당한지 확인
- unsupported inference, wrong edge, circular reasoning, missing premise 탐지

### Target Agent

- 질문이 요구한 목표를 graph가 충족하는지 확인
- task coverage gap, irrelevant reasoning, target mismatch 탐지

### Assumption Agent

- 전문 Agent가 제기한 missing premise가 안전한 언어적 추론인지 외부지식인지 검토
- 필요한 경우에만 실행

### Judge Agent

- specialist 간 충돌, high-severity defect, graph patch가 있을 때 최종 판단
- patch 선택 및 verdict 확정

## RAGTruth 평가 흐름

전체 정량 평가는 비용과 혼합효과를 줄이기 위해 기본적으로 다음 두 방법을 비교합니다.

```text
Raw Direct
vs
Balanced DualGraph
```

6-Agent 파이프라인은 selected case의 심층 진단과 Discussion Lab에 사용됩니다.
