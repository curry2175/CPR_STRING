# Impact Factor × Discussion 논리 구조 — 실행 계획

팀 CPR · YAI 해커톤 · 2026-08-01
담당 정리: 최현민

---

## 0. 한 줄 요약

같은 약, 같은 시기, 같은 결론("효과 없음")을 낸 COVID-19 RCT 들을 모아서,
**저널 IF 만 다르고 나머지는 최대한 같게** 만든 다음,
Discussion 이 그 결론에 도달하는 **논증 구조**를 Typed Claim Graph 로 정량화하고
IF 와의 관계를 본다.

우리가 이미 만든 Discussion Lab 이 측정 도구다. 새로 만들 것은 **수집기와 통계 레이어**뿐이다.

---

## 1. 왜 이 질문이 우리 프로젝트에 맞는가

v028 까지 우리는 "우리 Graph 가 Direct GPT 보다 오류를 잘 잡는가"를 물었다.
그건 **도구의 성능** 질문이다. 그것만으로는 prompt engineering 범주를 못 벗어난다는
지적(7/31 승현 형)이 정확했다.

IF × 논리구조는 **그 도구를 써서 세상에 대한 사실을 하나 밝히는** 질문이다.
도구가 novelty 를 만드는 게 아니라, 도구로 얻은 결과가 novelty 를 만든다.

부수 효과가 하나 더 있다. 이 분석이 성립하려면 우리 Graph 지표가
**논문 간 차이를 실제로 구별해야** 한다. 즉 이 연구는 우리 측정도구의
construct validity 를 시험하는 실험이기도 하다. 지표가 전부 비슷하게 나오면
그건 IF 와 무관하다는 결론이 아니라 **우리 지표가 둔하다는 발견**이다.
둘 다 보고 가치가 있다.

---

## 2. 데이터 설계

### 2.1 코어셋 (수동 · 이미 완료)

현민이 고른 5편. HCQ RCT, IF 84.5 → 2.6 으로 2 자릿수 스팬.

| Tier | 저널 | JIF | N | 1차 평가변수 | Discussion 단어수 |
|---|---|---:|---:|---|---:|
| 1 | NEJM (RECOVERY) | 84.5 | 4,716 | 28일 사망률 | 822 |
| 2 | The BMJ (Tang) | 55.1 | 150 | 28일 음전 (surrogate) | 1,226 |
| 3 | CMI (HYCOVID) | 8.7 | 250 | 14일 사망/삽관 | 788 |
| 4 | OFID (TEACH) | 3.8 | 128 | 14일 중증진행 | 739 |
| 5 | IDR/MDPI (Beltran) | 2.6 | 106 | 재원기간·악화 | 543 |

`corpus/hcq_discussions.jsonl` 에 원문 그대로 + 메타데이터로 들어가 있다.

이 5편의 역할은 **gold anchor** 다. 손으로 검증한 것이라, 자동 수집분에서
파싱 오류가 나면 이 5편과 대조해서 잡는다. 통계의 주력이 아니다.

### 2.2 확장셋 (자동 · 이번에 만든 것)

**N=5 로는 상관계수를 못 찍는다.** 점이 더 필요하다.

```
studysets/hcq_covid.json            HCQ/CQ RCT 만          예상 40~80편
studysets/covid_therapeutics_rct.json  COVID 약물 RCT 전반   예상 150~300편
```

먼저 좁은 셋으로 파이프라인을 검증하고, 부족하면 넓힌다.
넓히는 순간 '약물'이 새 교란변수가 되므로 층화가 필수다(§4).

---

## 3. 파이프라인

**중요: 어디서 도는지가 단계마다 다르다.**
이 클라우드 샌드박스는 NCBI/EuropePMC/Crossref 접근이 차단돼 있다(pypi 만 열림).
수집과 분석은 현민 PC 에서 돈다.

```
┌─ 현민 PC (인터넷 O, OpenAI 키 O) ────────────────────────────┐
│                                                              │
│  1_COLLECT.bat                                    무료       │
│    collect_discussions.py                                    │
│      Europe PMC search  →  OA 논문 목록                       │
│      fullTextXML        →  JATS XML                          │
│      extract_discussion →  Discussion 섹션만 (xref/표/그림 제거)│
│      journals.csv       →  저널 IF 매칭                       │
│    journal_frequency.py →  IF 를 채워야 할 저널 목록           │
│                          ↓ corpus/collected.jsonl            │
│                                                              │
│  2_ANALYZE.bat                                    유료       │
│    analyze_batch.py                                          │
│      modules/discussion_lab 의 generate_discussion_graph 호출 │
│      논문 1편 = 1 kg.json                                     │
│      resume 지원 (죽어도 이어서)                               │
│                          ↓ out/kg/*.json                     │
│                          ↓ out/metrics.csv  ← 평평한 분석표    │
└──────────────────────────────────────────────────────────────┘
                           ↓  metrics.csv 를 채팅에 올리면
┌─ 클라우드 (Claude) ──────────────────────────────────────────┐
│  3. 통계 · 시각화 · 리포트                                     │
│     Spearman / 편상관 / 층화 / 부트스트랩 CI                   │
│     산점도 · tier 별 프로파일 · 발표용 그림                     │
└──────────────────────────────────────────────────────────────┘
```

### 3.1 왜 IF 표를 먼저 안 만드는가

순서가 중요하다. 저널 IF 표를 미리 만들려 하면 200개 저널을 찾게 된다.
반대로 해야 한다.

1. IF 없이 전량 수집
2. 저널별 논문 수를 센다
3. **상위 20~30개 저널만** IF 를 채운다 → 보통 논문의 70~80% 커버
4. 나머지는 `jif=null` 로 남기고 분석에서 제외

`journal_frequency.py` 가 이 순서를 강제한다. `journals_todo.csv` 를 뱉는다.

### 3.2 IF 값은 지어내지 않는다

`journals.csv` 에는 현민이 확인한 5개 저널 값만 들어 있다.
나머지는 사람이 JCR 또는 SJR 에서 직접 확인해 채운다.
`metric_source` 열에 출처(JCR2025 / SJR2024 등)를 반드시 적는다.

우리가 논문의 ungrounded claim 을 잡는 도구를 만들면서
우리 데이터에 ungrounded number 를 넣을 수는 없다.

---

## 4. 교란변수 — 이 연구의 승부처

IF 와 상관되면서 논리구조에도 영향을 주는 변수들이다.
하나씩 어떻게 다룰지 미리 정한다. 결과 보고 정하면 그건 p-hacking 이다.

| 교란 | 왜 문제인가 | 처리 |
|---|---|---|
| **Discussion 길이** | 길면 노드·이슈가 자동으로 많아진다. 고IF 저널은 분량 제한이 다르다 | 모든 카운트를 **per 1,000 words** 로 정규화. 원자값과 병기 |
| **저널 서식 정책** | BMJ 는 "Strengths and limitations of study" 소제목을 **의무화**한다. limitation 노드가 많은 게 저자의 신중함인지 서식 강제인지 구별 불가 | `has_explicit_limitation_heading` 을 공변량으로. 있는 저널/없는 저널 층화 |
| **표본 크기·검정력** | 작은 연구일수록 한계를 많이 쓴다. 그리고 작은 연구는 저IF 로 간다 | `n_randomized`, `powered`, `early_termination` 을 공변량. 편상관 |
| **연구 디자인** | 이중맹검 placebo vs open-label. 디자인이 논증 구조를 규정한다 | 스터디셋을 RCT 로 고정. 그 안에서 blinding 을 공변량 |
| **평가변수 종류** | surrogate(음전) vs hard(사망)는 결론 도달 경로가 다르다 | `endpoint_type` 층화. surrogate→clinical 과잉해석은 별도 지표로 |
| **약물 (확장셋)** | 약마다 서사가 다르다 (HCQ 는 정치화됨) | 약물별 층화. Simpson's paradox 점검 |
| **출판 연도** | 2020년 초와 2022년의 서술 규범이 다르다 | 연도 공변량. 가능하면 좁은 창 |
| **OA 선택편향** ⚠ | **가장 위험하다.** 고IF 저널일수록 PMC OA 전문이 없다(NEJM). 자동수집만 쓰면 고IF 쪽이 통째로 빠진다 = 독립변수와 상관된 결측 | `missing_oa.csv` 에 전량 기록. 고IF 결측분은 **손으로 채운다**. `text_source` 로 구분해 민감도 분석 |

마지막 항목이 진짜 위험이다. 결측이 무작위가 아니라 **IF 가 높을수록 잘 빠진다**.
이걸 무시하면 우리가 만든 도구가 잡아야 할 바로 그 오류(informative missingness)를
우리가 저지르게 된다. `missing_oa.csv` 를 반드시 보고에 포함한다.

---

## 5. 측정 — 가설과 지표의 대응

현민이 고른 네 축이다. 각각 어떤 방향을 예상하는지 **미리** 적는다.

### A. 한계 인정 밀도 / hedging

| 지표 | 출처 |
|---|---|
| `limitation_per_1k_words` | role=limitation 노드 |
| `limitation_to_conclusion_ratio` | graph_metrics |
| `hedging_ratio` | certainty ∈ {may, suggests, likely, uncertain} / (그것 + {establishes, proves, concludes}) |

**가설 A1 (통념)**: IF↑ → 한계 인정↑, hedging↑ (고IF 저널이 더 신중하다)
**가설 A2 (반대)**: IF↑ → hedging↓ (강한 주장이라야 고IF 에 실린다)

둘 다 그럴듯하다. 이게 이 연구가 재미있는 이유다.
코어셋만 보면 이미 A1 을 흔드는 신호가 있다 — NEJM(822 words)보다
BMJ(1,226 words)가 한계를 훨씬 길게 쓴다. 다만 BMJ 는 서식 의무가 있다(§4).

### B. Grounding / unsupported conclusion

| 지표 | 출처 |
|---|---|
| `grounded_edge_ratio` | 근거→결론 엣지 중 신뢰도 0.65 이상 비율 |
| `conclusions_with_issue_ratio` | 이슈가 달린 결론 비율 |
| `issue_rule_confirmed_unsupported_n` | 구조적으로 지지 안 되는 결론 |
| `evidence_to_conclusion_ratio` | 결론 1개당 근거 노드 수 |

**가설 B**: IF↑ → grounding↑, unsupported↓
조나현이 정리한 측정치("근거 없이 살아남은 주장 비율")가 여기 그대로 들어간다.

### C. 인과 과잉주장

| 지표 | 출처 |
|---|---|
| `causal_to_association_ratio` | assertion_type causal / association |
| `issue_causal_overclaim_n` | IssueType |
| `issue_surrogate_to_clinical_overreach_n` | IssueType |
| `issue_scope_overreach_n`, `issue_unsupported_generalization_n` | IssueType |

**가설 C**: IF↑ → 과잉주장↓
BMJ(음전 = surrogate) 논문에서 surrogate→clinical 과잉해석이 잡히는지가 흥미롭다.
Discussion 을 읽어보면 오히려 스스로 "applicable only to..." 라고 범위를 좁힌다.
도구가 그걸 정확히 인식하는지가 곧 도구 검증이다.

### D. 구조 복잡도 (기술 통계로만)

`maximum_depth`, `maximum_width`, `mean_branching_factor`, `density`, `node_count`

**v028 문서가 명시적으로 경고한 대로, 복잡도 자체는 우수성 지표가 아니다.**
"고IF 논문의 논증이 더 깊다"를 "더 좋다"로 옮기면 안 된다.
탐색적으로만 보고하고, 주요 결론에 쓰지 않는다.

---

## 6. 통계

- 주 분석: **Spearman 순위상관** (IF는 극단적으로 오른쪽 꼬리가 길다. 84.5 vs 2.6)
- 보조: log(IF) 로 Pearson
- 교란 통제: **편상관** — 단어수, n_randomized, blinding, 연도, 서식 의무 여부
- 층화: endpoint_type / 약물 / OA 여부
- 불확실성: 부트스트랩 95% CI. N 이 작으므로 점추정만 보고하지 않는다
- 다중검정: 지표가 20개 넘는다. **주요 지표 4개를 미리 지정**하고(A: hedging_ratio,
  B: grounded_edge_ratio, C: causal_to_association_ratio, D: nodes_per_1k_words)
  나머지는 탐색으로 명시한다

### N 별로 할 수 있는 말

| N | 가능한 주장 |
|---:|---|
| 5 | 통계 불가. **case series**. "이런 차이가 관찰된다" + 파이프라인 실증 |
| 20~30 | Spearman 가능하나 CI 가 매우 넓다. 방향성 제시 수준 |
| 50~80 | 편상관·층화 가능. 해커톤 발표로는 충분 |
| 150+ | 다변량 회귀. 논문화 사정권 |

목표는 **50~80**. 오늘 안에 도달 가능한 현실선이다.

---

## 7. 타당성 위협 — 발표에서 먼저 말할 것

심사위원이 물어보기 전에 우리가 먼저 꺼낸다. 그게 이 팀의 색깔이다.

1. **생태학적 오류**: IF 는 저널 속성, 논리구조는 논문 속성이다.
   "고IF 저널에 실린 논문"과 "좋은 논문"은 다르다. 개별 논문 품질 추론 금지.
2. **인과 아님**: IF 가 논증을 좋게 만드는 게 아니다. 역방향(좋은 논증이 고IF 로 간다),
   공통원인(연구비·팀 규모·통계 지원)이 모두 가능하다. 상관으로만 보고한다.
3. **측정도구 미검증**: 우리 Graph 지표는 아직 human annotation 과 대조된 적이 없다.
   v028 의 discussion benchmark 는 **개발용 synthetic set** 이다.
   → 최소한 코어셋 5편은 팀원 2명이 blind 로 라벨링해 도구와 대조하자.
4. **LLM 판정의 비결정성**: 같은 문단을 두 번 넣으면 그래프가 달라질 수 있다.
   → 코어셋 5편을 3회 반복 실행해 **지표 신뢰도(ICC)** 를 낸다. 이게 없으면
   상관계수의 분모를 모르는 셈이다.
5. **OA 선택편향**: §4 마지막 항목. 가장 위험.
6. **JIF 자체의 문제**: 분야 간 비교 불가, 리뷰 논문에 좌우됨.
   단일 분야로 고정해 완화한다.

---

## 8. 오늘 순서

| # | 할 일 | 누가 | 시간 | 비용 |
|---|---|---|---|---|
| 1 | 코어셋 5편으로 `2_ANALYZE.bat` — 파이프라인 실증 | 현민 | 5분 | ~5회 호출 |
| 2 | metrics.csv 확인 → 지표가 5편을 구별하는가 | 팀 | 10분 | - |
| 3 | 코어셋 3회 반복 → 신뢰도 확인 | 현민 | 10분 | ~15회 |
| 4 | `1_COLLECT.bat` — 전량 수집 | 현민 | 20분 | 무료 |
| 5 | journals_todo.csv 상위 30개 IF 채우기 | 나현·승현 | 30분 | - |
| 6 | `2_ANALYZE.bat` 전량 | 현민 | 1~2시간 | N회 호출 |
| 7 | metrics.csv 올리면 통계·그림·리포트 | Claude | 30분 | - |
| 8 | 발표 자료 | 팀 | - | - |

**1~3 이 4~6 보다 먼저다.** 지표가 5편도 구별 못 하면 100편 모아도 소용없다.
돈 쓰기 전에 그걸 먼저 확인한다.

---

## 9. 결과가 어떻게 나오든 할 말이 있다

| 결과 | 우리가 하는 말 |
|---|---|
| 뚜렷한 상관 | "Discussion 논증 구조는 저널 tier 에 따라 체계적으로 다르다. 자동 측정 가능하다." |
| 상관 없음 | "논증 구조는 IF 로 설명되지 않는다. 즉 **IF 는 논증 품질의 대리지표가 아니다** — 이건 그 자체로 유용한 반증이다." |
| 지표가 논문을 구별 못 함 | "현재 지표는 실제 논문 간 변별력이 부족하다. 무엇을 고쳐야 하는지 확인했다." — 도구 개발 관점의 발견 |

세 번째 경우에도 발표할 게 있다. 이게 이 설계의 장점이다.

---

## 10. 파일

```
lab/ifxlogic/
├─ PLAN.md                     이 문서
├─ 1_COLLECT.bat               수집 (무료, 인터넷 필요)
├─ 2_ANALYZE.bat               분석 (유료, OpenAI 키 필요)
├─ collect_discussions.py      Europe PMC → Discussion 섹션
├─ journal_frequency.py        IF 채울 저널 우선순위
├─ analyze_batch.py            Discussion Lab 배치 + metrics.csv
├─ journals.csv                저널 → IF (사람이 채운다)
├─ studysets/
│   ├─ hcq_covid.json          좁은 셋
│   └─ covid_therapeutics_rct.json  넓은 셋
├─ corpus/
│   ├─ hcq_discussions.jsonl   코어셋 5편 (완료)
│   ├─ build_corpus.py         코어셋 생성 스크립트
│   └─ collected.jsonl         자동 수집분 (1_COLLECT 후)
└─ out/
    ├─ kg/*.json               논문별 Typed Claim Graph
    └─ metrics.csv             분석용 표
```

`verified-reasoning-graph/` 안은 건드리지 않았다. UPDATE.bat 이 덮어쓰는 영역이다.
