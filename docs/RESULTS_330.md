# RAGTruth QA — 330-Case Result Snapshot

## 데이터

- Dataset: RAGTruth
- Task: QA
- 완료: 330 / 875
- Clean: 263
- Hallucinated: 67
- Model: `gpt-5.4-nano`

## Character-level 성능

| Model | Character micro-F1 |
|---|---:|
| Raw Direct | 37.42% |
| Original DualGraph v046 | 35.55% |
| Best single global threshold | 42.30% |
| **Best relation-specific threshold** | **47.59%** |

Optimized DualGraph의 Raw 대비 차이는 **+10.17 percentage points**입니다.

## 최적 Gate

| Relation | Threshold |
|---|---:|
| `contradicted_by` | 0.99 |
| `partially_supported_by` | 0.84 |
| `qualified_by` | 0.89 |
| `not_found_in_source` | 0.42 |
| `requires_assumption` | 0.66 |

- Span mode: `core`
- Infer missing error label: `false`
- Search configurations: 24,410
- API calls during optimization: 0

## Response-level confusion matrices

### Raw Direct

| Actual \ Predicted | Hallucinated | Clean |
|---|---:|---:|
| Hallucinated | TP 66 | FN 1 |
| Clean | FP 133 | TN 130 |

- Sensitivity: 98.51%
- Specificity: 49.43%
- Response-level F1: 49.62%

### Original DualGraph v046

| Actual \ Predicted | Hallucinated | Clean |
|---|---:|---:|
| Hallucinated | TP 50 | FN 17 |
| Clean | FP 53 | TN 210 |

- Sensitivity: 74.63%
- Specificity: 79.85%
- Response-level F1: 58.82%

### Threshold-optimized DualGraph

| Actual \ Predicted | Hallucinated | Clean |
|---|---:|---:|
| Hallucinated | TP 44 | FN 23 |
| Clean | FP 20 | TN 243 |

- Sensitivity: 65.67%
- Specificity: 92.40%
- Response-level F1: 67.18%
- Accuracy: 86.97%
- Clean false-positive rate: 7.60%

## Rescue 해석

### Raw가 clean에 문제를 표시했지만 Graph가 문제없다고 한 사례

원래 실행에서 대표적으로 다음 case가 확인되었습니다.

- 15458
- 14282
- 13039
- 12133
- 15102
- 14760
- 13911
- 15464
- 14512
- 17235
- 16357

### Raw가 hallucination을 완전히 놓치고 Graph만 찾은 사례

- **Strict rescue: 0개**

따라서 현재 결과로 “Graph가 Raw가 완전히 못 본 hallucination을 새로 발견했다”고 주장하면 안 됩니다.

### Localization uplift 대표 사례

- **14806**: weather 값의 일부 탐지에서 거의 완전한 unsupported proposition으로 확장
- **14396**: `water and electrolyte loss` 조각에서 더 완전한 causal clause로 확장
- **15556**: unsupported travel recommendation에 대한 recall 개선

## 포함된 결과 파일

- `results/ragtruth_qa_330_console.log` — 실행 원본 콘솔 로그
- `results/ragtruth_qa_330_case_metrics.csv` — 330개 case별 파싱 결과
- `results/response_level_confusion_matrices.csv`
- `results/representative_cases.csv`
- `results/summary_330.json`
- `results/threshold_optimization_console.txt`

완전한 local `cases.jsonl`은 `scripts/IMPORT_LOCAL_RUN_WINDOWS.bat`으로 원래 PC에서 가져옵니다.
