# Reproducibility Guide

## 환경 구축

```bat
setup.bat
```

`.env`를 만들고 OpenAI API key를 설정합니다.

```text
OPENAI_API_KEY=...
```

`.env`는 Git에서 제외됩니다.

## Discussion Lab

```bat
run_hub.bat
```

```text
http://127.0.0.1:8765/discussion-lab
```

## Direct case review

```bat
run_case.bat 14806
```

출력에는 Source Graph, Response Graph, cross comparison, 6-Agent verdict, Gold overlap, projected span을 포함한 interactive HTML이 생성됩니다.

## RAGTruth 평가

```bat
run_ragqa.bat
```

## 330-case threshold 재현

원래 local files를 가져온 후:

```bat
scripts\IMPORT_LOCAL_RUN_WINDOWS.bat
run_threshold_optimizer.bat
```

최적화 없이 preset을 사용하려면 `config/optimized_gate_330.json`을 사용합니다.

## 정확한 rescue report

```bat
run_rescue_report.bat
```

이 명령은 original `cases.jsonl`과 optimizer가 만든 `best_reprojected_cases.jsonl`을 case ID로 비교하여 다음을 출력합니다.

- strict hallucination rescue
- clean rescue
- hallucination regression
- clean regression
