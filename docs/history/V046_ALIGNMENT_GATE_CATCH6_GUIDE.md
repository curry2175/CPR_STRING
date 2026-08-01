# v046 실행 가이드

## 1. 최초 설정

```bat
cd /d C:\vrg\v046
setup.bat
```

`.env`에 OpenAI API key를 입력합니다.

## 2. 전체 RAGTruth QA 정량실험

```bat
cd /d C:\vrg\v046
run_ragqa.bat
```

비교:

- Raw Direct
- v043 Balanced Dual-Graph + v046 conservative Alignment gate

전체 평가 중에는 six-agent catch review를 실행하지 않습니다. Raw가 놓치고 DualGraph가 잡은 후보 정보만 저장합니다.

Catch 저장 위치:

```text
outputs\ragtruth_raw_vs_dual_graph_nano\<run_id>\catch_candidates\
```

기존 v045 cache를 재사용하려면 이전 폴더의 다음 파일을 같은 상대경로로 복사합니다.

```text
outputs\ragtruth_raw_vs_dual_graph_nano\generation_cache_v040.json
```

Raw, Source Graph, Response Graph cache는 재사용되고, prompt version이 바뀐 Alignment만 다시 실행됩니다.

## 3. Catch 후보 중 하나를 6-Agent로 분석

```bat
cd /d C:\vrg\v046
run_catch.bat
```

화면에 저장된 후보가 표시됩니다.

```text
[1] case_id · raw_detection_miss_dual_hit · Raw F1=... · Dual F1=...
[2] case_id · dual_material_uplift · Raw F1=... · Dual F1=...
```

번호 또는 case ID를 입력합니다. 선택한 한 case에만 Source/Response six-agent Graph 검증과 Cross-Graph 분석이 실행됩니다.

결과:

```text
outputs\ragtruth_catch_6agent_reviews\catch6_<case_id>_<timestamp>\
├─ report.html
├─ result.json
└─ README.txt
```

`report.html`에는 다음이 표시됩니다.

- Source 원문과 Response
- Gold / Raw / Balanced prediction span
- Balanced Response Graph의 각 Claim Node
- 각 Node가 gold 또는 DualGraph prediction과 겹치는지
- 매칭된 Source Node
- 기존 Balanced alignment verdict
- 선택 실행한 six-agent verdict
- Source/Response Agent findings와 승인 patch
- 최종 validated Source/Response Graph

## 4. Discussion Hub

```bat
cd /d C:\vrg\v046
run_hub.bat
```

브라우저:

```text
http://127.0.0.1:8765/discussion-lab
```

Discussion Hub는 Balanced Compiler 뒤에 Evidence, Logic, Target, 조건부 Assumption, 조건부 Judge가 모두 유지됩니다.
