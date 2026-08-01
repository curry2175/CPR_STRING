# v047 Direct Case ID 6-Agent Guide

## One RAGTruth ID only

```bat
cd /d C:\vrg\v047
run_case.bat 12046
```

The full `run_ragqa.bat` benchmark does **not** need to finish first.

You may also double-click `run_case.bat` without an argument. It will ask for the ID.

## What runs

1. The requested RAGTruth QA record is loaded directly by ID.
2. The v043 task-complete Source Compiler creates the Source Graph.
3. The v043 Balanced Complete-Proposition Compiler creates the Response Graph and performs local node repair when needed.
4. The v046 conservative factuality-gated Alignment compares Source and Response.
5. The Source Graph and Response Graph are independently reviewed by the full graph agents:
   - Evidence
   - Logic
   - Target
   - conditional Assumption
   - conditional Judge
6. The validated graphs are compared by Cross-Graph Evidence Matcher, Logic, conditional Assumption/Judge, and Span Projector.

## Output

```text
outputs\ragtruth_case_6agent_reviews\catch6_<CASE_ID>_<TIMESTAMP>\
├─ report.html
├─ result.json
└─ README.txt
```

Open `report.html`. It includes:

- Source evidence
- Response and gold spans
- Balanced Source Graph
- Balanced Response Graph
- v046 alignment
- Source and Response six-agent findings
- Node-by-node matched Source claims
- Cross-Graph six-agent verdicts
- Final projected error spans

## Cache

The direct case command reuses:

```text
outputs\ragtruth_raw_vs_dual_graph_nano\generation_cache_v040.json
```

Therefore a case already processed by the full benchmark can reuse its Raw, Source Graph, Response Graph, and Alignment components. The six-agent review has a separate cache.

## Other commands

Full RAGQA quantitative evaluation:

```bat
run_ragqa.bat
```

Discussion Hub with Balanced Compiler and all downstream agents:

```bat
run_hub.bat
```
