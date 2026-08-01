# v048 direct-case interactive graph guide

## Run one RAGTruth case

```bat
cd /d C:\vrg\v048
setup.bat
run_case.bat 12046
```

The full RAGQA benchmark does not need to finish first.

## Output

```text
outputs\ragtruth_case_6agent_reviews\catch6_<CASE_ID>_<TIME>\
├─ report.html
├─ result.json
├─ graph_view.json
└─ README.txt
```

`report.html` opens automatically and contains three interactive graph tabs:

1. **Source Graph**
2. **Response Graph**
3. **Cross comparison**

Use **Graph stage** to switch between the original Balanced candidate graph and the graph after six-agent validation. Click any node or edge to inspect its exact text, normalized proposition, connections, Source match, Balanced alignment, and six-agent verdict.

To prevent automatic browser opening:

```bat
run_case.bat 12046 --no-open
```
