# STRING

**System for Traceable Reasoning Integrity through Networked Graphs**

STRING converts unrestricted natural-language reasoning into an inspectable node–edge graph and validates distinct error dimensions with six role-specialized agents. This repository is the unified CPR team release corresponding to the final presentation.

## What is included

| Component | Purpose |
|---|---|
| **STRING Core** | Six-agent reasoning-graph construction, validation, conditional arbitration, and interactive visualization |
| **Chrome Extension** | Send selected webpage text to the local STRING server and open the graph report |
| **RAGTruth Evaluation** | Raw Direct vs graph-based hallucination localization, optimized gate preset, and 330-case development results |
| **Impact-Factor Application** | Extract graph features from article discussions and explore their association with journal impact |
| **AIME Self-Revision** | Generate a math solution, audit it as a six-agent graph, revise it, and re-evaluate the graph |

## Six-agent architecture

1. **Compiler Agent** — transforms natural language into a typed reasoning graph.
2. **Evidence Agent** — checks grounding, provenance, missing qualifiers, and claim fidelity.
3. **Logic Agent** — checks reasoning edges, consistency, causal direction, and missing premises.
4. **Target Agent** — checks task relevance, answerability, and coverage.
5. **Assumption Agent** — conditionally evaluates implicit or missing premises.
6. **Judge Agent** — conditionally resolves conflicting findings and selects local graph repairs.

Evidence, Logic, and Target inspection run as specialist reviews. Assumption and Judge are activated only when the graph contains a relevant trigger; they are not identical majority voters.

## Quick start on Windows

```bat
setup.bat
run_string.bat
```

Open:

```text
http://127.0.0.1:8765/discussion-lab
```

Create a root `.env` from `.env.example` and set `OPENAI_API_KEY` before model-backed runs.

## Chrome extension

1. Start STRING with `run_string.bat`.
2. Open `chrome://extensions` and enable **Developer mode**.
3. Choose **Load unpacked** and select `extensions/chrome`.
4. Select text on a webpage and choose **Analyze with STRING**.

The extension talks only to the local backend. It does not contain or store an OpenAI API key.

## Applications

### RAGTruth hallucination detection

```bat
run_case.bat 14806
run_ragtruth.bat
run_threshold_optimizer.bat
run_rescue_report.bat
```

The fixed 330-case development preset is stored at `config/optimized_gate_330.json`.

| Method | Character micro-F1 |
|---|---:|
| Raw Direct | 37.42% |
| Original DualGraph | 35.55% |
| **Threshold-optimized DualGraph** | **47.59%** |

The optimized score is an in-sample development result. It improved Raw Direct by 10.17 percentage points, primarily through fewer false positives on clean responses and better localization in selected hallucinated cases. Strict hallucination rescue was 0 in this 330-case set. See `docs/RESULTS_330.md`.

### Reasoning-graph features and journal impact

```bat
applications\impact_factor_COLLECT.bat "OPEN_ACCESS:y AND HAS_FT:y AND (COVID-19)" 20
run_impact_factor.bat --limit 5
```

This is exploratory correlational analysis, not a causal claim about journal impact.

### AIME self-revision

```bat
run_aime.bat --file applicationsime_self_revision\sample_problem.txt --model gpt-5.4-nano
```

Per-round graph JSON and standalone HTML reports are generated under `outputs/`.

## Repository structure

```text
CPR_STRING/
├─ vrg/                         # single current STRING core
├─ static/                      # interactive graph interface
├─ extensions/chrome/           # browser integration
├─ applications/
│  ├─ impact_factor/            # discussion-graph feature analysis
│  └─ aime_self_revision/       # math audit and revision loop
├─ config/optimized_gate_330.json
├─ results/                     # compact reproducibility artifacts
├─ docs/                        # architecture, applications, results, presentation
├─ tests/
├─ app.py
├─ run_string.bat
├─ run_ragtruth.bat
├─ run_impact_factor.bat
└─ run_aime.bat
```

All applications import the **single root `vrg/` package**. Duplicate historical copies of the core from contributor repositories are intentionally excluded.

## Reproducibility and secrets

The repository excludes `.env`, `.venv`, model caches, downloaded corpora, and generated outputs. The compact 330-case console log and derived tables remain in `results/`. To import the original local `cases.jsonl`, use:

```bat
scripts\IMPORT_LOCAL_RUN_WINDOWS.bat
```

## Team

CPR, Track 1: CR — 김동현, 노승현, 조나현, 최현민.

## Source repositories and licenses

This unified release incorporates selected components from the four CPR project repositories. See `THIRD_PARTY_NOTICES.md` and `third_party/` for provenance, modifications, and retained license texts.

## References

The project presentation and literature list are available at `docs/STRING_CPR_Track1_CR.pdf`.
