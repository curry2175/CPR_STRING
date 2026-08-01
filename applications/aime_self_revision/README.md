# AIME Self-Revision with STRING

This application implements the presentation flow:

`AIME problem → initial solution → STRING six-agent reasoning graph → deterministic audit → focused revision → graph re-evaluation`

The graph audits the public solution; it does not replace the solution writer. Each revision is recompiled and rechecked. Per-round JSON and standalone HTML graph views are written next to the result.

## Run from the repository root

```bat
run_aime.bat --problem "Your AIME problem here"
```

or

```bat
.venv\Scripts\python.exe -m applications.aime_self_revision.math_self_revision --file problem.txt --model gpt-5.4-nano
```

An OpenAI API key must be present in the root `.env`. The included deterministic auditor checks structural problems such as unsupported conclusions, certainty escalation, scope widening, and self-supporting edges.
