# Quick Start

## Setup

```bat
setup.bat
copy .env.example .env
```

Add `OPENAI_API_KEY` to `.env`.

## Core STRING interface

```bat
run_string.bat
```

Open `http://127.0.0.1:8765/discussion-lab`.

## RAGTruth

```bat
run_case.bat 14806
run_ragtruth.bat
```

## Impact-factor application

```bat
applications\impact_factor_COLLECT.bat "OPEN_ACCESS:y AND HAS_FT:y AND (COVID-19)" 20
run_impact_factor.bat --limit 5
```

## AIME self-revision

```bat
run_aime.bat --file applicationsime_self_revision\sample_problem.txt
```

## Chrome extension

Load `extensions/chrome/` as an unpacked extension after starting the local server.
