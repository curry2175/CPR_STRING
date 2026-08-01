# Reasoning-Graph Features and Journal Impact

This application corresponds to the **Correlation with Impact Factor** demonstration in the STRING presentation.

1. `collect_discussions.py` retrieves open-access article discussions from Europe PMC and attaches journal metadata from `journals.csv`.
2. `analyze_batch.py` sends each discussion to the **single root STRING core** and exports graph metrics to `out/metrics.csv`.
3. `journal_frequency.py` summarizes journal coverage.

## Windows quick start

```bat
1_COLLECT.bat "OPEN_ACCESS:y AND HAS_FT:y AND (COVID-19)" 20
2_ANALYZE.bat --limit 5
```

The analysis is exploratory and correlational. Journal impact must not be interpreted as a causal consequence of reasoning-graph features. Large corpora, cached model outputs, and private study sets are ignored by Git.
