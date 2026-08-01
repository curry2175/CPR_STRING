# Integration Manifest

| Final path | Source | Included because |
|---|---|---|
| `vrg/`, `static/`, `app.py` | `curry2175/CPR_STRING` | Core six-agent architecture and interactive graph interface |
| `extensions/chrome/` | `skhyun15/verified-reasoning-graph/browser-extension` | Chrome extension shown in the model-overview slide |
| `applications/impact_factor/` | `Hyunmin-3428/STRING_agent/lab/ifxlogic` | Impact-factor application shown in the application slide |
| `applications/aime_self_revision/` | `GracieRho/solving-aime` | AIME self-revision application shown in the application slide |
| `results/`, RAGTruth launchers | `curry2175/CPR_STRING` | Hallucination-detection evaluation shown in the presentation |
| `docs/STRING_CPR_Track1_CR.pdf` | final CPR presentation | Defines the scope of the unified release |

## Integration rule

Only one `vrg/` core is allowed. Application modules must import the root package rather than bundle a historical copy.
