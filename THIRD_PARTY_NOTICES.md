# Third-Party Notices and Integration Provenance

This repository is a unified project release assembled from CPR team repositories. Only the components needed for the final STRING presentation were retained.

## Source repositories

### `curry2175/CPR_STRING`

- URL: https://github.com/curry2175/CPR_STRING
- Role: integration base; STRING six-agent core, web interface, RAGTruth evaluation, 330-case artifacts, and optimized gate configuration.
- License: MIT, retained in the root `LICENSE`.

### `skhyun15/verified-reasoning-graph`

- URL: https://github.com/skhyun15/verified-reasoning-graph
- Imported component: `browser-extension/` → `extensions/chrome/`.
- Modifications: user-facing branding changed to STRING; local API routes retained for compatibility; floating-button preference is honored.
- License copy: `third_party/LICENSE_verified_reasoning_graph_MIT.txt`.

### `GracieRho/solving-aime`

- URL: https://github.com/GracieRho/solving-aime
- Imported/adapted components: the math self-revision loop, OpenAI solver wrapper, deterministic graph auditor, and revision-question generation → `applications/aime_self_revision/`.
- Modifications: duplicate historical `vrg/` was removed; imports now use the single root STRING core; CLI and Windows launcher were standardized.
- License copy: `third_party/LICENSE_solving_aime_MIT.txt`.

### `Hyunmin-3428/STRING_agent`

- URL: https://github.com/Hyunmin-3428/STRING_agent
- Imported component: `lab/ifxlogic/` → `applications/impact_factor/`.
- Modifications: the mirrored Discussion Lab module was removed; analysis now imports the single root STRING core; runners and documentation were standardized.
- License copy: `third_party/LICENSE_STRING_agent_APACHE_2.0.txt`.

## Excluded material

The integration intentionally excludes duplicate core packages, old server copies, update/rollback wrappers, generated model outputs, local caches, private corpora, and unrelated historical benchmark launchers.
