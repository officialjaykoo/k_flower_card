# Project Agent Rules

These rules are mandatory for this repository.

0. Primary objective
- The top priority is to find the strongest Matgo AI model.
- A strong model is defined by maximizing gold while minimizing gold loss.
- Development path (phase 1): use Heuristic + Optuna and NEAT in early head-to-head runs to build initial strong models.
- Development path (phase 2): after initial modeling, prioritize model-vs-model league matches to iteratively select and improve stronger models.

1. Response clarity first
- Always answer clearly and state the conclusion first.

2. Deletion safety
- Before deleting anything, always provide a detailed delete list to the user.
- The list must include paths and clear scope.
- Do not delete until the user explicitly approves.
- Exception: empty folders and temporary files can be deleted immediately without prior approval.
- Temporary files scope (allowed for immediate deletion): `*.tmp`, `*.temp`, `*.bak`, `*.old`, `*.orig`, `*.swp`, `*.swo`, `~*`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.cache/`, `tmp/`, `temp/`.

3. Simulation execution control
- Do not run simulation commands on your own.
- Run simulation only when the user explicitly requests it in the current conversation.
- When simulation is requested, prefer parallel/multi-worker execution to utilize available CPU resources.
- Avoid single-worker simulation unless the user explicitly asks for single-worker mode.

4. Test game count lock
- For testing simulations, use exactly 1000 games.
- Full-league runs must save outputs under `logs/heuristic_duel/full_league/`.
- If a planned test simulation uses a different game count, stop and ask the user first.

5. Encoding lock
- Save files as UTF-8 with BOM (EF BB BF).
- Keep this encoding so IDEs detect Korean text correctly.

6. Path structure lock
- Keep all runtime/config files under `scripts/configs/`.
- Keep orchestration/entry scripts under `scripts/` (e.g., `phase1_run.ps1`, `phase1_eval.ps1`, `phase2_run.ps1`, `phase2_eval.ps1`).
- Do not create or reference root-level `configs/` paths.
- Use repository-relative paths consistently in scripts.

7. Fair-teacher hidden-information rule
- For imperfect-information gameplay and training, decisions must use `PublicState` only.
- `PublicState` must hide opponent hand identities and unknown deck identities while preserving counts.
- For rollout/simulation, Information Set Resampling is mandatory: unknown cards (opponent hand + deck) must be reshuffled/reassigned per sample before evaluation.
- GO decisions must reintroduce rollout contribution with a small weight (recommended range: `0.2` to `0.4`) to reflect probabilistic future value without freezing GO behavior.
- Any logic that directly relies on hidden opponent/deck card identities for decision-making is prohibited in fair-teacher mode.
