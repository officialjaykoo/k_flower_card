# Project Agent Rules

These rules are mandatory for this repository.
If rules conflict, the smaller section number takes precedence.

1. Primary objective
- The top priority is to find the strongest Matgo AI model.
- A strong model is defined by maximizing gold while minimizing gold loss.
- Development path (phase 1-3): prioritize training/evaluation against heuristic policies.
- Development path (phase 4+): TBD (not fixed yet).

2. Safety and irreversible actions
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
- For testing simulations, use exactly 1000 games.
- Full-league runs must save outputs under `logs/full_league/`.
- If a planned test simulation uses a different game count, stop and ask the user first.
- Clarification: training-time internal evaluator loops (e.g., `games_per_genome`) are not treated as "test simulation" in this lock.

4. Fail-fast and development posture
- Prefer fail-fast over compatibility fallbacks in all runtime/evaluation scripts.
- If behavior is ambiguous, throw an explicit error instead of guessing.
- If a result is uncertain or validation fails, do not mark the run as success.
- Do not silently swallow action-resolution failures; surface context in the error (`seed`, `step`, `actor`, `phase`).
- In development stage, do not add compatibility branches or silent default masking for invalid/missing inputs; stop with an explicit error.
- If the new direction is clearly correct, do not delay structural refactoring.
- Do not keep obsolete structure only for short-term convenience during development stage.
- If the full structural change is too large for one pass, split it into up to 5 explicit steps and instruct the user to run/approve each step in sequence.

5. Path and structure lock
- Keep all runtime/config files under `scripts/configs/`.
- Keep orchestration/entry scripts under `scripts/` (e.g., `phase_run.ps1`, `phase_eval.ps1`).
- Prefer a single entry script with explicit required arguments over per-phase wrapper scripts when logic is shared.
- Do not create or reference root-level `configs/` paths.
- Use repository-relative paths consistently in scripts.

6. Encoding lock
- Save files as UTF-8 with BOM (EF BB BF).
- Keep this encoding so IDEs detect Korean text correctly.

7. Fair-teacher hidden-information rule
- For imperfect-information gameplay and training, decisions must use `PublicState` only.
- `PublicState` must hide opponent hand identities and unknown deck identities while preserving counts.
- For rollout/simulation, Information Set Resampling is mandatory: unknown cards (opponent hand + deck) must be reshuffled/reassigned per sample before evaluation.
- GO decisions must reintroduce rollout contribution with a small weight (recommended range: `0.2` to `0.4`) to reflect probabilistic future value without freezing GO behavior.
- Any logic that directly relies on hidden opponent/deck card identities for decision-making is prohibited in fair-teacher mode.

8. CLI argument strictness
- For list-type CLI inputs, use exactly one canonical format only. Do not support multiple calling styles for compatibility.
- `scripts/heuristic_full_league_1000.ps1` must receive policies as one CSV string:
  - `-Policies "H-V4,H-V5,H-V5P,H-V6"`
- Do not use array/repeated variants such as `-Policies @(...)` or multiple `-Policies` bindings.
- If the input format is wrong, fail fast with a clear error message instead of adding fallback parsing branches.

9. Result reporting format lock
- When reporting duel or league results, use this structure for readability:
  - Reference file line
  - Compact comparison table
  - Seat split summary for the focus model
  - One-line core conclusion
- Preferred template:
  - `기준 파일: <result.json>`
  - Table headers: `항목 | <Model A> | <Model B>`
  - Required rows:
    - `전적`
    - `승률`
    - `평균 골드델타`
    - `GO 횟수`
    - `GO 발생 게임 수`
    - `GO 실패 수`
    - `GO 실패율`
  - Seat split block:
    - `좌석별(<Focus Model>):`
    - `선공: 승률 <...>, 평균 골드델타 <...>`
    - `후공: 승률 <...>, 평균 골드델타 <...>`
  - Final line:
    - `핵심 한줄: <single-sentence conclusion>`
- After a `scripts/neat_eval_worker.mjs` run completes, always present this compact duel block first:
  - `=== Model Duel (<Model A> vs <Model B>, games=<N>) ===`
  - `Win/Loss/Draw(A):  <...> / <...> / <...>  (WR=<...>)`
  - `Win/Loss/Draw(B):  <...> / <...> / <...>  (WR=<...>)`
  - `Seat A first:      WR=<...>, mean_gold_delta=<...>`
  - `Seat A second:     WR=<...>, mean_gold_delta=<...>`
  - `Seat B first:      WR=<...>, mean_gold_delta=<...>`
  - `Seat B second:     WR=<...>, mean_gold_delta=<...>`
  - `Gold delta(A):     mean=<...>, p10=<...>, p50=<...>, p90=<...>`
  - `GO A:              count=<...>, games=<...>, fail=<...>, fail_rate=<...>`
  - `GO B:              count=<...>, games=<...>, fail=<...>, fail_rate=<...>`
  - `Bankrupt:          A=<...>, B=<...>`
  - `Eval time:         <...>s`
  - `===========================================================`
- After the block, add one short Korean conclusion line.

10. Response clarity
- Always answer clearly and state the conclusion first.
