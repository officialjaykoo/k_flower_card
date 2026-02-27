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

8. CLI list-argument single-format rule
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

10. `neat_eval_worker.mjs` result summary lock
- After a `scripts/neat_eval_worker.mjs` run completes, always present the result in the compact duel block format first.
- Use this exact header/body order:
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

11. Fail-fast execution principle
- Prefer fail-fast over compatibility fallbacks in all runtime/evaluation scripts.
- If behavior is ambiguous, throw an explicit error instead of guessing.
- If a result is uncertain or validation fails, do not mark the run as success.
- Do not silently swallow action-resolution failures; surface context in the error (`seed`, `step`, `actor`, `phase`).
