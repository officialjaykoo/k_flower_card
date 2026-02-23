# Project Agent Rules

These rules are mandatory for this repository.

0. Response clarity first
- Always answer clearly and state the conclusion first.

1. Deletion safety
- Before deleting anything, always provide a detailed delete list to the user.
- The list must include paths and clear scope.
- Do not delete until the user explicitly approves.
- Exception: empty folders and temporary files can be deleted immediately without prior approval.
- Temporary files scope (allowed for immediate deletion): `*.tmp`, `*.temp`, `*.bak`, `*.old`, `*.orig`, `*.swp`, `*.swo`, `~*`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.cache/`, `tmp/`, `temp/`.

2. Simulation execution control
- Do not run simulation commands on your own.
- Run simulation only when the user explicitly requests it in the current conversation.
- When simulation is requested, prefer parallel/multi-worker execution to utilize available CPU resources.
- Avoid single-worker simulation unless the user explicitly asks for single-worker mode.

3. Test game count lock
- For testing simulations, use exactly 1000 games.
- If a planned test simulation uses a different game count, stop and ask the user first.

4. Encoding lock
- Save files as UTF-8 with BOM (EF BB BF).
- Keep this encoding so IDEs detect Korean text correctly.

5. Path structure lock
- Keep all runtime/config files under `scripts/configs/`.
- Keep orchestration/entry scripts under `scripts/` (e.g., `phase1_run.ps1`, `phase1_eval.ps1`, `phase2_run.ps1`, `phase2_eval.ps1`).
- Do not create or reference root-level `configs/` paths.
- Use repository-relative paths consistently in scripts.
