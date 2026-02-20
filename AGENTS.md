# Project Agent Rules

These rules are mandatory for this repository.

0. Response clarity first
- Always answer clearly and state the conclusion first.

1. Deletion safety
- Before deleting anything, always provide a detailed delete list to the user.
- The list must include paths and clear scope.
- Do not delete until the user explicitly approves.

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

5. CPU execution confirmation
- Before running training/evaluation commands with CPU, always ask the user first.
- Do not run CPU-based training/evaluation unless the user explicitly approves in the current conversation.

6. Value training fixed policy
- For `02_train_value.py`, use fixed epochs by data size:
  - `<= 20,000` games: `epochs=6`
  - `20,001 ~ 100,000` games: `epochs=8`
  - `>= 100,001` games: `epochs=8` (maximum `10`)
- For data size `>= 50,000`, cache is mandatory:
  - Use `--sample-cache auto --cache-backend lmdb`
  - Do not use `--sample-cache none`.
