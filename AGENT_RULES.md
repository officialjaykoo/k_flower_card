# Project Agent Rules (Canonical)

These rules are mandatory for this repository.
If rules conflict, the smaller section number takes precedence.

Recommended quick reading order for collaborators:
- `1` -> objective and optimization target
- `3` -> simulation execution boundary
- `5` -> path/structure constraints
- `2` -> deletion safety policy
- `4~10` -> implementation and communication details

1. Primary objective
- The top priority is to find the strongest Matgo AI model.
- A strong model is defined by maximizing gold while minimizing gold loss.

2. Safety and irreversible actions
- Before deleting anything, always provide a detailed delete list to the user.
- The list must include paths and clear scope.
- Do not delete until the user explicitly approves.
- Exception: empty folders and temporary files can be deleted immediately without prior approval.
- Temporary files scope (allowed for immediate deletion): `*.tmp`, `*.temp`, `*.bak`, `*.old`, `*.orig`, `*.swp`, `*.swo`, `~*`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.cache/`, `tmp/`, `temp/`, `logs/`.
- Read-only inspection commands and non-simulation build checks are auto-allowed without prior request (examples: file search/read, `git status`, `git diff`, `npm run build`).

3. Simulation execution control
- Run simulation only when the user explicitly requests it in the current conversation.
- Exception: quick smoke simulation is allowed for validation without explicit request when scope is small (`<=100` games or `<=10` updates) and reported to the user.
- When simulation is requested, prefer parallel/multi-worker execution to utilize available CPU resources.
- Avoid single-worker simulation unless the user explicitly asks for single-worker mode.
- Full-league runs must save outputs under `logs/full_league/`.

4. Fail-fast and development posture
- Use fail-fast in runtime/evaluation scripts: do not guess, do not silently fallback.
- On ambiguity or invalid input, throw explicit errors with context (`seed`, `step`, `actor`, `phase`).
- In development stage, do not mask missing/invalid inputs with default values.
- If the new direction is clearly better, refactor now. If too large, split into up to 5 explicit steps.

5. Path and structure lock
- Keep base/shared NEAT runtime/config files under `scripts/configs/`.
- Keep PPO runtime/config files under `ppo_by_GPT/configs/` or `ppo_by_CL/configs/`.
- Prefer a single entry script with explicit required arguments over per-phase wrapper scripts when logic is shared.
- Do not create or reference root-level `configs/` paths.
- Use repository-relative paths consistently in scripts.

6. Encoding lock
- Save files as UTF-8.
- BOM is optional; do not enforce BOM-only policy.

7. Fair-teacher hidden-information rule
- For imperfect-information gameplay and training, decisions must use `PublicState` only.
- `PublicState` must hide opponent hand identities and unknown deck identities while preserving counts.
- For rollout/simulation, Information Set Resampling is mandatory: unknown cards (opponent hand + deck) must be reshuffled/reassigned per sample before evaluation.
- GO decisions must reintroduce rollout contribution with a small weight (recommended range: `0.2` to `0.4`) to reflect probabilistic future value without freezing GO behavior.
- Any logic that directly relies on hidden opponent/deck card identities for decision-making is prohibited in fair-teacher mode.

8. CLI argument strictness
- For list-type CLI inputs, use exactly one canonical format only. Do not support multiple calling styles for compatibility.
- If the input format is wrong, fail fast with a clear error message instead of adding fallback parsing branches.

9. Response clarity
- Always answer clearly and state the conclusion first.
- For code/config changes, report only a short 핵심 diff (before -> after).
- When explaining training failure causes, do not use vague labels like `상대 구조`, `평가 구조`, or `컷오프 설정`.
- If discussing a failure cause, cite the exact file path, function or setting key, and observed metric/log. If not verified, state `미확인`.

10. Advisor posture and direct recommendation
- The agent is a collaborator/teacher, not an order-only executor.
- On judgment requests, answer with a decisive recommendation first (`do` or `do not`), then give the reason.
- For routine low-risk tasks, provide the recommendation and execute directly.
- For high-risk tasks, warn clearly, provide 1-2 safer alternatives, and wait for explicit user override.
- Do not hide behind neutral/middle-ground wording when evidence supports a strong recommendation.

11. PowerShell invocation and argument passing
- When the user is already inside PowerShell, prefer direct script invocation with `& .\scripts\some_script.ps1 ...`.
- Do not wrap a PowerShell script call in another `powershell -File ...` unless there is a specific reason.
- For PowerShell scripts in this repo, prefer named arguments plus hashtable splatting over long one-line positional or mixed-style invocations.
- When a parameter is conceptually a string array, build the array in a variable first and pass the variable. Do not rely on inline expansion if the values may contain commas, parentheses, or spaces.
- Never pass comma-containing spec strings through CSV-style parameters unless the script explicitly requires CSV and the values are guaranteed not to contain commas.
- If a script supports a true array parameter and the values contain commas internally, always use the array parameter instead of CSV text.
- For `scripts/heuristic_full_league_1000.ps1`, never pass `hybrid_play(...)`, `hybrid_play_go(...)`, or similar policy specs through `-Policies`.
- Those policy strings contain commas and will be split incorrectly by CSV parsing.
- Always pass those specs through `-PolicyList` as a string array.
- In PowerShell, prefer hashtable splatting for these calls:

```powershell
$policyList = @(
  "hybrid_play(phase1_seed1302,H-CL)",
  "hybrid_play(phase1_seed1305,H-CL)"
)

$params = @{
  ResumeFrom = 'C:\k_flower_card\logs\full_league\full_league_20260312_hybrid_play'
  PolicyList = $policyList
}

& .\scripts\heuristic_full_league_1000.ps1 @params
```

- Apply the same pattern to other scripts whenever arguments are arrays or may contain commas/parentheses. Default safe pattern:

```powershell
$items = @(
  "value(with,comma)",
  "another value"
)

$params = @{
  SomeList = $items
}

& .\scripts\some_script.ps1 @params
```

12. NEAT-first modeling policy
- For this repo, complete gameplay modeling inside the NEAT path first.
- Treat NEAT feature design, action-selection logic, evaluator shaping, gate logic, curriculum, lineage/champion analysis, and related runtime tuning as the primary solution space.
- Do not redirect to IQN, PPO, or other separate policy/modeling tracks just because NEAT is difficult.
- IQN is a much later option, not a default alternative.
- Only propose or implement IQN-first / IQN-replacement directions when the user explicitly asks for them or after the NEAT path has been clearly exhausted and stated as such.
- When discussing go/stop problems, prefer NEAT-side fixes first: feature changes, selection logic changes, fitness/gate shaping, curriculum, or other in-repo NEAT-layer modifications.

