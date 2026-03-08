# neat_by_GPT

Self-contained GPT-specific NEAT pipeline.

## Entry Points

Train:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/run.ps1 -Seed 9
```

Train by phase:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/run.ps1 -Phase 1 -Seed 9
powershell -ExecutionPolicy Bypass -File neat_by_GPT/run.ps1 -Phase 2 -Seed 9
powershell -ExecutionPolicy Bypass -File neat_by_GPT/run.ps1 -Phase 3 -Seed 9
```

- `-Phase 2` and `-Phase 3` automatically reuse the previous phase `winner_genome.pkl` as `--seed-genome`.
- The runner also carries `--base-generation` from the previous phase summary.
- Legacy GPT outputs under `logs/NEAT_GPT/neat_phase*_seed*` are accepted as fallback sources.

Evaluate:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/eval.ps1 -Seed 9
```

Evaluate by phase:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/eval.ps1 -Phase 1 -Seed 9
powershell -ExecutionPolicy Bypass -File neat_by_GPT/eval.ps1 -Phase 2 -Seed 9
powershell -ExecutionPolicy Bypass -File neat_by_GPT/eval.ps1 -Phase 3 -Seed 9
```

Duel:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/duel.ps1 -Human H-CL -Ai runtime_focus_cl_v1_seed9
```

Regression smoke:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/test.ps1
```

Optional runtime override:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/run.ps1 -Seed 9 -RuntimeConfig .\neat_by_GPT\configs\runtime_focus_cl_v1.json
```

`-RuntimeConfig` is still supported and takes priority over `-Phase`.

## Layout

- `run.ps1`: training entry point
- `eval.ps1`: winner evaluation entry point
- `duel.ps1`: GPT duel entry point
- `test.ps1`: GPT regression smoke entry point
- `configs/runtime_focus_cl_v1.json`: GPT runtime profile
- `configs/runtime_base_v1.json`: shared GPT runtime base
- `configs/runtime_phase1.json`: phase 1 GPT runtime
- `configs/runtime_phase2.json`: phase 2 GPT runtime
- `configs/runtime_phase3.json`: phase 3 GPT runtime
- `configs/neat_feedforward.ini`: NEAT topology/config
- `scripts/neat_train_worker.py`: training worker
- `scripts/neat_eval_worker.mjs`: evaluation worker
- `scripts/model_duel_worker.mjs`: GPT duel worker

## Teacher Policy

- `teacher_policy` is runtime-driven, not hardcoded in the evaluator
- Set `teacher_policy` to `H-GPT` to use heuristic imitation
- Set `teacher_policy` to `""` to disable teacher guidance explicitly

## Fitness Overrides

- Runtime JSON may define `fitness_overrides` as an object.
- Overrides are applied on top of the selected `fitness_profile`.
- Supported examples:
  - `tieBreakWeight`
  - `goldMeanWeight`
  - `goldCvar10Weight`
  - `goFailRateCap`
  - `goZeroForceScore`
  - `imitationWeights`
  - `scenarioShardWeights`

## Duel Specs

- Heuristic policy key: `H-CL`, `H-J2`, `H-GPT`
- GPT run name: `runtime_focus_cl_v1_seed9`
- Explicit model path: `model:logs/NEAT_GPT/runtime_focus_cl_v1_seed9/models/winner_genome.json`
- Hybrid main-model spec: `hybrid_play(phase2_seed203,H-CL)`
- Hybrid go-stop only spec: `hybrid_play_go(phase2_seed203,H-CL)`
- Hybrid go-stop override + heuristic fallback spec: `hybrid_play_go(phase2_seed203,H-NEXg,H-CL)`
- Duel summary includes `GO`, `GO Opp`, `Shake`, `Bomb`, `President`, `Gukjin`, and bankrupt stats.

## Outputs

- `logs/NEAT_GPT/*`
- `logs/NEAT_GPT/duels/*`

## Regression Smoke Coverage

- special `playing` candidates include `shake_start:*` and `bomb:*`
- hybrid opponent specs parse and load correctly
- eval worker accepts hybrid opponent policy/mix
- `fitness_overrides` merge into the selected profile as expected
