# neat_by_GPT

Self-contained GPT-specific NEAT pipeline.

## Entry Points

Train:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/run.ps1 -Seed 9
```

Evaluate:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/eval.ps1 -Seed 9
```

Duel:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/duel.ps1 -Human H-CL -Ai runtime_focus_cl_v1_seed9
```

Optional runtime override:

```powershell
powershell -ExecutionPolicy Bypass -File neat_by_GPT/run.ps1 -Seed 9 -RuntimeConfig .\neat_by_GPT\configs\runtime_focus_cl_v1.json
```

## Layout

- `run.ps1`: training entry point
- `eval.ps1`: winner evaluation entry point
- `duel.ps1`: GPT duel entry point
- `configs/runtime_focus_cl_v1.json`: GPT runtime profile
- `configs/neat_feedforward.ini`: NEAT topology/config
- `scripts/neat_train_worker.py`: training worker
- `scripts/neat_eval_worker.mjs`: evaluation worker
- `scripts/model_duel_worker.mjs`: GPT duel worker

## Teacher Policy

- `teacher_policy` is runtime-driven, not hardcoded in the evaluator
- Set `teacher_policy` to `H-GPT` to use heuristic imitation
- Set `teacher_policy` to `""` to disable teacher guidance explicitly

## Duel Specs

- Heuristic policy key: `H-CL`, `H-J2`, `H-GPT`
- GPT run name: `runtime_focus_cl_v1_seed9`
- Explicit model path: `model:logs/NEAT_GPT/runtime_focus_cl_v1_seed9/models/winner_genome.json`

## Outputs

- `logs/NEAT_GPT/*`
- `logs/NEAT_GPT/duels/*`
