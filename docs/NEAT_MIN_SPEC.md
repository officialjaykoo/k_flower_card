# NEAT-Python Training Spec

## Conclusion
The training path is now `neat-python` only.
`Deep CFR` and custom genome paths are out of the active runtime path.

## Active Files
- `configs/neat_feedforward.ini`
- `configs/neat_runtime.json`
- `configs/neat_runtime_i3_w6.json`
- `configs/neat_runtime_i3_w7.json`
- `scripts/neat/neat_train.py`
- `scripts/neat/neat_eval_worker.mjs`
- `scripts/neat/run_neat_train.ps1`
- `scripts/neat/run_neat_resume.ps1`
- `scripts/neat/run_neat_eval.ps1`
- `scripts/neat/run_neat_duel.ps1`

## 1-10 Execution Checklist
1. Install dependency: `.venv\Scripts\python -m pip install neat-python`
2. Use i3 default runtime profile: `configs/neat_runtime_i3_w6.json`
3. Use stress compare profile when needed: `configs/neat_runtime_i3_w7.json`
4. Use fixed train entrypoint: `scripts/neat/run_neat_train.ps1`
5. Use fixed resume entrypoint: `scripts/neat/run_neat_resume.ps1`
6. Use fixed genome eval entrypoint: `scripts/neat/run_neat_eval.ps1` (default `games=1000`)
7. Use runtime CLI overrides in `scripts/neat/neat_train.py`:
   - `--games-per-genome`, `--workers`, `--eval-timeout-sec`, `--max-eval-steps`
   - `--opponent-policy`, `--switch-seats|--fixed-seats`
   - `--checkpoint-every`, `--seed`
8. Use fitness CLI overrides in `scripts/neat/neat_train.py`:
   - `--fitness-gold-scale`
   - `--fitness-win-weight`
   - `--fitness-loss-weight`
   - `--fitness-draw-weight`
9. Debug worker failures via `logs/neat_python/eval_failures.log`
10. Judge runs by `logs/neat_python/run_summary.json` and `logs/neat_python/models/winner_genome.json`

## Runtime Flow
1. `scripts/neat/neat_train.py` loads feedforward config and runtime config.
2. Each genome is exported as `neat_python_genome_v1` JSON.
3. `scripts/neat/neat_eval_worker.mjs` runs game evaluations and returns fitness JSON.
4. `ParallelEvaluator` handles parallel genome scoring.
5. `Checkpointer` stores progress at `checkpoint_every` intervals.
6. `run_summary.json` stores effective runtime and applied overrides.

## i3-1315U Defaults
- Recommended: `workers=6` via `configs/neat_runtime_i3_w6.json`
- Compare mode: `workers=7` via `configs/neat_runtime_i3_w7.json`
- `games_per_genome=3`
- `pop_size=50`
- `conn_add_prob=0.05`, `node_add_prob=0.02`
- Compact action feature vector: 20 dimensions

## Commands
Start training:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/neat/run_neat_train.ps1
```

Dry run:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/neat/run_neat_train.ps1 -DryRun
```

Resume from checkpoint:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/neat/run_neat_resume.ps1 -ResumeCheckpoint logs/neat_python/checkpoints/neat-checkpoint-50
```

Evaluate one genome:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/neat/run_neat_eval.ps1 -GenomePath logs/neat_python/models/winner_genome.json
```

Duel two genomes (fixed 1000 games):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/neat/run_neat_duel.ps1 -GenomeAPath logs/neat_python_runA/models/winner_genome.json -GenomeBPath logs/neat_python_runB/models/winner_genome.json
```
