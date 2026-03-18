# K-HyperNEAT Matgo Integration

This folder contains `k_flower_card`-specific code that should stay outside the
public `K-HyperNEAT` core.

Moved here:

- Matgo substrate layout
- local training runner
- local runtime config
- local smoke/runtime bridge scripts
- phase-run / phase-eval wiring targets

Core engine:

- `../k_hyperneat_py/k_hyperneat/`

Local entrypoints:

- `train_k_hyperneat.py`
- `configs/runtime_phase1.json`
- `configs/k_hyperneat_cppn.ini`

Examples:

```powershell
python experiments/k_hyperneat_matgo/train_k_hyperneat.py --seed 1705
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/phase_run.ps1 -Phase 1 -Seed 1705 -LineageProfile k-hyperneat
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/phase_eval.ps1 -Phase 1 -Seed 1705 -LineageProfile k-hyperneat
```
