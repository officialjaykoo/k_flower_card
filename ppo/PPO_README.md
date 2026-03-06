# PPO Workspace

This folder is the PPO pipeline for Matgo. The priority is:
1) maximize expected gold delta
2) reduce catastrophic downside
3) keep win rate stable enough to avoid fragile models

## Source Structure (Execution Order)

### Training path
1. `ppo/run_ppo.ps1`
2. `ppo/scripts/train_ppo.py`
3. `ppo/scripts/ppo_env_bridge.mjs`
4. `src/engine/*`, `src/ai/*` (called by the bridge)
5. `logs/PPO/*` checkpoints and metrics

### Duel evaluation path
1. `ppo/run_duel_ppo.ps1`
2. `ppo/scripts/duel_ppo_vs_v5.py`
3. `ppo/scripts/ppo_env_bridge.mjs`
4. `logs/model_duel/*` result json

## Folder Map

- `ppo/configs/`
- `runtime_phase1_ppo.json`: phase1 single-actor training runtime
- `runtime_phase2_ppo.json`: phase2 self-play runtime (resume required)
- `runtime_phase1_resume_u2000.json`: phase1 resume runtime
- `runtime_phase1_from_p2_u2000.json`: phase1 retrain runtime from phase2 context
- `duel_ppo_vs_v5_1000.json`: PPO vs H-CL duel runtime

- `ppo/scripts/`
- `train_ppo.py`: masked PPO trainer with strict validation and checkpointing
- `ppo_env_bridge.mjs`: step/reset bridge to Matgo engine
- `duel_ppo_vs_v5.py`: multi-worker duel runner and risk metrics

- `ppo/run_ppo.ps1`: training entry wrapper
- `ppo/run_duel_ppo.ps1`: duel entry wrapper

## Standard Commands

```powershell
.\ppo\run_ppo.ps1 -RuntimeConfig .\ppo\configs\runtime_phase1_ppo.json -Seed 90
```

```powershell
.\ppo\run_ppo.ps1 -RuntimeConfig .\ppo\configs\runtime_phase1_ppo.json -Seed 92 -TotalUpdates 200 -LogEveryUpdates 20 -SaveEveryUpdates 20
```

```powershell
.\ppo\run_duel_ppo.ps1 -RuntimeConfig .\ppo\configs\duel_ppo_vs_v5_1000.json
```

```powershell
.\ppo\run_duel_ppo.ps1 -RuntimeConfig .\ppo\configs\duel_ppo_vs_v5_1000.json -Seed ppo_vs_hcl_phase1_seed93_best_1000 -CheckpointPath .\logs\PPO\vs_hcl_phase1_seed93\best_stage1.pt -ResultOut .\logs\model_duel\ppo_vs_hcl_phase1_seed93_best_1000_result.json -Games 1000 -OpponentPolicy H-CL:1.0
```

## Opponent Policy Format (Single Canonical)

For `training_mode=single_actor`, `opponent_policy` must use this exact format:

`POLICY:WEIGHT|POLICY:WEIGHT|...`

Example:

`H-CL:0.60|H-J2:0.15|H-AntiCL:0.15|H-NEXg:0.10`

Notes:
- all weights must be finite and `> 0`
- duplicate policies are rejected
- probabilities are normalized from the given weights

## Resume from Update 1325 Example

```powershell
.\ppo\run_ppo.ps1 `
  -RuntimeConfig .\ppo\configs\runtime_phase1_resume_u2000.json `
  -Seed 90 `
  -ResumeCheckpoint .\logs\PPO\vs_hcl_phase1_seed90\checkpoint_update_1325.pt `
  -OutputDir .\logs\PPO\vs_hcl_phase1_seed90_recover1325
```
