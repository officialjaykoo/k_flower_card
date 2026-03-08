# k_flower_card

Matgo engine + AI research workspace (NEAT/PPO).
Primary objective: maximize gold gain while minimizing gold loss.

## Quick Start
```powershell
npm install
npm run dev
```

## Run by Goal
- NEAT train: `npm run neat:run -- -Phase 1 -Seed 9`
- NEAT eval: `npm run neat:eval -- -Phase 1 -Seed 9`
- NEAT train (GPT variant): `npm run neat:run:gpt -- -Seed 9`
- NEAT eval (GPT variant): `npm run neat:eval:gpt -- -Seed 9`
- NEAT duel (GPT variant): `npm run neat:duel:gpt -- -Human H-CL -Ai runtime_focus_cl_v1_seed9`
- PPO train (GPT): `npm run ppo:gpt:train -- -RuntimeConfig .\ppo_by_GPT\configs\runtime_phase1_ppo.json -Seed 90`
- PPO duel (GPT): `npm run ppo:gpt:duel -- -RuntimeConfig .\ppo_by_GPT\configs\duel_ppo_vs_v5_1000.json`
- PPO ablation (GPT): `npm run ppo:gpt:ablation -- -TotalUpdates 700`
- PPO train (CL): `npm run ppo:cl:train -- -RuntimeConfig .\ppo_by_CL\configs\runtime_phase1_ppo.json -Seed 101`
- PPO duel (CL): `npm run ppo:cl:duel -- -RuntimeConfig .\ppo_by_CL\configs\duel_ppo_vs_v5_1000.json`
- PPO ablation (CL): `npm run ppo:cl:ablation -- -TotalUpdates 700`

## Environment
- OS: Windows (PowerShell scripts are first-class entry points)
- Node.js: 20+ recommended
- Python: 3.10+ recommended
- PowerShell: 5.1+ or PowerShell 7+

## Python Dependencies
This repository currently has no pinned Python dependency file (`requirements.txt` or `pyproject.toml`).
Install required runtime packages in your active venv:

```powershell
python -m pip install --upgrade pip
pip install torch neat-python optuna
```

## Outputs
- NEAT outputs: `logs/NEAT/neat_phase{1|2|3}_seed{N}/` (`checkpoints/`, `run_summary.json`, `gate_state.json`, `phase*_eval_1000.json`)
- NEAT GPT outputs: `logs/NEAT_GPT/*` (`models/winner_genome.json`, `run_summary.json`, `eval_*.json`, `eval_*_pass.json`)
- NEAT GPT duel outputs: `logs/NEAT_GPT/duels/*`
- PPO GPT outputs: `logs/PPO_GPT/*` (`latest.pt`, `checkpoint_update_*.pt`, `best_stage*.pt`, `metrics_update_*.json`)
- PPO CL outputs: `logs/PPO_CL/*` and some legacy arms under `logs/PPO/*`
- Duel outputs: `logs/model_duel/*`

## Troubleshooting
- `python not found` or `python executable not found`: verify venv path, or pass `-Python .\.venv\Scripts\python` to scripts.
- `phase1 checkpoint not found` in phase2/phase3: run previous phase first with same seed, or provide resume checkpoint.
- `opponent-policy=genome requires opponent_genome path`: set runtime `opponent_genome` or pass CLI override.
- `feature vector size mismatch`: sync NEAT input config with worker feature size (`scripts/configs/neat_feedforward*.ini`).

## Repository Map
- `src/`: game engine, AI policies, web UI
- `scripts/`: base/shared NEAT orchestration/evaluation scripts
- `scripts/configs/`: base NEAT runtime/config files
- `heuristic_tuning/`: heuristic tuning tools (`optuna_cl.py`, `optuna_gemini.py`, `optuna_nexg.py`, `optuna_gpt.py`, `optimizer_gpt.mjs`, `analyze_anticl_results_gpt.mjs`, `tune_anticl_go_gpt.mjs`)
- `neat_by_GPT/`: self-contained GPT-specific NEAT pipeline (`run.ps1`, `eval.ps1`, `scripts/`, `configs/`)
- `ppo_by_GPT/`: PPO (GPT) pipeline
- `ppo_by_CL/`: PPO (CL) pipeline
- `logs/`: training and evaluation outputs

## Document SoT
- Agent behavior and guardrails: `AGENT_RULES.md`
- Game rules source: `docs/rules/GAME_RULES.md`
- NEAT source-of-truth runbook: `docs/neat/README.md`
- GPT NEAT self-contained runbook: `neat_by_GPT/README.md`
- PPO GPT source-of-truth runbook: `ppo_by_GPT/README.md`
- PPO feature/reward specs: `docs/ppo/OBSERVATION_FEATURE_SPEC.md`, `docs/ppo/PHASE1_REWARD_STAGE_GUIDE.md`


