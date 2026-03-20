# k_flower_card

Matgo engine + AI research workspace (NEAT).
Primary objective: maximize gold gain while minimizing gold loss.

## Quick Start
```powershell
npm install
npm run dev
```

## Run by Goal
- NEAT train: `npm run neat:run -- -Phase 1 -Seed 9`
- NEAT eval: `npm run neat:eval -- -Phase 1 -Seed 9`

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
pip install neat-python optuna
```

## Outputs
- NEAT outputs: `logs/NEAT/neat_phase{1|2|3}_seed{N}/` (`checkpoints/`, `run_summary.json`, `gate_state.json`, `phase*_eval_1000.json`)
- Duel outputs: `logs/model_duel/*`

## Troubleshooting
- `python not found` or `python executable not found`: verify venv path, or pass `-Python .\.venv\Scripts\python` to scripts.
- `phase1 checkpoint not found`: run the previous phase first with the same seed, or provide a resume checkpoint.
- `opponent-policy=genome requires opponent_genome path`: set runtime `opponent_genome` or pass CLI override.
- `feature vector size mismatch`: sync NEAT input config with worker feature size (`scripts/configs/neat_feedforward*.ini`).

## Repository Map
- `src/`: game engine, AI policies, web UI
- `scripts/`: base/shared NEAT orchestration/evaluation scripts
- `scripts/configs/`: base NEAT runtime/config files
- `heuristic_tuning/`: heuristic tuning tools (`optuna_cl.py`, `optuna_gemini.py`, `optuna_nexg.py`, `optuna_gpt.py`, `optimizer_gpt.mjs`, `analyze_anticl_results_gpt.mjs`, `tune_anticl_go_gpt.mjs`)
- `logs/`: training and evaluation outputs

## Document SoT
- Agent behavior and guardrails: `AGENT_RULES.md`
- Game rules source: `docs/rules/GAME_RULES.md`
- NEAT source-of-truth runbook: `docs/neat/README.md`


