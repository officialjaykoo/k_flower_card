# PPO Workspace

PPO 전용 작업 경로입니다. 기존 `scripts/*` NEAT 라인과 분리되어 동작합니다.

## Files
- `ppo/scripts/ppo_env_bridge.mjs` : Matgo 엔진을 PPO 학습용 reset/step 환경으로 노출
- `ppo/scripts/train_ppo.py` : PyTorch Masked PPO 학습 루프
- `ppo/configs/runtime_phase1_ppo.json` : phase1 **self-play** 기본 런타임 설정(seed 90)
- `ppo/run_ppo.ps1` : 실행 래퍼

## Run
```powershell
.\ppo\run_ppo.ps1 -RuntimeConfig .\ppo\configs\runtime_phase1_ppo.json
```

seed만 바꿔 실행:
```powershell
.\ppo\run_ppo.ps1 -RuntimeConfig .\ppo\configs\runtime_phase1_ppo.json -Seed 91
```
`-Seed`를 주면 `output_dir_template`, `resume_checkpoint_template`의 `{seed}`가 자동 치환됩니다.

## Modes
- `training_mode = selfplay` : 동일 정책이 양쪽 턴을 모두 플레이 (현재 기본)
- `training_mode = single_actor` : 한쪽만 학습 + 휴리스틱 상대

## Phase Connection
- `runtime_phase2_ppo.json`은 `resume_checkpoint = logs/PPO/vs_hcl_phase1_seed90/latest.pt`로 고정 연결되어 있습니다.
- phase2 이상에서 resume가 비어 있으면 학습이 fail-fast로 중단됩니다.
- `-Seed`를 사용하면 `resume_checkpoint_template`도 함께 치환되어 같은 seed의 phase1 체크포인트를 참조합니다.
  예: `-Seed 91` -> `logs/PPO/vs_hcl_phase1_seed91/latest.pt`

## Duel (PPO vs v5)
1000게임 대결 실행:
```powershell
.\ppo\run_duel_ppo.ps1 -RuntimeConfig .\ppo\configs\duel_ppo_vs_v5_1000.json
```

## Requirements
- Python 3.10+
- `torch` 설치 필요
- Node.js 설치 필요
