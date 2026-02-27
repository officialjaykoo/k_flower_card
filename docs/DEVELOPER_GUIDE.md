# K-Flower Card Developer Guide (NEAT)

## 결론
- 현재 학습/평가 기준 파이프라인은 `scripts/neat_train.py -> scripts/neat_eval_worker.mjs -> scripts/model_duel_worker.mjs`다.
- Phase 1/2 실행·평가는 공통 래퍼(`scripts/phase_run.ps1`, `scripts/phase_eval.ps1`)로 통합되었다.
- 모든 런타임 설정은 `scripts/configs/runtime_phase*.json`을 단일 기준(Source of Truth)으로 사용한다.

## 1. 문서 범위
이 문서는 NEAT 기반 학습/평가 운영 기준, 파일 경계, 실행 명령, 산출물 해석 규칙을 정의한다.
레거시 실행 경로(`run_neat_*`, 루트 `configs/`)는 현재 기준에서 제외한다.

## 2. 현재 기준 파일 맵 (SoT)
### 2-1. 오케스트레이션 스크립트 (`scripts/`)
- `phase_run.ps1`: Phase 1/2 공통 학습 실행 래퍼
- `phase_eval.ps1`: Phase 1/2 공통 평가 실행 래퍼
- `phase1_run.ps1`, `phase2_run.ps1`: 공통 학습 래퍼 위임 진입점
- `phase1_eval.ps1`, `phase2_eval.ps1`: 공통 평가 래퍼 위임 진입점
- `phase3_run.ps1`, `phase3_eval.ps1`: Phase 3 학습/평가
- `phase4_run.ps1`, `phase4_eval.ps1`: Phase 4 학습/평가

### 2-2. 코어 실행기
- `neat_train.py`: NEAT 러너, 병렬 평가, 게이트/실패 감지, 체크포인트/요약 저장
- `neat_eval_worker.mjs`: 단일 유전체 평가(게임 반복, fitness 계산, imitation 계산)
- `model_duel_worker.mjs`: 휴리스틱/NEAT 모델 공용 대전 실행기 + kibo/dataset 출력

### 2-3. 설정 (`scripts/configs/`)
- `neat_feedforward.ini`: NEAT 토폴로지/변이 설정 (`num_inputs=47`)
- `runtime_phase1.json`
- `runtime_phase2.json`
- `runtime_phase3.json`
- `runtime_phase4.json`

## 3. 파이프라인 흐름
### 3-1. Phase 1
1. `phase1_run.ps1 -Seed <N>`
2. `phase1_eval.ps1 -Seed <N>`

### 3-2. Phase 2
1. `phase2_run.ps1 -Seed <N>`
2. `phase2_eval.ps1 -Seed <N>`

참고:
- `phase_run.ps1`는 Phase 2 실행 시 Phase 1 체크포인트를 자동 탐색한다.
- 우선순위: `runtime_phase1.json.generations`에 해당하는 체크포인트 -> 없으면 최신 세대 체크포인트.

### 3-3. Phase 3/4
- 현재 `phase3_run.ps1`, `phase4_run.ps1`은 resume 체크포인트 세대가 스크립트에 하드코딩되어 있다.
- Phase 3/4 세대 수 변경 시, 해당 run 스크립트의 resume 세대(`gen99`, `gen199`)와 `--base-generation`을 같이 갱신해야 한다.

## 4. 실행 명령
### 4-1. 학습
```powershell
powershell -ExecutionPolicy Bypass -File scripts/phase1_run.ps1 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase2_run.ps1 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase3_run.ps1 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase4_run.ps1 -Seed 9
```

### 4-2. 평가 (고정 1000게임)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/phase1_eval.ps1 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase2_eval.ps1 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase3_eval.ps1 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase4_eval.ps1 -Seed 9
```

### 4-3. 휴리스틱 대전 (고정 1000게임)
```powershell
node scripts/model_duel_worker.mjs --human heuristic_v5 --ai phase4_seed5 --games 1000 --seed v5_vs_phase4s5 --first-turn-policy alternate --continuous-series 1 --max-steps 600
```

## 5. 런타임 키 규칙
### 5-1. 공통 핵심 키
- `generations`, `eval_workers`, `games_per_genome`
- `max_eval_steps`, `eval_timeout_sec`
- `opponent_policy`, (`opponent_genome` when policy=`genome`)
- `fitness_gold_scale`, `fitness_win_weight`, `fitness_loss_weight`, `fitness_draw_weight`
- `gate_mode`, `gate_ema_window`, `transition_*`, `failure_*`

### 5-2. Phase 1/2 평가 통과 기준
`phase_eval.ps1` 우선순위:
1. `eval_pass_win_rate_min`, `eval_pass_mean_gold_delta_min`
2. 없으면 `transition_ema_win_rate`, `transition_mean_gold_delta_min`
3. 없으면 기본값 `win_rate >= 0.48`, `mean_gold_delta >= 100`

### 5-3. 모순 방지 규칙
- `failure_generation_min`은 해당 phase `generations`보다 크지 않게 유지한다.
- `checkpoint_every`는 세대 수 대비 적절히 설정한다(체크포인트 누락 방지).

## 6. 산출물 디렉터리
### 6-1. Phase별 출력 루트
- `logs/NEAT/neat_phase1_seed<Seed>`
- `logs/NEAT/neat_phase2_seed<Seed>`
- `logs/NEAT/neat_phase3_seed<Seed>`
- `logs/NEAT/neat_phase4_seed<Seed>`

### 6-2. 핵심 산출물
- `checkpoints/neat-checkpoint-gen*`
- `models/winner_genome.json`
- `run_summary.json`
- `gate_state.json`
- `generation_metrics.ndjson`
- `eval_metrics.ndjson`
- `eval_failures.log`
- `phase*_eval_1000.json` (평가 실행 시)
- `phase*_pass_state.json` (phase1/2 평가 실행 시)

## 7. Worker 동작 메모
상세 포맷 문서:
- `docs/KIBO_DATASET_GUIDE.md` (kibo/dataset 구조, 필드 의미, unresolved 해석)

### 7-1. `neat_eval_worker.mjs`
- feature vector는 47차원 고정.
- `opponent_policy=heuristic_v6`일 때 내부 fast tuning 파라미터를 적용해 평가 시간을 줄인다.
- teacher dataset cache(`--teacher-dataset-cache`)가 있으면 imitation 계산 소스를 cache로 전환한다.

### 7-2. `model_duel_worker.mjs`
- `--games`는 `1000` 이상만 허용된다. (기본값 `1000`)
- 결과 report(`result.json`)는 항상 1개 저장된다.
- 기본 출력 폴더는 `logs/model_duel/<human>_vs_<ai>_<YYYYMMDD>/` 규칙을 따른다.
- kibo/dataset은 옵션으로만 저장된다.
- `--human`, `--ai` 입력은 정책 키 또는 `phase4_seed5` 형식 모두 지원한다.

### 7-3. `model_duel_worker.mjs` 옵션 상세
- 공통 문법: `--key value` 또는 `--key=value` 둘 다 지원.
- `--human` (필수): human 슬롯 모델 지정 (`src/ai/policies.js` 정책 키 또는 `phase4_seed5`).
- `--ai` (필수): ai 슬롯 모델 지정 (`src/ai/policies.js` 정책 키 또는 `phase4_seed5`).
- `--games` (선택, 기본 `1000`): `1000` 이상만 허용.
- `--seed` (선택, 기본 `model-duel`): 실행 시드 문자열.
- `--max-steps` (선택, 기본 `600`): 게임당 최대 스텝. 최소값은 `20`.
- `--first-turn-policy` (선택, 기본 `alternate`): `alternate` 또는 `fixed`.
- `--fixed-first-turn` (선택, 기본 `human`): `fixed`일 때 `human`만 허용.
- `--continuous-series` (선택, 기본 `1`): `1=true`, `2=false`.
- `--result-out` (선택): report JSON 저장 경로. 미지정 시 자동 생성.
- `--kibo-detail` (선택, 기본 `none`): `none|lean|full`.
  - `lean`: 턴별 카드 배열 대신 `handsCount/boardCount/deckCount` 중심으로 기록(파일 작음, 대량 실험 권장).
  - `full`: 턴별 `hands/board/deck` 카드 배열까지 기록(리플레이 분석용, 파일 큼).
- `--kibo-detail`이 `none`이고 `--kibo-out`만 지정되면 내부적으로 `lean`으로 동작한다.
- `--kibo-detail`이 `lean/full`이고 `--kibo-out` 미지정이면 report 폴더 아래 `<seed>_kibo.jsonl` 자동 생성.
- `--kibo-out` (선택): 게임 단위 JSONL 출력 경로. 각 줄에 `game_index`, `seed`, `first_turn`, `human`, `ai`, `winner`, `result`, `kibo_detail`, `kibo` 저장.
- `--dataset-out` (선택): 의사결정 후보 단위 JSONL 출력 경로.
  - `auto`를 주면 report 폴더 아래 `<seed>_dataset.jsonl` 자동 생성.
  - 명시 경로를 주면 해당 경로로 저장.
- `--dataset-actor` (선택, 기본 `all`): 데이터셋 기록 actor 필터. `all|human|ai`.
- `--dataset-decision-types` (선택, 기본 `all`): 데이터셋 기록 decision_type 필터. `all|play|match|option` 또는 CSV(`play,option`).
- `--dataset-option-candidates` (선택, 기본 `all`): `decision_type=option`에서 후보 action 필터.
  - 허용값: `all` 또는 CSV(`go,stop,shaking_yes,shaking_no,president_stop,president_hold,five,junk`)
  - 별칭: `go-stop` (`go,stop`과 동일)
  - 이 옵션 지정 시 내부적으로 `option` decision만 dataset에 기록.
- GO/STOP만 dataset으로 수집하려면:
  - `--dataset-decision-types option --dataset-option-candidates go-stop`
- unresolved는 별도 출력 옵션 없이 dataset 사용 시 내부 통계(`dataset_unresolved_decisions`)로만 계산한다.
- report JSON의 `result_out`, `kibo_out`, `dataset_out` 경로 필드는 상대경로(`logs/...`)로 기록된다.
- stdout 결과: 마지막 줄에 고정 스키마의 compact summary JSON 1줄만 출력된다.
  - 상세 지표(`seat_split_*`, unresolved 상세 분포 등)는 `result_out` 파일에서 확인한다.
- 대표 실행 예시:
```powershell
node scripts/model_duel_worker.mjs --human heuristic_v5 --ai phase4_seed5 --games 1200 --seed v5_vs_phase4s5_1200 --first-turn-policy alternate --continuous-series 1 --max-steps 600 --kibo-detail full --dataset-out auto
```

## 8. 트러블슈팅
- `python not found: .venv\Scripts\python.exe`
  - 가상환경 생성/활성화 후 `neat-python` 설치 확인.
- `phase1 checkpoint not found` (Phase 2)
  - 먼저 같은 Seed로 Phase 1 학습 결과가 생성되어야 한다.
- `opponent-policy=genome requires opponent_genome path`
  - runtime 또는 CLI에 opponent genome 경로를 설정한다.
- `feature vector size mismatch`
  - `scripts/configs/neat_feedforward.ini`의 `num_inputs`와 worker feature 정의를 맞춘다.

## 9. 운영 체크리스트
1. 경로는 항상 `scripts/` + `scripts/configs/` 기준으로 유지한다.
2. 시뮬레이션/평가 테스트는 1000게임 규칙을 유지한다.
3. 설정 변경 시 문서와 phase 래퍼 스크립트를 같이 갱신한다.
4. 결과 검증은 `run_summary.json`, `gate_state.json`, `phase*_eval_1000.json` 순으로 확인한다.
5. 풀리그 산출물은 `logs/model_duel/full_league/` 경로에 저장한다.
6. 저장 인코딩은 UTF-8 BOM을 유지한다.
