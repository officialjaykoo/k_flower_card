# K-Flower Card NEAT Runbook

## 결론
- 현재 학습/평가 기준 파이프라인은 `scripts/neat_train.py -> scripts/neat_eval_worker.mjs -> scripts/model_duel_worker.mjs`다.
- Phase 실행·평가는 단일 진입점(`scripts/phase_run.ps1`, `scripts/phase_eval.ps1`)으로 통합되었다.
- classic NEAT 런타임 기준은 `scripts/configs/runtime_phase1.json`이다.
- `K-HyperNEAT` 로컬 실험은 `experiments/k_hyperneat_matgo/configs/runtime_phase1.json`을 별도로 쓴다.

빠른 참고:
- 반복 확인하는 설정/fitness/정책 메모는 `docs/neat/QUICK_NOTES.md`

## 1. 문서 범위
이 문서는 NEAT 기반 학습/평가 운영 기준, 파일 경계, 실행 명령, 산출물 해석 규칙을 정의한다.
레거시 실행 경로(`run_neat_*`, 루트 `configs/`)는 현재 기준에서 제외한다.

## 2. 현재 기준 파일 맵 (SoT)
### 2-1. 오케스트레이션 스크립트 (`scripts/`)
- `phase_run.ps1`: Phase 1/2/3 공통 학습 실행 진입점 (`-Phase`, `-Seed` 필수)
- `phase_eval.ps1`: Phase 1/2/3 공통 평가 실행 진입점 (`-Phase`, `-Seed` 필수)

### 2-2. 코어 실행기
- `neat_train.py`: NEAT 러너, 병렬 평가, 게이트/실패 감지, 체크포인트/요약 저장
- `neat_eval_worker.mjs`: 단일 유전체 평가(게임 반복, fitness 계산, imitation 계산)
- `model_duel_worker.mjs`: 휴리스틱/NEAT 모델 공용 대전 실행기 + kibo/dataset 출력

### 2-3. 설정 (`scripts/configs/`)
- `neat_feedforward.ini`: 10D 프로필용 NEAT 토폴로지/변이 설정 (`num_inputs=10`)
- `neat_feedforward_race2.ini`: `race2` 전용 NEAT 토폴로지/변이 설정 (`num_inputs=2`)
- `neat_feedforward_race5.ini`: `race5` 전용 NEAT 토폴로지/변이 설정 (`num_inputs=5`)
- `neat_feedforward_race6.ini`: `race6` 전용 NEAT 토폴로지/변이 설정 (`num_inputs=6`)
- `neat_feedforward_race7.ini`: `race7` 전용 NEAT 토폴로지/변이 설정 (`num_inputs=7`)
- `neat_feedforward_race8.ini`: `race8` 전용 NEAT 토폴로지/변이 설정 (`num_inputs=8`)
- `neat_feedforward_hand7.ini`: `hand7` 전용 NEAT 토폴로지/변이 설정 (`num_inputs=7`)
- `runtime_phase1.json`: 모든 phase/classic 런의 공통 runtime 기준 파일

## 3. 파이프라인 흐름
### 3-1. Phase 1
1. `phase_run.ps1 -Phase 1 -Seed <N>`
2. `phase_eval.ps1 -Phase 1 -Seed <N>`

### 3-2. Phase 2/3
1. `phase_run.ps1 -Phase 2 -Seed <N>`
2. `phase_eval.ps1 -Phase 2 -Seed <N>`
3. `phase_run.ps1 -Phase 3 -Seed <N>`
4. `phase_eval.ps1 -Phase 3 -Seed <N>`

참고:
- phase 번호는 출력 경로/재개 흐름에만 쓰고, runtime 값은 모두 `runtime_phase1.json`에서 읽는다.
- `phase_run.ps1`는 Phase 2/3 실행 시 이전 phase 체크포인트를 자동 탐색한다.

## 4. 실행 명령
### 4-1. 학습
```powershell
powershell -ExecutionPolicy Bypass -File scripts/phase_run.ps1 -Phase 1 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase_run.ps1 -Phase 2 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase_run.ps1 -Phase 3 -Seed 9
```

### 4-2. 평가 (고정 1000게임)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/phase_eval.ps1 -Phase 1 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase_eval.ps1 -Phase 2 -Seed 9
powershell -ExecutionPolicy Bypass -File scripts/phase_eval.ps1 -Phase 3 -Seed 9
```

### 4-3. 휴리스틱 대전 (고정 1000게임)
```powershell
node scripts/model_duel_worker.mjs --human heuristic_h_cl --ai phase3_seed5 --games 1000 --seed cl_vs_phase3s5 --first-turn-policy alternate --continuous-series 1 --max-steps 600
```

## 5. 런타임 키 규칙
### 5-1. 공통 핵심 키
- `generations`, `eval_workers`, `games_per_genome`
- `max_eval_steps`, `eval_timeout_sec`
- `opponent_policy`, (`opponent_genome` when policy=`genome`)
- `split_head_output0_inputs`, `split_head_output1_inputs` (둘 다 채우면 vendor/neat에서 output별 허용 입력을 강제한다)
- `fitness_gold_scale`, `fitness_gold_neutral_delta`, `fitness_win_weight`, `fitness_gold_weight`, `fitness_win_neutral_rate`
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

### 6-2. 핵심 산출물
- `checkpoints/neat-checkpoint-gen*`
- `models/winner_genome.json`
- `run_summary.json`
- `gate_state.json`
- `generation_metrics.ndjson`
- `eval_metrics.ndjson`
- `eval_failures.log`
- `phase*_eval_1000.json` (평가 실행 시)
- `phase*_pass_state.json` (phase 평가 실행 시)

## 7. Worker 동작 메모
상세 포맷 문서:
- `docs/data/KIBO_DATASET_GUIDE.md` (kibo/dataset 구조, 필드 의미, unresolved 해석)

### 7-1. `neat_eval_worker.mjs`
- active feature vector 기본값은 `race8` 8차원이다.
- 학습 시 `--feature-profile race2|race5|race6|race7|race8|race20|hand7|hand10|material10|race10|oracle10|oracle10v2` 으로 CLI override 가능하다.
- `race2`는 `candidate_public_known_ratio + opp_stop_pressure_norm`만 쓰는 2D 극단 축소 프로필이다.
- `race5`는 `material10` winner에서 강하게 살아남은 `play 3 + option 2` 신호만 모은 5D 압축 프로필이다.
- `race6`는 `race5`에 `candidate_combo_gain`을 더한 6D 압축 프로필이다.
- `race7`는 `material10`에서 `current_multiplier_norm`, `opp_combo_threat_norm`, `score_diff_tanh`를 제거한 7D 압축 프로필이다.
- `race8`는 `material10`에서 `opp_combo_threat_norm`, `score_diff_tanh`만 제거하고 `current_multiplier_norm`은 마지막 보조 슬롯으로 남긴 8D 압축 프로필이다.
- `material10` alternate builder도 `10D` 기준으로 유지한다.
- `race10`은 이제 `action 5 + option/global 5`의 split-head 혼합 10D 프로필이다.
- `oracle10`은 원래 `play 6 + option 4`에서 출발한 10D residual 프로필이다.
- `oracle10v2`는 그 10D 구성을 유지하되, `candidate_pi_value`/`combined_post_block_norm` 대신 `candidate_safe_discard`/`opp_stop_pressure_norm`을 넣은 개정 residual 프로필이다.
- 현재 `oracle10`/`oracle10v2`는 이름만 유지된 residual feature profile이다.
- 오라클 base scorer 실험 코드는 제거했고, 지금은 `play/option` 모두 `genome residual`만으로 선택한다.
- `hand10`은 기존 10D alternate builder로 유지한다.
- winner playoff 기본 tie-break:
  - `win_rate` 차이 `<= 1.0%p` 는 동률
  - `mean_gold_delta` 차이 `<= 100` 은 동률
  - `go_take_rate` 차이 `<= 2.0%p` 는 동률
  - 동률 구간에서는 `go_take_rate`, 그다음 `go_fail_rate`, 마지막 `fitness` 순으로 비교한다.
- `opponent_policy=heuristic_h_gpt`일 때 내부 fast tuning 파라미터를 적용해 평가 시간을 줄인다.
- teacher dataset cache(`--teacher-dataset-cache`)가 있으면 imitation 계산 소스를 cache로 전환한다.

### 7-2. `model_duel_worker.mjs`
- `--games`는 양의 정수만 허용된다. (기본값 `1000`)
- 결과 report(`result.json`)는 항상 1개 저장된다.
- 기본 출력 폴더는 `logs/model_duel/<human>_vs_<ai>_<YYYYMMDD>/` 규칙을 따른다.
- kibo/dataset은 옵션으로만 저장된다.
- `--human`, `--ai` 입력은 정책 키, `phase3_seed5`, `hybrid_play(...)`, `k_hyperneat(path.json)`, 직접 `.json` 경로를 지원한다.

### 7-3. `model_duel_worker.mjs` 옵션 상세
- 공통 문법: `--key value` 또는 `--key=value` 둘 다 지원.
- `--human` (필수): human 슬롯 모델 지정 (`src/ai/policies.js` 정책 키 또는 `phase3_seed5`).
- `--ai` (필수): ai 슬롯 모델 지정 (`src/ai/policies.js` 정책 키 또는 `phase3_seed5`).
  - 예: `hybrid_play(phase1_seed208,H-CL)`
  - 예: `k_hyperneat(experiments/k_hyperneat_matgo/smoke_runtime.json)`
  - 예: `experiments/k_hyperneat_matgo/smoke_runtime.json`
- `--games` (선택, 기본 `1000`): 양의 정수만 허용.
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
node scripts/model_duel_worker.mjs --human heuristic_h_cl --ai phase3_seed5 --games 1000 --seed cl_vs_phase3s5_1000 --first-turn-policy alternate --continuous-series 1 --max-steps 600 --kibo-detail full --dataset-out auto
```

## 8. 트러블슈팅
- `python not found: .venv\Scripts\python.exe`
  - 가상환경 생성/활성화 후 `neat-python` 설치 확인.
- `phase1 checkpoint not found`
  - 먼저 같은 Seed로 이전 phase 학습 결과가 생성되어야 한다.
- `opponent-policy=genome requires opponent_genome path`
  - runtime 또는 CLI에 opponent genome 경로를 설정한다.
- `k_hyperneat(path.json)` 또는 직접 `.json` 경로
  - 현재는 `format_version = neat_python_genome_v1` 또는 `k_hyperneat_executor_v1` 둘 다 모델 로더에서 허용한다.
- `feature vector size mismatch`
  - `race2`는 `scripts/configs/neat_feedforward_race2.ini`
  - `race5`는 `scripts/configs/neat_feedforward_race5.ini`
  - `race6`는 `scripts/configs/neat_feedforward_race6.ini`
  - `race7`는 `scripts/configs/neat_feedforward_race7.ini`
  - `race8`는 `scripts/configs/neat_feedforward_race8.ini`
  - `hand7`은 `scripts/configs/neat_feedforward_hand7.ini`
- `hand10/material10/race10/oracle10/oracle10v2`은 `scripts/configs/neat_feedforward.ini`
  - 프로필과 `num_inputs`를 맞춘다.

## 9. 공식 확장 포인트 기준 현재 방향
공식 문서 기준 핵심 링크:
- `Customizing Behavior`: `https://neat-python.readthedocs.io/en/latest/customization.html`
- `Genome Interface`: `https://neat-python.readthedocs.io/en/latest/genome-interface.html`
- `Reproduction Interface`: `https://neat-python.readthedocs.io/en/latest/reproduction-interface.html`
- `Configuration file description`: `https://neat-python.readthedocs.io/en/latest/config_file.html`
- `FAQ`: `https://neat-python.readthedocs.io/en/latest/faq.html`
- `Module summaries`: `https://neat-python.readthedocs.io/en/latest/module_summaries.html`

해석:
- `CustomGenome`, `CustomReproduction`, `CustomSpeciesSet`, `CustomStagnation` 교체는 공식 확장 포인트다.
- 따라서 맞고용 NEAT 개조는 억지 포크가 아니라 문서가 허용한 정식 개조 경로로 보는 게 맞다.
- 다만 라이브러리 기본 루프는 여전히 scalar `fitness` 중심이므로, 진짜 2축 selection은 `Reproduction`만이 아니라 `Population.run` 또는 수동 generation loop까지 손볼 가능성이 있다.

현재 repo 기준 1차 방향:
- `stock neat-python 2.0` 위에 genome-level decision parameter만 얹는다.
- 현재 반영된 파라미터:
  - `go_stop_threshold`
  - `shaking_threshold`
  - `president_threshold`
  - `gukjin_threshold`
- 이 4개는 `vendor/neat/genome.py`의 per-genome scalar attribute로 추가되어 있고, export 시 `decision_params`에 저장된다.
- 런타임은 `src/ai/modelPolicyEngine.js`에서 typed decision pair를 해석할 때 threshold를 사용한다.

현재 output head 기준:
- `play`
- `match`
- `go`
- `stop`
- `shaking_yes`
- `shaking_no`
- `president_stop`
- `president_hold`
- `gukjin_five`
- `gukjin_junk`

의미:
- `play`와 `match`는 일반 candidate scoring head다.
- `go/stop`, `shaking yes/no`, `president stop/hold`, `gukjin five/junk`는 각각 대응하는 threshold gene으로 최종 선택 경계를 조정한다.
- 즉 지금 단계는 `CustomGenome + typed output heads`까지 반영된 상태다.

다음 확장 우선순위:
1. `CustomReproduction`
   - outcome 축 + decision quality 축을 같이 쓰는 selection
   - archive 주입, quota, multi-criterion survivor rule 같은 로직은 여기서 다루는 게 맞다.
2. `CustomSpeciesSet`
   - 유전 거리만이 아니라 행동 거리까지 speciation에 반영
   - 특히 `go/stop` 경계형 개체 보호가 목적이면 여기가 자연스럽다.
3. `CustomStagnation`
   - 단순 outcome stagnation이 아니라 판단 품질 개선 정체도 같이 반영
4. 필요 시 `Custom Population.run`
   - scalar best-genome 전제에서 벗어난 수동 generation loop

현재 판단:
- 1차는 `Genome`만 건드려서 threshold/typed output을 넣는 것이 가장 싸고 효과 검증이 쉽다.
- 2축 NEAT를 진짜로 하려면 중심은 결국 `Reproduction/Population`이다.
- reporter 훅만으로 해결하려는 방향은 한계가 있다.

## 10. 운영 체크리스트
1. 경로는 항상 `scripts/` + `scripts/configs/` 기준으로 유지한다.
2. 시뮬레이션/평가 테스트는 1000게임 규칙을 유지한다.
3. 설정 변경 시 문서와 phase 래퍼 스크립트를 같이 갱신한다.
4. 결과 검증은 `run_summary.json`, `gate_state.json`, `phase*_eval_1000.json` 순으로 확인한다.
5. 풀리그 산출물은 `logs/full_league/` 경로에 저장한다.
6. 저장 인코딩은 UTF-8 BOM을 유지한다.


