# K-Flower Card Developer Guide (NEAT)

## 결론
- 현재 학습/평가 경로는 `neat-python` 단일 경로다.
- 실행 진입점은 `scripts/run_neat_*.ps1`이며, 코어 러너는 `scripts/neat_train.py`다.
- 평가/대전 워커는 `scripts/neat_eval_worker.mjs`, `scripts/neat_duel_worker.mjs`를 사용한다.

## 1. 문서 범위
이 문서는 NEAT 도입 이후 개발자가 수정해야 할 파일 경계, 실행 절차, 산출물 해석 기준을 정의한다.
레거시 경로(Deep CFR, 커스텀 유전체 포맷)는 활성 런타임 범위에서 제외한다.

## 2. 단일 기준 파일(SoT)
### 2-1. 실행 스크립트
- `scripts/run_neat_train.ps1`: 학습 시작
- `scripts/run_neat_resume.ps1`: 체크포인트 재개
- `scripts/run_neat_eval.ps1`: 단일 유전체 평가
- `scripts/run_neat_duel.ps1`: 유전체 A/B 1000게임 대전

### 2-2. 코어 로직
- `scripts/neat_train.py`: NEAT 러너, 병렬 평가, 게이트/실패 감지, 로그 저장
- `scripts/neat_eval_worker.mjs`: 유전체 단일 평가(게임 반복, fitness 계산)
- `scripts/neat_duel_worker.mjs`: 유전체 A/B 대전 결과 계산

### 2-3. 설정
- `scripts/configs/neat_feedforward.ini`: neat-python 토폴로지/돌연변이 설정 (`num_inputs=47`)
- `configs/neat_runtime.json`: 기본 런타임 진입점(현재 phase2 i3_w6 확장)
- `configs/neat_runtime_i3_w6.json`: i3 6워커 프로필
- `configs/neat_runtime_i3_w7.json`: i3 7워커 비교 프로필
- `configs/neat/common.runtime.json`: 공통 런타임 파라미터
- `configs/neat/phases/phase1.runtime.json`: hybrid 게이트(초기 모방+승률)
- `configs/neat/phases/phase2.runtime.json`: win-rate only 게이트
- `configs/neat/phases/phase3.runtime.json`: 강화된 win-rate 게이트

## 3. 환경 준비
1. Python 가상환경 준비
2. `neat-python` 설치

```powershell
.venv\Scripts\python -m pip install neat-python
```

3. Node 의존성 준비

```powershell
npm ci
```

## 4. 실행 명령
### 4-1. 학습 시작
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_neat_train.ps1
```

### 4-2. 학습 Dry Run (시뮬레이션 없이 배선 확인)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_neat_train.ps1 -DryRun
```

### 4-3. 체크포인트 재개
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_neat_resume.ps1 -ResumeCheckpoint logs/neat_python/checkpoints/neat-checkpoint-50
```

### 4-4. 단일 유전체 평가
기본 게임 수는 `1000`이다.
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_neat_eval.ps1 -GenomePath logs/neat_python/models/winner_genome.json
```

### 4-5. 유전체 A/B 대전
프로젝트 규칙으로 게임 수는 고정 `1000`이다.
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_neat_duel.ps1 -GenomeAPath logs/runA/models/winner_genome.json -GenomeBPath logs/runB/models/winner_genome.json
```

## 5. 런타임 오버라이드 규칙
`run_neat_train.ps1`/`run_neat_resume.ps1`는 아래 핵심 오버라이드를 전달할 수 있다.

- `--generations`
- `--workers` (최소 2)
- `--games-per-genome`
- `--eval-timeout-sec`
- `--max-eval-steps`
- `--opponent-policy`
- `--checkpoint-every`
- `--seed`
- `--switch-seats` 또는 `--fixed-seats`
- `--fitness-gold-scale`
- `--fitness-win-weight`
- `--fitness-loss-weight`
- `--fitness-draw-weight`

실행 시 실제 적용값은 `logs/neat_python/run_summary.json`의 `runtime_effective`, `applied_overrides`로 확인한다.

## 6. Fitness/평가 지표
### 6-1. 최종 fitness
`neat_eval_worker.mjs` 기준:

```text
fitness = (mean_gold_delta / fitness_gold_scale)
        + (win_rate * fitness_win_weight)
        - (loss_rate * fitness_loss_weight)
        + (draw_rate * fitness_draw_weight)
```

### 6-2. 보조 지표
- `win_rate`, `loss_rate`, `draw_rate`
- `p10/p50/p90_gold_delta`
- `imitation_play/match/option_ratio`
- `imitation_weighted_score`
- `eval_time_ms`

## 7. 게이트/전환/실패 감지
게이트 상태는 `logs/neat_python/gate_state.json`에 저장된다.

- `gate_mode`: `hybrid` 또는 `win_rate_only`
- `ema_imitation`, `ema_win_rate`: 전환 판정용 EMA
- `transition_generation`: 전환 조건 충족 세대
- `failure_generation`: 실패 조건 충족 세대

세대별 상세는 `logs/neat_python/generation_metrics.ndjson`, 개체별 평가는 `logs/neat_python/eval_metrics.ndjson`에서 분석한다.

## 8. 산출물 디렉터리 기준
기본 출력 디렉터리: `logs/neat_python`

- `checkpoints/neat-checkpoint-*`
- `models/winner_genome.pkl`
- `models/winner_genome.json` (`neat_python_genome_v1`)
- `run_summary.json`
- `eval_metrics.ndjson`
- `generation_metrics.ndjson`
- `gate_state.json`
- `eval_failures.log`

## 9. 개발 시 필수 제약
- NEAT 경로 수정 시, 스크립트/설정/문서의 경로 일관성을 같이 갱신한다.
- `scripts/neat_eval_worker.mjs`와 `scripts/neat_duel_worker.mjs`는 엔진 API(`src/engine/index.js`, `src/engine/runner.js`) 변경에 민감하므로 함께 점검한다.
- 유전체 JSON 포맷은 `format_version = neat_python_genome_v1`만 지원한다.
- 저장 인코딩은 UTF-8 BOM을 유지한다.

## 10. 운영 정책 (시뮬레이션/평가)
- 시뮬레이션은 명시적 요청이 있을 때만 실행한다.
- 테스트성 시뮬레이션 게임 수는 `1000` 고정이다.
- 가능하면 멀티 워커 병렬 실행을 기본으로 사용한다.

## 11. 트러블슈팅
- `neat-python is not installed`:
  `neat-python`을 가상환경에 설치한다.
- `checkpoint not found`:
  `-ResumeCheckpoint` 경로를 절대경로 또는 작업경로 기준으로 재확인한다.
- 평가 실패 반복:
  `logs/neat_python/eval_failures.log` 마지막 레코드의 `reason`, `traceback`을 우선 확인한다.
- 입력 차원 오류(`feature vector size mismatch`):
  `scripts/configs/neat_feedforward.ini`의 `num_inputs`와 워커 feature 벡터 정의를 동기화한다.

## 12. 변경 체크리스트
1. `npm run build` 통과
2. `scripts/run_neat_train.ps1 -DryRun` 통과
3. `run_summary.json`에 의도한 `runtime_effective` 반영 확인
4. 게이트 산출물(`gate_state.json`, `generation_metrics.ndjson`) 생성 확인
5. 문서/스크립트/설정 경로 불일치 없음 확인

## 13. `neat_eval_worker` 47차원 피처 고정 규격
- `featureVector`는 단일 47차원만 지원한다. (`base/extended` 분기 없음)
- 설정은 `scripts/configs/neat_feedforward.ini`에서 `num_inputs=47`로 고정한다.
- `#16`은 `bakTotal`이 아니라 `self_score_total_norm = tanhNorm(scoreSelf.total, 10)`를 사용한다.
- `#24/#25`는 분리한다.
  - `#24`: 후보 월 기준 매칭 강도(연속값)
  - `#25`: 현재 후보 즉시 매칭 가능 여부(이진)
- `#47 candidate_month_known_ratio`는 공개정보 + 자기 손패만 사용하고 상대 손패는 사용하지 않는다.

### 13-1. 피처 목록 (1~47)
1. `phase === "playing"`
2. `phase === "select-match"`
3. `phase === "go-stop"`
4. `phase === "president-choice"`
5. `phase === "gukjin-choice"`
6. `phase === "shaking-confirm"`
7. `decisionType === "play"`
8. `decisionType === "match"`
9. `decisionType === "option"`
10. `deck/30`
11. `self_hand/10`
12. `opp_hand/10`
13. `self_goCount/5`
14. `opp_goCount/5`
15. `tanhNorm(scoreSelf.total - scoreOpp.total, 10)`
16. `tanhNorm(scoreSelf.total, 10)` (`self_score_total_norm`)
17. `legalCount/10`
18. `candidate.piValue` 정규화
19. `candidate_is_kwang`
20. `candidate_is_ribbon`
21. `candidate_is_five`
22. `candidate_is_junk`
23. `candidate_is_double_pi` (`junk&&piValue>=2` 또는 `I0`)
24. `match_opportunity_density` (후보 월 기준)
25. `immediate_match_possible`
26. `optionCode`
27. `selfGwangCount/5`
28. `oppGwangCount/5`
29. `selfPiCount/20`
30. `oppPiCount/20`
31. `self_godori_progress/3`
32. `opp_godori_progress/3`
33. `self_cheongdan_progress/3`
34. `opp_cheongdan_progress/3`
35. `self_hongdan_progress/3`
36. `opp_hongdan_progress/3`
37. `self_chodan_progress/3`
38. `opp_chodan_progress/3`
39. `self_can_stop` (`scoreSelf.total >= 7`)
40. `opp_can_stop` (`scoreOpp.total >= 7`)
41. `has_shake`
42. `current_multiplier` 정규화
43. `has_bomb`
44. `self_pi_bak_risk`
45. `self_gwang_bak_risk`
46. `self_mong_bak_risk`
47. `candidate_month_known_ratio` (공개정보 + 자기 손패, 상대 손패 제외)
