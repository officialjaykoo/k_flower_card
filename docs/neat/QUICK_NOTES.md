# NEAT Quick Notes

반복해서 확인한 운영 메모를 모아둔 문서다. 수시로 바뀌는 값은 실행 전에 설정 파일을 다시 보고, 자주 묻는 구조적 사실은 여기 기준으로 본다.

## 1. 현재 런타임 스냅샷

기준 파일:
- `scripts/configs/runtime_phase1.json`
- `scripts/configs/runtime_phase2.json`
- `scripts/configs/runtime_phase3.json`

현재 값:
- Phase 1: `generations = 50`, `games_per_genome = 200`
- Phase 1: `opponent_policy_mix = [hybrid_play(phase1_seed203,H-CL) 50, hybrid_play(phase2_seed204,H-CL) 50]`
- Phase 1: `fitness_win_weight = 0.99`, `fitness_gold_weight = 0.01`, `fitness_win_neutral_rate = 0.50`
- Phase 2: `opponent_policy_mix = [H-CL 50, H-GPT-AllButGo+CL 50]`
- Phase 2: `fitness_win_weight = 0.75`, `fitness_gold_weight = 0.25`, `fitness_win_neutral_rate = 0.50`
- Phase 3: `opponent_policy_mix = [hybrid_play(phase2_seed203,H-CL) 50, H-GPT-PlayMatch+CL 50]`
- Phase 3: `fitness_win_weight = 0.7`, `fitness_gold_weight = 0.3`, `fitness_win_neutral_rate = 0.50`

## 2. Fitness 식

코드 기준 위치:
- `scripts/neat_eval_worker.mjs` 983행 부근

실제 계산:

```text
fitness = fitnessGoldWeight * goldNorm + fitnessWinWeight * resultNorm
```

세부:

```text
goldNorm = tanh((meanGoldDelta - fitnessGoldNeutralDelta) / fitnessGoldScale)
```

```text
expectedResult = winRate + 0.5 * drawRate - lossRate
neutralExpectedResult = 2 * fitnessWinNeutralRate - 1
```

- `resultNorm`은 `expectedResult`를 중립점 기준 `[-1, 1]`로 정규화한 값이다.
- `fitnessWinWeight`, `fitnessGoldWeight`는 합이 1이 되도록 내부 정규화된다.
- `go_rate`, `go_count`, `go_fail_rate`는 리포트에는 남지만 fitness에 직접 들어가지 않는다.

## 3. 8:2 vs 7:3 vs 6:4 차이

`8:2`와 `7:3`은 단순히 골드 쪽으로 `10%p`를 더 싣는 수준이 아니다.

- 승률 비중: `0.8 -> 0.7`, 약 `12.5%` 감소
- 골드 비중: `0.2 -> 0.3`, 정확히 `50%` 증가
- `6:4`는 여기서 한 번 더 골드 쪽으로 이동한 값이다.
- 승률 비중: `0.7 -> 0.6`, 약 `14.3%` 감소
- 골드 비중: `0.3 -> 0.4`, 약 `33.3%` 증가
- `8:2`와 `6:4`를 직접 비교하면 승률 비중은 `25%` 감소하고, 골드 비중은 `100%` 증가한다.

같은 개체에 대해:

```text
fitness_8:2 - fitness_7:3 = 0.1 * (resultNorm - goldNorm)
fitness_7:3 - fitness_6:4 = 0.1 * (resultNorm - goldNorm)
fitness_8:2 - fitness_6:4 = 0.2 * (resultNorm - goldNorm)
```

의미:
- `resultNorm > goldNorm`이면 `8:2`가 유리하다.
- `goldNorm > resultNorm`이면 `7:3`이 유리하다.
- 따라서 `7:3`은 승률이 비슷한 후보들 사이에서 골드 많이 먹는 개체를 더 강하게 밀어준다.
- 같은 방향으로 `6:4`는 `7:3`보다도 골드형 개체를 더 강하게 밀어준다.

## 4. 정책 조합 메모

- 기본 `H-GPT`는 `go/stop`도 GPT가 한다.
- `H-GPT-AllButGo+CL`은 `go/stop`만 CL을 쓰고, 나머지는 GPT를 쓴다.
- `H-GPT-PlayMatch+CL`은 play/match만 GPT를 쓰고, 나머지는 CL을 쓴다.
- `H-GPT-PlayMatchGo+CL`은 play/match/go만 GPT를 쓰고, 나머지는 CL을 쓴다.

관련 파일:
- `src/ai/heuristicPolicyEngine.js`
- `src/ai/policies.js`

## 5. 순수 모델 액션 공간 메모

현재는 순수 NEAT 모델도 실제로 흔들기와 폭탄을 사용할 수 있다.

패치된 파일:
- `src/ai/modelPolicyEngine.js`
- `scripts/model_duel_worker.mjs`
- `scripts/neat_eval_worker.mjs`

핵심:
- `playing` 단계 후보 공간에 `shake_start:<cardId>`와 `bomb:<month>`를 넣는다.
- 흔들기는 UI 팝업처럼 `yes/no`를 따로 누르는 방식이 아니라, 휴리스틱과 같은 경로로 `declareShaking -> playTurn`을 바로 탄다.
- 폭탄은 `declareBomb(...)`를 직접 탄다.
- `go/stop`, 총통, 국진은 원래부터 별도 phase라 모델이 처리 가능했다.

즉 예전의 `Shake B = 0`, `Bomb B = 0`은 전략 문제가 아니라 액션 공간 누락 문제였다.

## 6. 최근 4개 휴리스틱 풀리그 메모

결과 파일:
- `logs/full_league/full_league_20260308_233716_top4_gpt_allbutgo_cl_20260308/full_league_summary.json`

정책:
- `H-CL`
- `H-J2`
- `H-GPT-AllButGo+CL`
- `H-NEXg`

순위:
- `H-CL`: `1582-1403-15`, 승률 `52.7%`, 평균 골드 `+342`
- `H-NEXg`: `1475-1504-21`, 승률 `49.2%`, 평균 골드 `-225`
- `H-J2`: `1473-1512-15`, 승률 `49.1%`, 평균 골드 `-358`
- `H-GPT-AllButGo+CL`: `1437-1548-15`, 승률 `47.9%`, 평균 골드 `+241`

`H-GPT-AllButGo+CL` 평균 골드 `+241`의 출처:
- vs `H-CL`: `+684.8`
- vs `H-NEXg`: `+196.0`
- vs `H-J2`: `-156.6`

즉 이 정책은 종합 승률은 낮지만, `H-CL` 상대로는 골드 기대값이 높게 나오는 편이다.

## 7. 해석 메모

- `continuous-series = 1`이면 승률과 평균 골드가 어긋날 수 있다.
- `transition_ready = false`인 winner는 다음 phase 기준 모델로 보기 애매할 수 있다.
- phase 상대 선택은 "누구를 더 자주 이기느냐"와 "누구 상대로 큰 골드를 먹느냐"를 구분해서 봐야 한다.

## 8. Seed204 학습 모드 메모

`play/match`만 모델이 하고, 나머지는 `H-CL`이 강제로 처리하는 학습 모드를 지원한다.

runtime 키:
- `control_policy_mode = "hybrid_play_match_only"`
- `control_policy_mode = "hybrid_go_stop_only"`
- `control_heuristic_policy = "H-CL"`

의미:
- 모델이 직접 맡는 것: `play`, `select-match`
- `H-CL`이 맡는 것: `go/stop`, 흔들기/폭탄, 총통, 국진, 기타 phase
- imitation 집계도 모델이 실제로 움직인 `play/match`만 센다.

`hybrid_go_stop_only` 의미:
- 모델이 직접 맡는 것: `play`, `select-match`, 흔들기/폭탄, 총통, 국진 등 `go/stop`을 제외한 나머지
- `H-CL`이 맡는 것: `go/stop`만
- imitation 집계도 모델이 실제로 움직인 `non-go-stop` 결정만 센다.

주의:
- 기존 `hybrid_play(model,H-CL)` 토큰은 실전/듀얼용 hybrid이고, 이 학습 모드는 eval worker의 control actor 평가 경로를 바꾸는 별도 스위치다.
- 메인 학습 런처 [phase_run.ps1](C:/k_flower_card/scripts/phase_run.ps1)에 `-ControlPolicyMode hybrid_play_match_only -ControlHeuristicPolicy H-CL`를 넘겨서 켠다.
