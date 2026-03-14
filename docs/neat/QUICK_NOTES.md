# NEAT Quick Notes

반복해서 확인한 운영 메모를 모아둔 문서다. 수시로 바뀌는 값은 실행 전에 설정 파일을 다시 보고, 자주 묻는 구조적 사실은 여기 기준으로 본다.

## 1. 현재 런타임 스냅샷

기준 파일:
- `scripts/configs/runtime_phase1.json`

현재 값:
- `generations = 30`, `games_per_genome = 200`
- `opponent_policy_mix = [hybrid_play(phase1_seed208,H-CL) 50, hybrid_play(phase1_seed204,H-CL) 50]`
- `fitness_win_weight = 0.99`, `fitness_gold_weight = 0.01`, `fitness_win_neutral_rate = 0.50`

## 2. Fitness 식

코드 기준 위치:
- `scripts/neat_eval_worker.mjs` 983행 부근

현재 scalar v2 계산:

```text
base = fitnessGoldWeight * goldNorm + fitnessWinWeight * resultNorm

fitness = base
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
- `bankruptScore`와 파산 count/rate는 경기 결과 진단용 통계다.
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

## 4. 순수 모델 액션 공간 메모

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

## 5. 최근 4개 휴리스틱 풀리그 메모

결과 파일:
- `logs/full_league/full_league_20260308_233716_top4_gpt_allbutgo_cl_20260308/full_league_summary.json`

정책:
- `H-CL`
- `H-J2`
- `H-GPT`
- `H-NEXg`

순위:
- `H-CL`: `1582-1403-15`, 승률 `52.7%`, 평균 골드 `+342`
- `H-NEXg`: `1475-1504-21`, 승률 `49.2%`, 평균 골드 `-225`
- `H-J2`: `1473-1512-15`, 승률 `49.1%`, 평균 골드 `-358`
- `H-GPT`: 실운영 비교 대상에서 제외된 구형 혼합 정책 대신 순수 GPT 계열만 유지

## 6. 해석 메모

- `continuous-series = 1`이면 승률과 평균 골드가 어긋날 수 있다.
- `transition_ready = false`인 winner는 다음 phase 기준 모델로 보기 애매할 수 있다.
- phase 상대 선택은 "누구를 더 자주 이기느냐"와 "누구 상대로 큰 골드를 먹느냐"를 구분해서 봐야 한다.

## 7. Control 평가 메모

- 현재 `neat_eval_worker.mjs`의 control actor는 `pure_model`만 사용한다.
- control 전용 hybrid mode(`hybrid_play_match_only`, `hybrid_go_stop_only`, `hybrid_option_only`)는 제거됐다.
- 상대 정책 spec은 `heuristic`, `model`, `hybrid_play(model,heuristic)` 3종만 지원한다.

## 8. 공식 문서 기준 개조 메모

공식 문서 링크:
- `Customizing Behavior`
- `Genome Interface`
- `Reproduction Interface`
- `Configuration file description`
- `FAQ`
- `Module summaries`

정리:
- `Genome / Reproduction / SpeciesSet / Stagnation` 교체는 공식 확장 포인트다.
- 맞고용 NEAT 개조는 문서 기준으로도 정식 경로다.
- 다만 진짜 `2축 selection`은 결국 `Reproduction`과 필요 시 `Population.run`까지 손댈 가능성이 크다.

현재 repo 반영 상태:
- `stock neat-python 2.0` 기반 유지
- genome-level threshold gene 4개 추가
  - `go_stop_threshold`
  - `shaking_threshold`
  - `president_threshold`
  - `gukjin_threshold`
- output head는 `10개`
  - `play`, `match`, `go`, `stop`, `shaking_yes`, `shaking_no`, `president_stop`, `president_hold`, `gukjin_five`, `gukjin_junk`

현재 설계 해석:
- 1차는 `CustomGenome`만으로 decision boundary를 넣는다.
- 2차부터 `CustomReproduction`이 핵심이다.
- `behavior-aware speciation`이 필요하면 `SpeciesSet`까지 간다.

현재 `scripts/configs/neat_feedforward.ini` 프로필은 `중간형`으로 본다.
- output `10개`와 threshold gene `4개`를 감안해 구조 mutation은 약간 보수적으로 낮췄다.
- 대신 `pop_size`, `compatibility_threshold`, `max_stagnation`, `min_species_size`는 조금 올려서 장기 학습 안정성을 확보한다.
- 목적은 `과도한 판 흔들기`가 아니라 `threshold gene이 천천히 누적 적응`되게 하는 것이다.
