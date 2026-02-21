# Last 2000 Games Matchup Summary (tuneA)

- Generated date: 2026-02-21
- Source reports:
- `logs/heuristic_v4_vs_v3_1000_tuneA-report.json`
- `logs/heuristic_v3_vs_v4_1000_tuneA-report.json`
- Source logs:
- `logs/heuristic_v4_vs_v3_1000_tuneA.jsonl`
- `logs/heuristic_v3_vs_v4_1000_tuneA.jsonl`
- Match count: 1000 + 1000 = 2000 (first-turn balanced)

## 1) Wins

- V4 wins: 1015
- V3 wins: 979
- Draws: 6
- V4 win rate: 50.75%
- V3 win rate: 48.95%

## 2) Average score

- V4 average score: 6.6095
- V3 average score: 5.8055

## 3) Gold

- V4 average gold: 1,103,616.65
- V3 average gold: 896,383.35
- V4 - V3 average gold delta: +207,233.30 per game
- V4 - V3 cumulative delta over 2000 games: +414,466,600

## 4) GO/STOP and comeback losses

Definition: comeback loss = declared `choose_go` and then lost the game.

- V4 `choose_go`: 563
- V4 `choose_stop`: 871
- V4 comeback losses: 33
- V3 `choose_go`: 223
- V3 `choose_stop`: 931
- V3 comeback losses: 1

## 5) Model files used (exact)

- V4: heuristic policy `heuristic_v4` (no model file)
- V3: heuristic policy `heuristic_v3` (no model file)

## Conclusion

- On this latest 2000-game run, V4 outperformed V3 on win rate, score, and gold.

---

## Heuristic V4 vs Danmok V15 (2000 games)

- Source reports:
- `logs/v15_vs_v4_1000-report.json`
- `logs/v4_vs_v15_1000-report.json`
- Source logs:
- `logs/v15_vs_v4_1000.jsonl`
- `logs/v4_vs_v15_1000.jsonl`
- Match count: 1000 + 1000 = 2000 (first-turn balanced)

### 1) Wins

- V4 wins: 1534
- V15 wins: 462
- Draws: 4
- V4 win rate: 76.70%
- V15 win rate: 23.10%

### 2) Average score

- V4 average score: 9.3745
- V15 average score: 3.2240

### 3) Gold

- V4 average gold: 1,405,507.30
- V15 average gold: 594,492.70
- V4 - V15 average gold delta: +811,014.60 per game
- V4 - V15 cumulative delta over 2000 games: +1,622,029,200

### 4) GO/STOP and comeback losses

Definition: comeback loss = declared `choose_go` and then lost the game.

- V4 `choose_go`: 972
- V4 `choose_stop`: 1299
- V4 comeback losses: 16
- V15 `choose_go`: 0
- V15 `choose_stop`: 421
- V15 comeback losses: 0

### 5) Model files used (exact from logs)

- V4: heuristic policy `heuristic_v4` (no model file)
- V15 attack policy: `models/policy-danmokv15-attack.json`
- V15 defense policy: `models/policy-danmokv15-defense.json`
- V15 value model: not recorded in these historical matchup logs

---

## Heuristic V4 vs Danmok V14 (2000 games)

- Source reports:
- `logs/v14_vs_v4_1000-report.json`
- `logs/v4_vs_v14_1000-report.json`
- Source logs:
- `logs/v14_vs_v4_1000.jsonl`
- `logs/v4_vs_v14_1000.jsonl`
- Match count: 1000 + 1000 = 2000 (first-turn balanced)

### 1) Wins

- V4 wins: 1624
- V14 wins: 375
- Draws: 1
- V4 win rate: 81.20%
- V14 win rate: 18.75%

### 2) Average score

- V4 average score: 9.7350
- V14 average score: 3.0520

### 3) Gold

- V4 average gold: 1,417,959.45
- V14 average gold: 582,040.55
- V4 - V14 average gold delta: +835,918.90 per game
- V4 - V14 cumulative delta over 2000 games: +1,671,837,800

### 4) GO/STOP and comeback losses

Definition: comeback loss = declared `choose_go` and then lost the game.

- V4 `choose_go`: 970
- V4 `choose_stop`: 1401
- V4 comeback losses: 9
- V14 `choose_go`: 7
- V14 `choose_stop`: 354
- V14 comeback losses: 4

### 5) Model files used (exact from logs)

- V4: heuristic policy `heuristic_v4` (no model file)
- V14 attack policy: `models/policy-danmokv14-attack.json`
- V14 defense policy: `models/policy-danmokv14-defense.json`
- V14 value model: not recorded in these historical matchup logs

---

## Heuristic V4 vs Danmok V13 (2000 games)

- Source reports:
- `logs/v13_vs_v4_1000-report.json`
- `logs/v4_vs_v13_1000-report.json`
- Source logs:
- `logs/v13_vs_v4_1000.jsonl`
- `logs/v4_vs_v13_1000.jsonl`
- Match count: 1000 + 1000 = 2000 (first-turn balanced)

### 1) Wins

- V4 wins: 1594
- V13 wins: 405
- Draws: 1
- V4 win rate: 79.70%
- V13 win rate: 20.25%

### 2) Average score

- V4 average score: 9.6310
- V13 average score: 3.1000

### 3) Gold

- V4 average gold: 1,460,774.95
- V13 average gold: 539,225.05
- V4 - V13 average gold delta: +921,549.90 per game
- V4 - V13 cumulative delta over 2000 games: +1,843,099,800

### 4) GO/STOP and comeback losses

Definition: comeback loss = declared `choose_go` and then lost the game.

- V4 `choose_go`: 907
- V4 `choose_stop`: 1392
- V4 comeback losses: 5
- V13 `choose_go`: 0
- V13 `choose_stop`: 389
- V13 comeback losses: 0

### 5) Model files used (exact from logs)

- V4: heuristic policy `heuristic_v4` (no model file)
- V13 attack policy: `models/policy-danmokv13-attack-2m.json`
- V13 defense policy: `models/policy-danmokv13-defense-2m.json`
- V13 value model: not recorded in these historical matchup logs

---

## Danmok V15 vs Danmok V16 (2000 games)

- Source reports:
- `logs/v15_vs_v16_1000-report.json`
- `logs/v16_vs_v15_1000-report.json`
- Source logs:
- `logs/v15_vs_v16_1000.jsonl`
- `logs/v16_vs_v15_1000.jsonl`
- Match count: 1000 + 1000 = 2000 (first-turn balanced)

### 1) Wins

- V16 wins: 1000
- V15 wins: 995
- Draws: 5
- V16 win rate: 50.00%
- V15 win rate: 49.75%

### 2) Average score

- V16 average score: 5.9770
- V15 average score: 5.7080

### 3) Gold

- V16 average gold: 994,452.95
- V15 average gold: 1,005,547.05
- V16 - V15 average gold delta: -11,094.10 per game
- V16 - V15 cumulative delta over 2000 games: -22,188,200

### 4) Model files used (exact)

- V15 attack policy: `models/policy-danmokv15-attack.json`
- V15 defense policy: `models/policy-danmokv15-defense.json`
- V15 value model: `models/value-danmokv15-gold.json`
- V16 attack policy: `models/policy-danmokv16-attack.json`
- V16 defense policy: `models/policy-danmokv16-defense.json`
- V16 value model: `models/value-danmokv16-gold.json`

---

## Heuristic V4 vs Danmok V16 (2000 games)

- Source reports:
- `logs/v4_vs_v16_1000-report.json`
- `logs/v16_vs_v4_1000-report.json`
- Source logs:
- `logs/v4_vs_v16_1000.jsonl`
- `logs/v16_vs_v4_1000.jsonl`
- Match count: 1000 + 1000 = 2000 (first-turn balanced)

### 1) Wins

- V4 wins: 1539
- V16 wins: 453
- Draws: 8
- V4 win rate: 76.95%
- V16 win rate: 22.65%

### 2) Average score

- V4 average score: 9.6015
- V16 average score: 3.2035

### 3) Gold

- V4 average gold: 1,430,639.85
- V16 average gold: 569,360.15
- V4 - V16 average gold delta: +861,279.70 per game
- V4 - V16 cumulative delta over 2000 games: +1,722,559,400

### 4) Model files used (exact)

- V4: heuristic policy `heuristic_v4` (no model file)
- V16 attack policy: `models/policy-danmokv16-attack.json`
- V16 defense policy: `models/policy-danmokv16-defense.json`
- V16 value model: `models/value-danmokv16-gold.json`

---

## Danmok V17 vs Danmok V16 (2000 games)

- Source reports:
- `logs/v17_vs_v16_1000-report.json`
- `logs/v16_vs_v17_1000-report.json`
- Source logs:
- `logs/v17_vs_v16_1000.jsonl`
- `logs/v16_vs_v17_1000.jsonl`
- Match count: 1000 + 1000 = 2000 (first-turn balanced)

### 1) Wins

- V17 wins: 960
- V16 wins: 1036
- Draws: 4
- V17 win rate: 48.00%
- V16 win rate: 51.80%

### 2) Average score

- V17 average score: 5.5165
- V16 average score: 5.6655

### 3) Gold

- V17 average gold: 980,490.85
- V16 average gold: 1,019,509.15
- V17 - V16 average gold delta: -39,018.30 per game
- V17 - V16 cumulative delta over 2000 games: -78,036,600

### 4) GO/STOP and comeback losses

Definition: comeback loss = declared `choose_go` and then lost the game.

- V17 `choose_go`: 79
- V17 `choose_stop`: 907
- V17 comeback losses: 9
- V16 `choose_go`: 67
- V16 `choose_stop`: 974
- V16 comeback losses: 6

### 5) Model files used (exact)

- V17 attack policy: `models/policy-danmokv17-attack.json`
- V17 defense policy: `models/policy-danmokv17-defense.json`
- V17 value model: `models/value-danmokv17-gold.json`
- V16 attack policy: `models/policy-danmokv16-attack.json`
- V16 defense policy: `models/policy-danmokv16-defense.json`
- V16 value model: `models/value-danmokv16-gold.json`

---

## Heuristic V4 vs Danmok V17 (2000 games)

- Source reports:
- `logs/v4_vs_v17_1000-report.json`
- `logs/v17_vs_v4_1000-report.json`
- Source logs:
- `logs/v4_vs_v17_1000.jsonl`
- `logs/v17_vs_v4_1000.jsonl`
- Match count: 1000 + 1000 = 2000 (first-turn balanced)

### 1) Wins

- V4 wins: 1578
- V17 wins: 418
- Draws: 4
- V4 win rate: 78.90%
- V17 win rate: 20.90%

### 2) Average score

- V4 average score: 9.7460
- V17 average score: 3.0835

### 3) Gold

- V4 average gold: 1,258,274.55
- V17 average gold: 741,725.45
- V4 - V17 average gold delta: +516,549.10 per game
- V4 - V17 cumulative delta over 2000 games: +1,033,098,200

### 4) GO/STOP and comeback losses

Definition: comeback loss = declared `choose_go` and then lost the game.

- V4 `choose_go`: 1021
- V4 `choose_stop`: 1348
- V4 comeback losses: 25
- V17 `choose_go`: 18
- V17 `choose_stop`: 389
- V17 comeback losses: 3

### 5) Model files used (exact)

- V4: heuristic policy `heuristic_v4` (no model file)
- V17 attack policy: `models/policy-danmokv17-attack.json`
- V17 defense policy: `models/policy-danmokv17-defense.json`
- V17 value model: `models/value-danmokv17-gold.json`
