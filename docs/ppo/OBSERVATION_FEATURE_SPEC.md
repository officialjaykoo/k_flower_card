# PPO Observation Feature Spec (Current Code)

## TL;DR
- Current PPO observation size is **246**, not 450.
- Formula:
  - `OBS_DIM = 6 (phase) + 3 (decision) + 103 (macro/public strategy) + 70 (hand slots) + 56 (match slots) + 8 (option mask)`
  - `OBS_DIM = 246`
- Action mask size is `ACTION_DIM = 26` (`10 play + 8 match + 8 option`).

## Source of Truth
- `ppo/scripts/ppo_env_bridge.mjs`
- Verified against code on 2026-03-06.

## Global Layout (Index Map)

| Index Range | Size | Block |
|---|---:|---|
| `0..5` | 6 | phase one-hot |
| `6..8` | 3 | decision type one-hot |
| `9..111` | 103 | macro/public strategy |
| `112..181` | 70 | hand card slots (`10 x 7`) |
| `182..237` | 56 | match card slots (`8 x 7`) |
| `238..245` | 8 | option availability flags |

---

## 1) Phase One-Hot (`0..5`)

| Index | Feature |
|---:|---|
| 0 | `phase == playing` |
| 1 | `phase == select-match` |
| 2 | `phase == go-stop` |
| 3 | `phase == president-choice` |
| 4 | `phase == gukjin-choice` |
| 5 | `phase == shaking-confirm` |

## 2) Decision Type One-Hot (`6..8`)

| Index | Feature |
|---:|---|
| 6 | `decisionType == play` |
| 7 | `decisionType == match` |
| 8 | `decisionType == option` |

## 3) Macro/Public Strategy (`9..111`, total 103)

### 3-a) Core Public Counts/Scores (`9..30`, 22)

| Index | Feature |
|---:|---|
| 9 | `deck_len / 30` |
| 10 | `self_hand_len / 10` |
| 11 | `opp_hand_len / 10` |
| 12 | `board_len / 24` |
| 13 | `self_go_count / 5` |
| 14 | `opp_go_count / 5` |
| 15 | `tanh(score_self_total / 10)` |
| 16 | `tanh(score_opp_total / 10)` |
| 17 | `tanh((score_self_total - score_opp_total) / 10)` |
| 18 | `tanh(self_gold / 50000)` |
| 19 | `tanh(opp_gold / 50000)` |
| 20 | `tanh((self_gold - opp_gold) / 25000)` |
| 21 | `(carryOverMultiplier - 1) / 12` |
| 22 | `turnSeq / 200` |
| 23 | `self_captured_kwang / 5` |
| 24 | `self_captured_five / 5` |
| 25 | `self_captured_ribbon / 10` |
| 26 | `self_scoring_pi / 20` |
| 27 | `opp_captured_kwang / 5` |
| 28 | `opp_captured_five / 5` |
| 29 | `opp_captured_ribbon / 10` |
| 30 | `opp_scoring_pi / 20` |

### 3-b) High-Signal Public Features (`31..49`, 19)

| Index | Feature |
|---:|---|
| 31 | `self_can_stop` (self score >= go min score) |
| 32 | `opp_can_stop` |
| 33 | `legal_actions_count / 10` |
| 34 | `immediate_match_possible` |
| 35 | `match_opportunity_density` |
| 36 | `has_bomb` |
| 37 | `has_shaking` |
| 38 | `current_multiplier_norm` |
| 39 | `self_godori_progress` |
| 40 | `self_cheongdan_progress` |
| 41 | `self_hongdan_progress` |
| 42 | `self_chodan_progress` |
| 43 | `opp_godori_progress` |
| 44 | `opp_cheongdan_progress` |
| 45 | `opp_hongdan_progress` |
| 46 | `opp_chodan_progress` |
| 47 | `self_score_gap_to_stop_norm` |
| 48 | `opp_score_gap_to_stop_norm` |
| 49 | `candidate_month_known_ratio` |

### 3-c) Month Distribution Profiles (`50..97`, 48)

Each month bucket is normalized by denominator 4.

| Index Range | Meaning |
|---|---|
| `50..61` | board month distribution (`month 1..12`) |
| `62..73` | self captured month distribution |
| `74..85` | opponent captured month distribution |
| `86..97` | self hand month distribution |

### 3-d) Opponent Shaking-Revealed Month Flags (`98..109`, 12)

| Index Range | Meaning |
|---|---|
| `98..109` | opponent declared/revealed shaking month flags (`month 1..12`) |

### 3-e) Strategic Auxiliary Public Features (`110..111`, 2)

| Index | Feature |
|---:|---|
| 110 | `multiplier_risk_norm` |
| 111 | `go_expected_value_norm_public` |

## 4) Hand Card Slots (`112..181`, `10 x 7`)

- Slot count: 10 (`PLAY_SLOTS`)
- Per-slot feature vector order (7):
  1. `is_present` (0/1)
  2. `month_norm = month / 12`
  3. `is_kwang`
  4. `is_five`
  5. `is_ribbon`
  6. `is_junk`
  7. `pi_value_norm = piValue / 5`
- Empty slot is all zeros.

## 5) Match Candidate Slots (`182..237`, `8 x 7`)

- Slot count: 8 (`MATCH_SLOTS`)
- Same 7-dim card feature schema as hand slots.
- Filled only from current `match` candidates; others are zero-padded.

## 6) Option Availability Flags (`238..245`, 8)

`OPTION_ORDER`:
1. `go`
2. `stop`
3. `shaking_yes`
4. `shaking_no`
5. `president_stop`
6. `president_hold`
7. `five`
8. `junk`

Each index stores legality flag from action mask (`1 legal / 0 illegal`).

## Related: Action Space (Not in Observation)

- `ACTION_DIM = 26`
- Partition:
  - `0..9`: play slots
  - `10..17`: match slots
  - `18..25`: option actions

## Fail-Fast Guards

Bridge throws explicit errors when:
- observation has non-finite values
- final observation length is not `246`
- action mask length is not `26`

