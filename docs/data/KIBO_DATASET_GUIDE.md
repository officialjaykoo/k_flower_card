# Kibo & Dataset Guide (`model_duel_worker.mjs`)

## Conclusion
- `kibo` and `dataset` serve different purposes.
- `kibo` is a **game-flow/replay log**, while `dataset` is a **decision-level training sample set**.
- You can generate a `dataset` even if you do not save a `kibo` file.
- However, `dataset` cannot fully replace full replay context.

## 1. Generation Rules
Reference script: `scripts/model_duel_worker.mjs`

### 1-1. Default Output
- One `result` report JSON is always saved.
- Default folder pattern:
`logs/model_duel/<human>_vs_<ai>_<YYYYMMDD>/`
- Default file pattern:
`<seed>_result.json`

### 1-2. When Kibo Is Generated
- Saved when `--kibo-detail lean|full` is used.
- If `--kibo-out` is not provided, auto path is:
`<report-folder>/<seed>_kibo.jsonl`
- If `--kibo-detail none` is used with `--kibo-out`, internal behavior is `lean`.

### 1-3. When Dataset Is Generated
- Saved when `--dataset-out <path>` or `--dataset-out auto` is used.
- With `--dataset-out auto`, auto path is:
`<report-folder>/<seed>_dataset.jsonl`

### 1-4. Console vs Report
- Console (stdout): one-line compact summary JSON
- Result file: full report JSON with detailed metrics

## 2. Kibo Structure
File format: `JSONL` (one line = one game)

### 2-1. Top-Level Record Fields
- `game_index`: game index (0-based)
- `seed`: per-game seed string
- `first_turn`: `human|ai`
- `human`: label of the human-slot player
- `ai`: label of the ai-slot player
- `winner`: `human|ai|""`
- `result`: final score/multiplier/settlement info
- `kibo_detail`: `lean|full`
- `kibo`: event array

### 2-2. Main `kibo` Event Types
- `initial_deal`
- `shaking_declare`
- `bonus_use`
- `turn_end`
- `go`
- `stop`
- `president_hold`
- `president_stop`
- `gukjin_mode`
- `round_end`

Depending on engine state, some event types may not appear in a given game.

### 2-3. `lean` vs `full`
The key difference is snapshot depth.

- `initial_deal`:
  - `lean`: `handsCount`, `boardCount`, `deckCount`
  - `full`: full card arrays in `hands`, `board`, `deck`
- `turn_end`:
  - `lean`: `handsCount`, `boardCount`, `deckCount`
  - `full`: full card arrays in `hands`, `board` (plus `deckCount`)
- Core fields like `action`, `events`, `steals`, and `ppukState` are recorded in both modes.

### 2-4. Kibo Example (Summary)
```json
{
  "game_index": 0,
  "seed": "cl_vs_gemini_1000|g=0|first=ai|sr=0",
  "first_turn": "ai",
  "human": "heuristic_h_cl",
  "ai": "heuristic_h_gemini",
  "winner": "human",
  "kibo_detail": "lean",
  "kibo": [
    { "no": 1, "type": "initial_deal", "handsCount": { "human": 10, "ai": 10 } },
    { "no": 3, "type": "turn_end", "turnNo": 1, "actor": "ai", "action": { "type": "play" } },
    { "no": 19, "type": "round_end", "winner": "human" }
  ]
}
```

## 3. Dataset Structure
File format: `JSONL` (one line = one candidate)

### 3-1. Row Granularity
- If one decision has `N` legal candidates, the dataset writes `N` rows.
- In the same decision group, `chosen=1` is usually present on exactly one row.
- If inference fails (`unresolved=1`), a group may have no `chosen=1` row.

### 3-2. Record Fields
- `game_index`, `seed`, `first_turn`, `step`
- `actor`: `human|ai`
- `actor_policy`: policy label
- `action_source`: `heuristic|model|fallback_random`
- `decision_type`: `play|match|option`
- `legal_count`: number of legal candidates in that decision
- `candidate`: candidate identifier
- `chosen`: `0|1`
- `chosen_candidate`: inferred selected candidate
- `unresolved`: `0|1`
- `features`: 16-dimensional numeric vector (`compact16`)
- `features13`: optional legacy compact13 prefix slice for go-stop dataset rows

### 3-3. `features` (16D) Index Definition
The order is identical to `featureVectorForCandidate()` in `model_duel_worker.mjs`.

#### Decision Context

1. `is_play_decision`
Range: `0/1`
Rule: `decisionType === "play" ? 1 : 0`
Examples: `play=[1,0]`, `match=[0,1]`, `option=[0,0]`

2. `is_match_decision`
Range: `0/1`
Rule: `decisionType === "match" ? 1 : 0`
Note: `1,2` are context flags because normal play/match cards usually have `3=0.0`

3. `action_code_norm`
Range: `0..1`
Role: candidate action kind code
Values: `go=0.125`, `stop=0.25`, `shaking_yes=0.375`, `shaking_no=0.5`, `president_stop=0.625`, `president_hold=0.75`, `five=0.875`, `junk=1.0`, `shake_start=0.9`, `bomb=0.95`
Note: normal play/match card candidates usually use `0.0`

#### Score Context

4. `score_diff_tanh`
Range: `-1..1`
Rule: `tanh((selfScore.total - oppScore.total) / 10)`
Meaning: positive when leading, negative when trailing, `0` when tied
Examples: `self 6, opp 3 -> 0.291`, `self 3, opp 6 -> -0.291`, `self 7, opp 6 -> 0.100`

5. `self_score_progress_norm`
Range: `0..1`
Stage 1 (`0..7`): `0.72 * (score / 7)^1.35`
Stage 2 (`7+`): `0.72 + 0.28 * log2(1 + min(score - 7, 8)) / log2(9)`
Examples: `1->0.05`, `5->0.46`, `7->0.72`, `10->0.90`, `15+->1.0`

6. `opp_stop_pressure_norm`
Range: `0..1`
Rule: staged by opponent score
Values: `0->0.0`, `1->0.05`, `2->0.1`, `3->0.15`, `4->0.3`, `5->0.6`, `6->0.9`, `7+->1.0`
Note: soft pressure only; `13` separately marks hard stop availability

7. `current_multiplier_norm`
Range: `0..1`
Rule: `clamp01(log2(scoreSelf.multiplier * state.carryOverMultiplier) / 4)`
Includes: go multiplier (`3go x2`, `4go x4`, `5go x8`, ...), shaking, bomb, bak, carry-over
Excludes: additive go score (`+1` per go), president-hold post-resolution multiplier
Examples: `x1->0`, `x2->0.25`, `x4->0.5`, `x8->0.75`, `x16+->1.0`

#### Candidate Value

8. `candidate_combo_gain`
Range: `0..1`
Rule: `clamp01((kwang_gain_delta + godori_gain + dan_gain) / 11)`
Details: visible-state simulation only
Examples: `2kwang->bi3kwang=+2`, `2kwang->3kwang=+3`, `3kwang->4kwang=+1/+2`, `4kwang->5kwang=+11`, `godori=+5`, `hong/cheong/cho-dan=+3`

9. `candidate_pi_value_norm`
Range: `0..1`
Rule: `effective_pi = card.piValue + card.bonus.stealPi`; `clamp01(effective_pi / 4)`
Examples: normal pi `1->0.25`, double pi `2->0.5`, bonus double+steal `3->0.75`, bonus triple+steal `4->1.0`

10. `immediate_match_possible`
Range: `0/1`
Rule: `decisionType === "match" ? 1 : (month>0 && board has same-month card ? 1 : 0)`
Meaning: just match availability, not match quality or count

11. `candidate_public_known_ratio`
Range: `0..1`
Rule: `known_month_cards / total_month_cards`
Note: total cards are `4` for months `1..12`, `2` for bonus month `13`
Examples: `13-month 1/2 -> 0.5`, `13-month 2/2 -> 1.0`

#### Thresholds And Go/Stop

12. `self_can_stop`
Range: `0/1`
Rule: `selfScore.total >= 7 ? 1 : 0`
Meaning: hard threshold flag for my stop availability

13. `opp_can_stop`
Range: `0/1`
Rule: `oppScore.total >= 7 ? 1 : 0`
Meaning: hard threshold flag for opponent stop availability

14. `is_go_candidate_gated`
Range: `0/0.5/1`
Rule: if not `go-stop`: `0.5`; in `go-stop`: `go->1.0`, `stop->0.0`, else `0.5`
Meaning: only active in go-stop option context

15. `self_go_count_norm_gated`
Range: `0..1`
Rule: in `go-stop` only, staged from my `goCount`; otherwise `0.0`
Values: `0go->0.0`, `1go->0.2`, `2go->0.35`, `3go->0.7`, `4go->0.9`, `5go+->1.0`

16. `opp_go_count_norm_gated`
Range: `0..1`
Rule: in `go-stop` only, staged from opponent `goCount`; otherwise `0.0`
Values: same staged mapping as `15`

Notes:
- `1,2` are not redundant with `3`. `3` mainly distinguishes option/special action kind, while `1,2` tell the model whether the current candidate is a play-card choice or a match-choice context.
- `4,5,6,7,12,13` split score information into different roles: gap (`4`), my progress (`5`), opponent stop pressure (`6`), multiplier pressure (`7`), and hard threshold flags (`12,13`).
- `10` is intentionally binary. Match quality/count is not encoded here; it is partially reflected elsewhere through public-known ratio (`11`) and the outcome-sensitive combo/pi features (`8,9`).

### 3-4. Dataset Example
```json
{
  "game_index": 0,
  "seed": "cl_vs_gemini_1000|g=0|first=ai|sr=0",
  "step": 0,
  "actor": "ai",
  "actor_policy": "heuristic_h_gemini",
  "decision_type": "play",
  "legal_count": 10,
  "candidate": "H3",
  "chosen": 0,
  "chosen_candidate": "A1",
  "unresolved": 0,
  "features": [1,0,0,0,0,0,1,0,0,0.7333]
}
```

## 4. Meaning of `unresolved`
- Meaning: a decision where `chosen_candidate` could not be matched against legal candidates
- Computation: internal stats during dataset generation
- Report fields:
  - `dataset_unresolved_decisions`
  - `unresolved_decision_rate`
  - `unresolved_by_actor/policy/decision_type/action_source`

Even if `kibo` file output is disabled, the engine still attempts chosen-candidate inference from runtime state (`state.kibo`).

## 5. Can You Generate Dataset Without Kibo?
Yes.

- You can enable `dataset_out` without `kibo_out`.
- Reason: chosen-candidate inference runs on in-memory runtime state (`stateBefore/stateAfter`), not on saved kibo files.
- Tradeoff: manual replay-level auditing is weaker without saved kibo.

## 6. What to Enable by Use Case
- Fast benchmark: `result` only
- Training data collection: `result + dataset`
- Debugging/replay analysis: `result + kibo`
- Reproducible deep analysis: `result + kibo + dataset`

## 7. Recommended Command
```powershell
node scripts/model_duel_worker.mjs --human heuristic_h_cl --ai phase3_seed5 --games 1000 --seed cl_vs_phase3s5 --kibo-detail lean --dataset-out auto
```

Generated files example:
- `logs/model_duel/heuristic_h_cl_vs_phase3_seed5_20260226/cl_vs_phase3s5_result.json`
- `logs/model_duel/heuristic_h_cl_vs_phase3_seed5_20260226/cl_vs_phase3s5_kibo.jsonl`
- `logs/model_duel/heuristic_h_cl_vs_phase3_seed5_20260226/cl_vs_phase3s5_dataset.jsonl`
