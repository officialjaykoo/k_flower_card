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

1. is_play_decision
2. is_match_decision
3. action_code_norm
4. score_diff_tanh
5. self_score_tanh
6. opp_stop_pressure_norm
7. current_multiplier_norm
8. candidate_combo_gain
9. candidate_pi_value_norm
10. immediate_match_possible
11. candidate_public_known_ratio
12. self_can_stop
13. opp_can_stop
14. is_go_candidate_gated
15. self_go_count_norm_gated
16. opp_go_count_norm_gated

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
