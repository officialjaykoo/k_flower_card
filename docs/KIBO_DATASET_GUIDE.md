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
  "seed": "v5_vs_v7_1000|g=0|first=ai|sr=0",
  "first_turn": "ai",
  "human": "heuristic_v5",
  "ai": "heuristic_v7_gold_digger",
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
- `features`: 47-dimensional numeric vector

### 3-3. `features` (47D) Index Definition
The order is identical to `featureVectorForCandidate()` in `model_duel_worker.mjs`.

1. phase_playing
2. phase_select_match
3. phase_go_stop
4. phase_president_choice
5. phase_gukjin_choice
6. phase_shaking_confirm
7. decision_play
8. decision_match
9. decision_option
10. deck_len_norm
11. self_hand_len_norm
12. opp_hand_len_norm
13. self_go_count_norm
14. opp_go_count_norm
15. score_diff_tanh
16. self_score_tanh
17. legal_count_norm
18. candidate_pi_value_norm
19. candidate_is_kwang
20. candidate_is_ribbon
21. candidate_is_five
22. candidate_is_junk
23. candidate_is_double_pi
24. match_opportunity_density
25. immediate_match_possible
26. option_code_norm
27. self_gwang_norm
28. opp_gwang_norm
29. self_pi_norm
30. opp_pi_norm
31. self_godori_norm
32. opp_godori_norm
33. self_cheongdan_norm
34. opp_cheongdan_norm
35. self_hongdan_norm
36. opp_hongdan_norm
37. self_chodan_norm
38. opp_chodan_norm
39. self_can_stop
40. opp_can_stop
41. has_shaking_available
42. current_multiplier_norm
43. has_bomb_available
44. self_bak_pi
45. self_bak_gwang
46. self_bak_mongbak
47. candidate_month_known_ratio

### 3-4. Dataset Example
```json
{
  "game_index": 0,
  "seed": "v5_vs_v7_1000|g=0|first=ai|sr=0",
  "step": 0,
  "actor": "ai",
  "actor_policy": "heuristic_v7_gold_digger",
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
node scripts/model_duel_worker.mjs --human heuristic_v5 --ai phase3_seed5 --games 1000 --seed v5_vs_phase3s5 --kibo-detail lean --dataset-out auto
```

Generated files example:
- `logs/model_duel/heuristic_v5_vs_phase3_seed5_20260226/v5_vs_phase3s5_result.json`
- `logs/model_duel/heuristic_v5_vs_phase3_seed5_20260226/v5_vs_phase3s5_kibo.jsonl`
- `logs/model_duel/heuristic_v5_vs_phase3_seed5_20260226/v5_vs_phase3s5_dataset.jsonl`
