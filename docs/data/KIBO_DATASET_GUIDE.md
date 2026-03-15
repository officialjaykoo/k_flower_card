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
- Feature profile can be forced with `--feature-profile hand10|material10|position11`.
- If omitted, `model_duel_worker.mjs` uses `auto` and tries to infer from loaded model `feature_spec.profile`.

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
- `features`: profile-dependent numeric vector
  - `hand10` / `material10`: 10D
  - `position11`: 11D
  - legacy `800`-series models: auto-inferred `legacy13` 13D

### 3-3. `features` Index Definition
The order is identical to `featureVectorForCandidate()` in `model_duel_worker.mjs`.

Active default profile: `hand10`

Important:
- `hand10` features are computed on a visible post-candidate state.
- For `play` / `match`, the worker first applies the candidate to a masked visible-state simulation, then extracts hand features from that resulting state.
- This keeps the vector candidate-sensitive without leaking hidden deck information.

#### `hand10` (10D)
1. `candidate_combo_gain`
Range: `0..1`
Rule: immediate combo/completion value unlocked by the current candidate

2. `candidate_pi_value`
Range: `0..1`
Rule: weighted pi-like value of the current candidate card

3. `candidate_match_flag`
Range: `0/1`
Rule: `1.0` if the candidate immediately matches board month context, else `0.0`

4. `candidate_safe_discard`
Range: `0/1`
Rule: `1.0` if the candidate month is currently a safe discard month, else `0.0`

5. `candidate_danger_exposure`
Range: `0..1`
Rule: `max(combo_risk, pi_risk)` for exposing this candidate as a live target
Notes:
- combo risk uses opponent combo-target matching on the candidate card
- pi risk uses opponent pi-pressure times candidate pi-value
- if the candidate immediately matches, exposure is treated as `0`

6. `post_hand_matchable_ratio`
Range: `0..1`
Rule: `count(post-hand cards whose month exists on board) / post_hand_count`

7. `post_hand_triple_flag`
Range: `0/1`
Rule: `1.0` if any month appears `>= 3` times in post-hand, else `0.0`

8. `post_hand_high_value_density`
Range: `0..1`
Rule: `count(post-hand ∈ {kwang, ssangpi-like}) / post_hand_count`
Notes:
- current ssangpi-like check uses the same weighted high-value family used elsewhere in the repo
- includes `K1=2`, `L3=2`, `I0(gukjin)=2`, `M0=3`, `M1=4`

9. `combined_post_block_norm`
Range: `0..1`
Rule: `max(post_pi_block, post_combo_block)`
Meaning:
- strongest remaining hold/block value after the candidate is applied
- combines opponent pi blocking and opponent combo blocking on the post-candidate visible state

10. `global_context_trigger`
Range: `0..1`
Rule: `clamp01(0.5 + 0.5 * positional_advantage_signed)`
Meaning:
- signed whole-position advantage compressed into a stable `0..1` signal
- keeps candidate evaluation aware of whether the actor is currently ahead or behind overall

Important:
- `hand10` is no longer a static hand-ratio profile.
- It is now an `Action 5 + Post-State 5` profile:
  - `1~5`: immediate candidate gain / safety / exposure
  - `6~10`: remaining hand quality and whole-position context after applying the candidate

#### `position11` (11D, alternate profile)
1. `self_win_path_score`
2. `self_material_concentration`
3. `self_ssangpi_hoard_norm`
4. `self_combo_completion_norm`
5. `self_go_safety_norm`
6. `self_tempo_advantage`
7. `opp_win_proximity_norm`
8. `opp_ssangpi_threat_norm`
9. `hand_critical_block_norm`
10. `opp_go_threat_norm`
11. `board_danger_ratio`

Notes:
- `position11` is a whole-position profile built from visible post-candidate state.
- It emphasizes self win path, opponent win threat, and GO/STOP context rather than local candidate score only.

#### `material10` (10D, alternate profile)
1. `current_multiplier_norm`
2. `candidate_combo_gain`
3. `opp_combo_threat_norm`
4. `candidate_block_gain_norm`
5. `candidate_public_known_ratio`
6. `immediate_match_possible`
7. `self_ssangpi_control_norm`
8. `ssangpi_revealed_ratio_norm`
9. `opp_stop_pressure_norm`
10. `score_diff_tanh`

Notes:
- `material10` remains available as an alternate NEAT profile.
- `position11` remains available as an 11-input alternate NEAT profile.
- `self_ssangpi_control_norm` / `ssangpi_revealed_ratio_norm` use weighted ssangpi value sum normalized by total `13`, not card-count `/5`.
- Legacy 13-input `800`-series models are auto-inferred as `legacy13`.
- Legacy 6-output `1200`-series models are no longer supported.

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
