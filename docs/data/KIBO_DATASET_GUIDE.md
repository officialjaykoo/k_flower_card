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
- Feature profile can be forced with `--feature-profile race2|race5|race6|race7|race8|race20|hand7|hand10|material10|race10|oracle10|oracle10v2`.
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
  - `race2`: 2D
  - `race5`: 5D
  - `race6`: 6D
  - `race7`: 7D
  - `race8`: 8D
  - `race20`: 20D
  - `hand7`: 7D
  - `hand10` / `material10` / `race10` / `oracle10` / `oracle10v2`: 10D
  - legacy `800`-series models: auto-inferred `legacy13` 13D

### 3-3. `features` Index Definition
The order is identical to `featureVectorForCandidate()` in `model_duel_worker.mjs`.

Active default profile: `race8`

#### `race2` (2D)
1. `candidate_public_known_ratio`
Range: `0..1`
Rule: known/public month saturation ratio for the candidate month
Meaning:
- minimal candidate-specific public information signal

2. `opp_stop_pressure_norm`
Range: `0..1`
Rule: staged opponent stop-pressure signal reused from `material10`
Meaning:
- minimal shared option/global pressure signal

Important:
- `race2` is an extreme diagnostic profile.
- intent is to test whether a tiny `play candidate info + option pressure` pair can outperform larger noisy profiles under the current NEAT budget.

#### `race5` (5D)
1. `candidate_public_known_ratio`
Range: `0..1`
Rule: known/public month saturation ratio for the candidate month
Meaning:
- strongest material-style candidate information signal

2. `immediate_match_possible`
Range: `0..1`
Rule: `1.0` when the candidate month already exists on board, else `0.0`
Meaning:
- strongest material-style direct match signal

3. `ssangpi_revealed_ratio_norm`
Range: `0..1`
Rule: weighted revealed ssangpi-family value ratio
Meaning:
- material-style public ssangpi exposure signal used by the strongest `material10` winner

4. `opp_stop_pressure_norm`
Range: `0..1`
Rule: staged opponent stop-pressure signal reused from `material10`
Meaning:
- strongest option/global pressure signal

5. `candidate_block_gain_norm`
Range: `-1..1`
Rule: immediate block-value unlocked by the candidate against opponent combo progress
Meaning:
- strongest material-style defensive/option bridge signal

Important:
- `race5` is a compressed `play 3 + option 2` profile.
- it is built directly from the strongest surviving `material10` winner signals rather than from broader race/context theory.

#### `race6` (6D)
1. `candidate_public_known_ratio`
Range: `0..1`
Rule: known/public month saturation ratio for the candidate month
Meaning:
- strongest material-style candidate information signal

2. `immediate_match_possible`
Range: `0..1`
Rule: `1.0` when the candidate month already exists on board, else `0.0`
Meaning:
- strongest material-style direct match signal

3. `ssangpi_revealed_ratio_norm`
Range: `0..1`
Rule: weighted revealed ssangpi-family value ratio
Meaning:
- material-style public ssangpi exposure signal used by the strongest `material10` winner

4. `opp_stop_pressure_norm`
Range: `0..1`
Rule: staged opponent stop-pressure signal reused from `material10`
Meaning:
- strongest option/global pressure signal

5. `candidate_block_gain_norm`
Range: `-1..1`
Rule: immediate block-value unlocked by the candidate against opponent combo progress
Meaning:
- strongest material-style defensive/option bridge signal

6. `candidate_combo_gain`
Range: `0..1`
Rule: immediate combo/completion value unlocked by the current candidate
Meaning:
- next strongest surviving option-side material signal, added on top of `race5`

Important:
- `race6` is `race5 + candidate_combo_gain`.
- it keeps the strongest observed `material10` play 3 + option 2 core and adds the next strongest material signal rather than reintroducing broad context noise.

#### `race7` (7D)
1. `candidate_combo_gain`
2. `candidate_block_gain_norm`
3. `candidate_public_known_ratio`
4. `immediate_match_possible`
5. `self_ssangpi_control_norm`
6. `ssangpi_revealed_ratio_norm`
7. `opp_stop_pressure_norm`

Meaning:
- `race7` is `material10` minus the three weakest winner-side signals:
  - `current_multiplier_norm`
  - `opp_combo_threat_norm`
  - `score_diff_tanh`
- it keeps the stronger play/material/option core while staying below the full 10D search cost.

#### `race8` (8D)
1. `opp_stop_pressure_norm`
2. `candidate_public_known_ratio`
3. `ssangpi_revealed_ratio_norm`
4. `candidate_block_gain_norm`
5. `candidate_combo_gain`
6. `immediate_match_possible`
7. `self_ssangpi_control_norm`
8. `current_multiplier_norm`

Meaning:
- `race8` is `material10` minus the two weakest winner-side signals:
  - `opp_combo_threat_norm`
  - `score_diff_tanh`
- the retained 8 are reordered by observed practical importance, with `current_multiplier_norm` left as the weakest retained slot.

#### `hand7` (7D)
1. `match_certainty_norm`
Range: `0..1`
Rule:
- `1.0` if `decisionType === "match"`
- `1.0 / 0.8 / 0.5 / 0.0` for board month count `>=3 / 2 / 1 / 0`
Meaning:
- strongest candidate-level match pressure signal, upgraded from a binary match flag

2. `candidate_safe_discard`
Range: `0..1`
Rule: graduated discard-safety score for the candidate month
Meaning:
- keeps the profile candidate-sensitive while favoring safe plays over brittle hand-shape signals

3. `candidate_combo_gain`
Range: `0..1`
Rule: immediate combo/completion value unlocked by the current candidate

4. `combined_post_block_norm`
Range: `0..1`
Rule: `max(post_pi_block, post_combo_block)` on the visible post-candidate state
Meaning:
- strongest remaining hold/block value after applying the candidate

5. `candidate_public_known_ratio`
Range: `0..1`
Rule: known/public month saturation ratio for the candidate month
Meaning:
- candidate-specific public information advantage signal reused from `material10`

6. `global_context_trigger`
Range: `0..1`
Rule: `clamp01(0.5 + 0.5 * positional_advantage_signed)`
Meaning:
- whole-position context kept as a single shared/global slot

7. `opp_stop_pressure_norm`
Range: `0..1`
Rule: staged opponent stop-pressure signal reused from `material10`
Meaning:
- shared option/global pressure slot kept even before full play/option feature splitting

Important:
- `hand7` is a reduced `5 play + 2 option/global` profile.
- intent is to cut search-space noise before fully splitting `play/match` and `go/stop` inputs.

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
Range: `0..1`
Rule: graduated discard-safety score for the candidate month
Notes:
- `1.0` if the candidate immediately matches the board
- otherwise uses `max(board_month_support, known_month_saturation)`
- this keeps the slot candidate-sensitive without collapsing to a binary safe/unsafe flag

5. `candidate_combo_feed_risk`
Range: `0..1`
Rule: combo-only exposure risk for leaving this candidate as a live opponent combo target
Notes:
- combo risk uses opponent combo-target matching on the candidate card
- if the candidate immediately matches, exposure is treated as `0`

6. `candidate_pi_feed_risk`
Range: `0..1`
Rule: opponent pi-pressure multiplied by the candidate's weighted pi-like value
Meaning:
- isolates the pi/ssangpi feed risk instead of mixing it into combo exposure

7. `post_hand_combo_reserve_norm`
Range: `0..1`
Rule: `count(post-hand ∈ {kwang, five, ribbon}) / post_hand_count`
Meaning:
- how much combo material is still preserved after applying the candidate
- more general and less brittle than a binary triple/stack flag

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
- It is now effectively an `Action 6 + Post-State 4` profile:
  - `1~6`: immediate candidate gain / safety / separated feed risks
  - `7~10`: remaining hand quality and whole-position context after applying the candidate

#### `race10` (10D, split-head mixed profile)
1. `match_certainty_norm`
2. `candidate_safe_discard`
3. `candidate_combo_gain`
4. `combined_post_block_norm`
5. `candidate_public_known_ratio`
6. `opp_stop_pressure_norm`
7. `current_multiplier_norm`
8. `self_ssangpi_control_norm`
9. `ssangpi_revealed_ratio_norm`
10. `global_context_trigger`

Meaning:
- `1~5` are action/play-side signals
- `6~10` are option/go-stop-side signals
- intended for hard split-head use with:
  - `output0(action_score) -> [0,1,2,3,4]`
  - `output1(option_bias) -> [5,6,7,8,9]`

Notes:
- `race10` is no longer a `material10` alias.
- it is now a mixed profile built from:
  - `hand7/hand10` action-side candidate signals
  - `material10` option/global pressure signals
- the goal is to separate card-choice information from go/stop information at the feature level as well as the head level.

#### `oracle10` (10D, fixed split-head residual profile)
Play head (`output0`, slots `0..5`)
1. `candidate_public_known_ratio`
2. `match_certainty_norm`
3. `candidate_combo_gain`
4. `candidate_pi_value`
5. `candidate_pi_feed_risk`
6. `combined_post_block_norm`

Option head (`output1`, slots `6..9`)
7. `post_hand_high_value_density`
8. `combined_post_block_norm`
9. `global_context_trigger`
10. `post_hand_combo_reserve_norm`

Meaning:
- `oracle10` keeps a fixed `play 6 / option 4` residual input profile.
- these 10 slots are not the oracle itself; they are the NEAT residual context inputs.
- current decision path does not add oracle base scores directly.
- the fixed 6:4 slots are currently just the NEAT residual context inputs.

#### `oracle10v2` (10D residual profile, oracle-aware revision)
Play head (`output0`, slots `0..5`)
1. `candidate_public_known_ratio`
2. `match_certainty_norm`
3. `candidate_combo_gain`
4. `candidate_pi_feed_risk`
5. `combined_post_block_norm`
6. `candidate_safe_discard`

Option head (`output1`, slots `6..9`)
7. `post_hand_high_value_density`
8. `post_hand_combo_reserve_norm`
9. `global_context_trigger`
10. `opp_stop_pressure_norm`

Meaning:
- `oracle10v2` keeps the same `play 6 / option 4`-derived 10D residual input budget.
- compared with `oracle10`, it removes residual slots that overlap more with oracle-covered rule scoring and replaces them with uncertainty/risk context.
- current decision path does not add oracle base scores directly.

#### `race20` (20D, split-head candidate pool)
Action head pool (`output0`, slots `0..9`)
1. `match_certainty_norm`
2. `immediate_match_possible`
3. `candidate_safe_discard`
4. `candidate_combo_gain`
5. `candidate_pi_value`
6. `candidate_public_known_ratio`
7. `combined_post_block_norm`
8. `candidate_block_gain_norm`
9. `candidate_combo_feed_risk`
10. `candidate_pi_feed_risk`

Option head pool (`output1`, slots `10..19`)
11. `opp_stop_pressure_norm`
12. `current_multiplier_norm`
13. `score_diff_tanh`
14. `self_ssangpi_control_norm`
15. `ssangpi_revealed_ratio_norm`
16. `opp_combo_threat_norm`
17. `global_context_trigger`
18. `post_hand_combo_reserve_norm`
19. `post_hand_high_value_density`
20. `combined_post_block_norm`

Meaning:
- `race20` is no longer a flat union list.
- it is a curated `play 10 / option 10` split candidate pool built from the strongest ideas across `material10`, `hand10`, and the recent `race` experiments.
- intended hard split-head mask:
  - `output0(action_score) -> [0..9]`
  - `output1(option_bias) -> [10..19]`
- the profile is now used as a fixed split candidate pool without trainer-side input gates.

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
- `race2` is a minimal 2D diagnostic profile built from `candidate_public_known_ratio` and `opp_stop_pressure_norm`.
- `race5` is a compressed 5D profile built from the strongest observed `material10` play/option signals.
- `race6` is a compressed 6D profile that adds `candidate_combo_gain` to the `race5` core.
- `race7` is a 7D `material10` compression that removes `current_multiplier_norm`, `opp_combo_threat_norm`, and `score_diff_tanh`.
- `race8` is an 8D `material10` compression that removes only `opp_combo_threat_norm` and `score_diff_tanh`.
- `race10` is now a split-head mixed profile rather than a `material10` alias.
- `oracle10` is a fixed 6:4 split residual profile built from the strongest surviving `1602` playoff features.
- `oracle10v2` is the oracle-aware revision that swaps `candidate_pi_value -> candidate_safe_discard` and `combined_post_block_norm(option) -> opp_stop_pressure_norm`.
- its runtime is oracle-augmented: rule-based base scoring plus NEAT residual.
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
  "features": [0.8,0.6,1,0.7,0.5,0.7333,0.2]
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
