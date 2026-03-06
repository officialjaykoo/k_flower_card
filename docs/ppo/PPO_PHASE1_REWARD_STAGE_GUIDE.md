# PPO Phase1 Reward + Stage(1/2/3) Complete Guide

## Scope
- This document covers **Phase1 PPO single-actor training** reward and stage mechanics.
- Source of truth:
  - `ppo/scripts/ppo_env_bridge.mjs`
  - `ppo/scripts/train_ppo.py`
  - `ppo/configs/runtime_phase1_ppo.json`
- Verified on 2026-03-06.

## TL;DR
- Reward is **not terminal-only**.
- Actual training reward is:
  - bridge reward (gold delta + terminal terms + phase1 shaping + go bonus)
  - minus catastrophic penalty applied in trainer.
- Stage 1/2/3 changes opponent mix and schedules (LR, go explore), and resets stage metric windows.
- Reward formula itself is the same across stage 1/2/3.

---

## 1) Reward Pipeline (Exact)

### 1.1 Bridge reward (`ppo_env_bridge.mjs`)

At each policy action:

```text
R_bridge =
  R_gold_delta
  + R_go_action_bonus
  + R_terminal_bonus
  - R_downside_penalty
  + R_outcome_win_bonus
  - R_outcome_loss_penalty
  + R_phase1_shaping_total
```

Where:
- `R_gold_delta = (afterDiff - beforeDiff) * reward_scale`
- `R_go_action_bonus = go_action_bonus` only when policy selected `go` in `go-stop`
- If `done`:
  - `R_terminal_bonus = afterDiff * terminal_bonus_scale`
  - `R_downside_penalty = max(0, -afterDiff) * downside_penalty_scale`
  - `R_outcome_win_bonus = terminal_win_bonus` if `afterDiff > 0`
  - `R_outcome_loss_penalty = terminal_loss_penalty` if `afterDiff < 0`

### 1.2 Trainer-side catastrophic penalty (`train_ppo.py`)

On terminal step:

```text
if final_gold_diff <= catastrophic_loss_threshold:
    cat_gap = catastrophic_loss_threshold - final_gold_diff
    cat_penalty = cat_gap * catastrophic_penalty_scale
    reward -= cat_penalty
```

Final PPO reward:

```text
R_train = R_bridge - catastrophic_penalty
```

---

## 2) Phase1 Shaping (Full List, Code Constants)

Shaping is active only when:
- `phase == 1`
- `training_mode == single_actor`

Constants (`PHASE1_SHAPING`):

| Key | Weight | Trigger Summary |
|---|---:|---|
| `godoriComplete` | `+0.18` | godori completed |
| `cheongdanComplete` | `+0.16` | cheongdan completed |
| `hongdanComplete` | `+0.16` | hongdan completed |
| `chodanComplete` | `+0.16` | chodan completed |
| `gwang3Complete` | `+0.20` | reach 3 gwang |
| `gwang4Complete` | `+0.32` | reach 4 gwang |
| `gwang5Complete` | `+0.50` | reach 5 gwang |
| `jjob` | `+0.04` | event delta count |
| `ppuk` | `+0.03` | event delta count |
| `yeonPpuk` | `+0.07` | event delta count |
| `ddadak` | `+0.04` | event delta count |
| `jabbeok` | `+0.04` | event delta count |
| `pansseul` | `+0.05` | event delta count |
| `shaking` | `+0.05` | event delta count |
| `bomb` | `+0.05` | event delta count |
| `goScoreUp` | `+0.12` | go decision and score improved |
| `goWin` | `+0.20` | episode done, not truncated, go declared, win |
| `goLoss` | `-0.12` | episode done, not truncated, go declared, loss |
| `doublePiGet` | `+0.03` | per new double-pi card |
| `pi10Reached` | `+0.06` | crossing pi >= 10 |
| `piBakRisk` | `-0.10` | entering pi-bak risk state |
| `gwangBakRisk` | `-0.10` | entering gwang-bak risk state |
| `piBakSuffered` | `-0.40` | suffered pi-bak at loss |
| `gwangBakSuffered` | `-0.40` | suffered gwang-bak at loss |
| `mongBakSuffered` | `-0.40` | suffered mong-bak at loss |

Notes:
- `goWin/goLoss` and `*BakSuffered` require `done && !truncated`.
- All shaping terms are aggregated into `reward_phase1_shaping_total`.

---

## 3) Stage 1/2/3 (Phase1 Single-Actor)

## 3.1 Default stage schedule (`runtime_phase1_ppo.json`)

| Stage | Update Range (default total=2000) | Opponent Policy |
|---|---|---|
| 1 | `1..500` | `H-J1:0.50|H-Gemini:0.20|H-AntiCL:0.20|H-J2:0.10` |
| 2 | `501..1000` | `H-J2:0.40|H-NEXg:0.30|H-AntiCL:0.20|H-GPT:0.10` |
| 3 | `1001..2000` | `H-CL:0.40|H-J2:0.30|H-NEXg:0.20|H-GPT:0.10` |

Stage index is derived from `opponent_policy_schedule.start_update`.

## 3.2 Stage-dependent schedules in trainer

Constants:
- `STAGE_DECAY_RATIO = 0.70`

### Learning rate by stage

Single-actor stage LR is linearly interpolated inside each stage:

```text
stage_lr_start = learning_rate * 0.70^(stage_index-1)
stage_lr_end   = learning_rate * 0.70^stage_index   (if not last stage)
stage_lr_end   = learning_rate_final                (if last stage)
```

Current default config (`learning_rate=3e-4`, `learning_rate_final=1.2e-4`) gives:
- Stage1: `3.0e-4 -> 2.1e-4`
- Stage2: `2.1e-4 -> 1.47e-4`
- Stage3: `1.47e-4 -> 1.2e-4`

### Go explore by stage

```text
go_explore_prob_now = go_explore_prob * 0.70^(stage_index-1)
```

Current default (`go_explore_prob=0.25`):
- Stage1: `0.25`
- Stage2: `0.175`
- Stage3: `0.1225`

---

## 4) What Changes at Stage Boundary

When stage/policy changes:
- env workers are recreated with new opponent policy
- recurrent state is reinitialized
- stage best metrics/checkpoint context resets
- 1000-game rolling windows are cleared (no cross-stage mixing)
- stage patience counter resets

Files emitted per stage best:
- `best_stage1.pt`, `best_metrics_stage1.json`
- `best_stage2.pt`, `best_metrics_stage2.json`
- `best_stage3.pt`, `best_metrics_stage3.json`

---

## 5) Best Update / Early Stop Logic (Stage-Scoped)

`best_eval_ready` requires:
- `metric_window_games >= 300`

Best comparison (`is_better_update`) summary:
1. Prefer candidate if catastrophic rate passes safety filter (`<= 0.27`) while current best does not.
2. Compare score:
   - `best_score = mean_final_gold_diff_1000 + win_adjust(win_rate_1000)`
   - margin threshold: `1000`
3. If close, lower catastrophic rate wins.
4. If still tied, higher win rate wins.
5. If still tied, earlier update wins.

Early stop trigger (stage scoped):
- `stage_update_index >= early_stop_min_updates`
- `stage_updates_since_best >= early_stop_patience_updates`

---

## 6) Reward-Related Default Values (Current Phase1 Runtime)

From `ppo/configs/runtime_phase1_ppo.json`:

| Key | Value |
|---|---:|
| `reward_scale` | `0.001` |
| `downside_penalty_scale` | `0.0005` |
| `terminal_bonus_scale` | `0.003` |
| `terminal_win_bonus` | `1.0` |
| `terminal_loss_penalty` | `1.0` |
| `go_action_bonus` | `0.2` |
| `go_explore_prob` | `0.25` |
| `catastrophic_loss_threshold` | `-7000` |
| `catastrophic_penalty_scale` | `0.0002` |

---

## 7) Fast Log Checklist (No Re-analysis Needed)

For one run directory `logs/PPO/<run_name>/`:
- `run_meta.json`: exact runtime and overrides
- `metrics_update_*.json`: per-checkpoint metrics
- `best_metrics_stage*.json`: stage best snapshots
- `latest.pt`, `checkpoint_update_*.pt`, `best_stage*.pt`

Recommended read order:
1. `run_meta.json`
2. last `metrics_update_*.json`
3. `best_metrics_stage1/2/3.json`
4. compare stage trend by `win_rate_1000`, `mean_final_gold_diff_1000`, `catastrophic_loss_rate_1000`

