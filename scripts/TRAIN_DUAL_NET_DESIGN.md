# train_dual_net design notes

## Goal
- Replace separate `01_train_policy.py` + `02_train_value.py` training with one dual-head trainer.
- Keep current pipeline usable while improving data/feature quality and GPU usage.
- Optimize for expected value (EV), not only win/loss.

## Current implementation status
- File: `scripts/train_dual_net.py`
- Architecture:
  - Token branch: `EmbeddingBag`
  - Numeric branch: MLP projection
  - Shared trunk: deep MLP with normalization
  - Heads: policy, go/stop policy, value
- Training:
  - AMP supported on CUDA
  - Early stopping
  - JSONL -> `.pt` cache pipeline
  - Go/Stop policy and value weights are separated

## Feature strategy
- Base context:
  - phase, decision type, turn order, deck/hand/go counts, candidate count
- Score-critical features:
  - bak risks (pi/gwang/mong)
  - dan states (hong/cheong/cho), godori
  - go tiers (3+/4+/5+/6+), shaking multiplier signal
  - nagari and carry-over multiplier (when present)
- Economy:
  - gold self/opp, steal gold/pi, go/stop efficiency signals

## Value target strategy
- Primary direction: EV shaping with game outcome context.
- Target is bounded with `tanh(...)`.
- Reward shaping currently uses available fields:
  - `nagari`, `bakEscape`, `goDecision`, `eventFrequency`
- Long-term target:
  - direct gold EV target (`gold_self - gold_opp`) as primary objective

## Evaluation policy
- Primary metric:
  - average gold delta per game
  - cumulative gold delta over 1000 games
- Secondary metric:
  - decisive win rate

## Migration plan
1. Keep current 00-05 loop stable.
2. Run dual-net in parallel experiments.
3. Promote only if champion match improves on primary gold metrics repeatedly.
4. After stability, make dual-net default training path.
