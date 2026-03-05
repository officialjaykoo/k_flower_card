#!/usr/bin/env python3
from __future__ import annotations

"""
train_ppo.py
- Masked PPO trainer for Matgo via ppo_env_bridge.mjs
- Strict runtime validation + fail-fast error handling
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Execution order (training):
# run_ppo.ps1 -> train_ppo.py -> BridgeEnv(node) -> ppo_env_bridge.mjs -> engine/ai runtime.
# This file is the single source of truth for PPO update logic and checkpoint lifecycle.

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    print("torch is required. install first: pip install torch")
    sys.exit(1)


# ------------------------------
# Runtime schema / scoring knobs
# ------------------------------

REQUIRED_KEYS = [
    "format_version",
    "phase",
    "seed",
    "training_mode",
    "node_bin",
    "env_bridge_script",
    "rule_key",
    "first_turn_policy",
    "fixed_first_turn",
    "max_episode_steps",
    "reward_scale",
    "downside_penalty_scale",
    "terminal_bonus_scale",
    "terminal_win_bonus",
    "terminal_loss_penalty",
    "go_action_bonus",
    "go_explore_prob",
    "env_workers",
    "total_updates",
    "rollout_steps",
    "minibatch_size",
    "ppo_epochs",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "learning_rate",
    "learning_rate_final",
    "entropy_coef",
    "value_coef",
    "value_clip_coef",
    "target_kl",
    "max_grad_norm",
    "catastrophic_loss_threshold",
    "catastrophic_penalty_scale",
    "early_stop_patience_updates",
    "early_stop_min_updates",
    "hidden_size",
    "device",
    "log_every_updates",
    "save_every_updates",
    "output_dir",
]

GO_ACTION_INDEX = 18  # ACTION_DIM=26, option index start(18), OPTION_ORDER[0] == "go"
ROLLING_METRIC_GAMES = 1000
MIN_METRIC_WINDOW_GAMES = 300
BEST_CATA_FILTER = 0.27
BEST_WIN_BASELINE = 0.30
BEST_WIN_CLAMP = 0.10
BEST_WIN_DEADZONE = 0.02
BEST_WIN_TO_GOLD_SCALE = 35000.0
BEST_SCORE_MARGIN = 150.0


def fail(msg: str) -> None:
    raise RuntimeError(str(msg))


# ------------------------------
# Runtime parsing / validation
# ------------------------------

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        fail(f"runtime config must be a JSON object: {path}")
    return raw


def require_keys(cfg: Dict[str, Any], keys: List[str], path: str) -> None:
    for key in keys:
        if key not in cfg:
            fail(f"runtime config missing required key '{key}': {path}")


def as_non_empty_str(cfg: Dict[str, Any], key: str) -> str:
    v = str(cfg.get(key, "")).strip()
    if not v:
        fail(f"runtime key '{key}' must be non-empty string")
    return v


def as_pos_int(cfg: Dict[str, Any], key: str) -> int:
    raw = cfg.get(key, None)
    try:
        n = int(raw)
    except Exception:
        fail(f"runtime key '{key}' must be integer, got={raw}")
    if n <= 0:
        fail(f"runtime key '{key}' must be > 0, got={n}")
    return n


def as_nonneg_float(cfg: Dict[str, Any], key: str) -> float:
    raw = cfg.get(key, None)
    try:
        n = float(raw)
    except Exception:
        fail(f"runtime key '{key}' must be float, got={raw}")
    if not math.isfinite(n) or n < 0:
        fail(f"runtime key '{key}' must be finite and >= 0, got={n}")
    return n


def as_finite_float(cfg: Dict[str, Any], key: str) -> float:
    raw = cfg.get(key, None)
    try:
        n = float(raw)
    except Exception:
        fail(f"runtime key '{key}' must be float, got={raw}")
    if not math.isfinite(n):
        fail(f"runtime key '{key}' must be finite, got={n}")
    return n


def parse_opponent_policy_schedule(cfg: Dict[str, Any], training_mode: str, default_policy: str) -> List[Dict[str, Any]]:
    raw = cfg.get("opponent_policy_schedule", None)

    if training_mode != "single_actor":
        if raw is None:
            return []
        if isinstance(raw, list) and len(raw) <= 0:
            return []
        fail("opponent_policy_schedule is only allowed in single_actor mode")

    if raw is None:
        return [{"start_update": 1, "opponent_policy": default_policy}]

    if not isinstance(raw, list) or len(raw) <= 0:
        fail("opponent_policy_schedule must be a non-empty array")

    out: List[Dict[str, Any]] = []
    prev_start = 0
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            fail(f"opponent_policy_schedule[{idx}] must be object")

        allowed = {"start_update", "opponent_policy"}
        keys = set(item.keys())
        extra = sorted(list(keys - allowed))
        missing = sorted(list(allowed - keys))
        if len(missing) > 0:
            fail(f"opponent_policy_schedule[{idx}] missing keys: {missing}")
        if len(extra) > 0:
            fail(f"opponent_policy_schedule[{idx}] has unknown keys: {extra}")

        start_raw = item.get("start_update")
        try:
            start_update = int(start_raw)
        except Exception:
            fail(f"opponent_policy_schedule[{idx}].start_update must be integer, got={start_raw}")
        if start_update <= 0:
            fail(f"opponent_policy_schedule[{idx}].start_update must be > 0, got={start_update}")
        if start_update <= prev_start:
            fail(
                f"opponent_policy_schedule must be strictly increasing by start_update: "
                f"index={idx}, start_update={start_update}, prev_start_update={prev_start}"
            )

        policy = str(item.get("opponent_policy", "")).strip()
        if not policy:
            fail(f"opponent_policy_schedule[{idx}].opponent_policy must be non-empty string")

        out.append({"start_update": int(start_update), "opponent_policy": policy})
        prev_start = start_update

    if out[0]["start_update"] != 1:
        fail(
            "opponent_policy_schedule must start at update 1 "
            f"(first start_update={out[0]['start_update']})"
        )

    return out


def resolve_opponent_policy_for_update(schedule: List[Dict[str, Any]], update: int) -> tuple[str, int]:
    if len(schedule) <= 0:
        fail("opponent_policy_schedule is empty")
    if int(update) <= 0:
        fail(f"update must be > 0 for policy schedule, got={update}")

    active_policy = ""
    active_start = 0
    for entry in schedule:
        start_update = int(entry["start_update"])
        if start_update > int(update):
            break
        active_policy = str(entry["opponent_policy"])
        active_start = start_update

    if not active_policy:
        fail(f"no opponent policy resolved for update={update}")
    return active_policy, active_start


def resolve_stage_index_for_start_update(schedule: List[Dict[str, Any]], start_update: int) -> int:
    target = int(start_update)
    for idx, entry in enumerate(schedule):
        if int(entry.get("start_update", 0)) == target:
            return int(idx + 1)
    fail(f"stage start_update not found in opponent_policy_schedule: start_update={start_update}")
    return 0


def normalize_runtime(cfg: Dict[str, Any], cfg_path: str) -> Dict[str, Any]:
    require_keys(cfg, REQUIRED_KEYS, cfg_path)
    runtime: Dict[str, Any] = {}

    runtime["format_version"] = as_non_empty_str(cfg, "format_version")
    if runtime["format_version"] != "ppo_runtime_phase1_v1":
        fail(f"invalid format_version: {runtime['format_version']} (expected ppo_runtime_phase1_v1)")

    runtime["phase"] = as_non_empty_str(cfg, "phase")
    runtime["seed"] = as_non_empty_str(cfg, "seed")
    runtime["training_mode"] = as_non_empty_str(cfg, "training_mode").lower()
    runtime["node_bin"] = as_non_empty_str(cfg, "node_bin")
    runtime["env_bridge_script"] = as_non_empty_str(cfg, "env_bridge_script")
    runtime["rule_key"] = as_non_empty_str(cfg, "rule_key")
    runtime["control_actor"] = ""
    runtime["opponent_policy"] = ""
    runtime["first_turn_policy"] = as_non_empty_str(cfg, "first_turn_policy").lower()
    runtime["fixed_first_turn"] = as_non_empty_str(cfg, "fixed_first_turn").lower()
    runtime["device"] = as_non_empty_str(cfg, "device")

    if runtime["training_mode"] not in ("single_actor", "selfplay"):
        fail(f"invalid training_mode: {runtime['training_mode']} (allowed: single_actor|selfplay)")
    if runtime["training_mode"] == "single_actor":
        runtime["control_actor"] = as_non_empty_str(cfg, "control_actor").lower()
        runtime["opponent_policy"] = as_non_empty_str(cfg, "opponent_policy")
        runtime["opponent_policy_schedule"] = parse_opponent_policy_schedule(
            cfg=cfg,
            training_mode=runtime["training_mode"],
            default_policy=runtime["opponent_policy"],
        )
        if runtime["control_actor"] not in ("human", "ai"):
            fail(f"invalid control_actor: {runtime['control_actor']}")
    else:
        control_actor = str(cfg.get("control_actor", "")).strip()
        opponent_policy = str(cfg.get("opponent_policy", "")).strip()
        if control_actor:
            fail("control_actor must be empty in selfplay mode")
        if opponent_policy:
            fail("opponent_policy must be empty in selfplay mode")
        runtime["opponent_policy_schedule"] = parse_opponent_policy_schedule(
            cfg=cfg,
            training_mode=runtime["training_mode"],
            default_policy="",
        )

    if runtime["first_turn_policy"] not in ("alternate", "fixed"):
        fail(f"invalid first_turn_policy: {runtime['first_turn_policy']}")
    if runtime["fixed_first_turn"] not in ("human", "ai"):
        fail(f"invalid fixed_first_turn: {runtime['fixed_first_turn']}")

    runtime["max_episode_steps"] = as_pos_int(cfg, "max_episode_steps")
    runtime["reward_scale"] = as_finite_float(cfg, "reward_scale")
    runtime["downside_penalty_scale"] = as_nonneg_float(cfg, "downside_penalty_scale")
    runtime["terminal_bonus_scale"] = as_nonneg_float(cfg, "terminal_bonus_scale")
    runtime["terminal_win_bonus"] = as_nonneg_float(cfg, "terminal_win_bonus")
    runtime["terminal_loss_penalty"] = as_nonneg_float(cfg, "terminal_loss_penalty")
    runtime["go_action_bonus"] = as_nonneg_float(cfg, "go_action_bonus")
    runtime["go_explore_prob"] = as_nonneg_float(cfg, "go_explore_prob")
    if abs(runtime["reward_scale"]) <= 0:
        fail("reward_scale must be non-zero")
    if runtime["go_explore_prob"] > 1.0:
        fail(f"go_explore_prob must be <= 1.0, got={runtime['go_explore_prob']}")

    runtime["env_workers"] = as_pos_int(cfg, "env_workers")
    runtime["total_updates"] = as_pos_int(cfg, "total_updates")
    runtime["rollout_steps"] = as_pos_int(cfg, "rollout_steps")
    runtime["minibatch_size"] = as_pos_int(cfg, "minibatch_size")
    runtime["ppo_epochs"] = as_pos_int(cfg, "ppo_epochs")

    runtime["gamma"] = as_finite_float(cfg, "gamma")
    runtime["gae_lambda"] = as_finite_float(cfg, "gae_lambda")
    runtime["clip_coef"] = as_nonneg_float(cfg, "clip_coef")
    runtime["learning_rate"] = as_nonneg_float(cfg, "learning_rate")
    runtime["learning_rate_final"] = as_nonneg_float(cfg, "learning_rate_final")
    runtime["entropy_coef"] = as_nonneg_float(cfg, "entropy_coef")
    runtime["value_coef"] = as_nonneg_float(cfg, "value_coef")
    runtime["value_clip_coef"] = as_nonneg_float(cfg, "value_clip_coef")
    runtime["target_kl"] = as_nonneg_float(cfg, "target_kl")
    runtime["max_grad_norm"] = as_nonneg_float(cfg, "max_grad_norm")
    runtime["catastrophic_loss_threshold"] = as_finite_float(cfg, "catastrophic_loss_threshold")
    runtime["catastrophic_penalty_scale"] = as_nonneg_float(cfg, "catastrophic_penalty_scale")
    runtime["early_stop_patience_updates"] = as_pos_int(cfg, "early_stop_patience_updates")
    runtime["early_stop_min_updates"] = as_pos_int(cfg, "early_stop_min_updates")
    runtime["hidden_size"] = as_pos_int(cfg, "hidden_size")
    runtime["log_every_updates"] = as_pos_int(cfg, "log_every_updates")
    runtime["save_every_updates"] = as_pos_int(cfg, "save_every_updates")
    runtime["output_dir"] = as_non_empty_str(cfg, "output_dir")
    resume_cfg = str(cfg.get("resume_checkpoint", "")).strip()
    runtime["resume_checkpoint"] = resume_cfg

    if not (0 < runtime["gamma"] <= 1):
        fail(f"gamma out of range: {runtime['gamma']}")
    if not (0 < runtime["gae_lambda"] <= 1):
        fail(f"gae_lambda out of range: {runtime['gae_lambda']}")
    if runtime["learning_rate_final"] > runtime["learning_rate"]:
        fail(
            "learning_rate_final must be <= learning_rate "
            f"({runtime['learning_rate_final']} > {runtime['learning_rate']})"
        )
    if runtime["target_kl"] <= 0:
        fail(f"target_kl must be > 0, got={runtime['target_kl']}")
    if runtime["early_stop_min_updates"] > runtime["total_updates"]:
        fail(
            "early_stop_min_updates must be <= total_updates "
            f"({runtime['early_stop_min_updates']} > {runtime['total_updates']})"
        )
    if runtime["minibatch_size"] > runtime["rollout_steps"] * runtime["env_workers"]:
        fail(
            "minibatch_size must be <= rollout_steps * env_workers "
            f"({runtime['minibatch_size']} > {runtime['rollout_steps'] * runtime['env_workers']})"
        )

    bridge_abs = os.path.abspath(runtime["env_bridge_script"])
    if not os.path.exists(bridge_abs):
        fail(f"env_bridge_script not found: {runtime['env_bridge_script']}")
    runtime["env_bridge_script"] = bridge_abs
    if runtime["resume_checkpoint"]:
        runtime["resume_checkpoint"] = os.path.abspath(runtime["resume_checkpoint"])
    return runtime


# ------------------------------
# Bridge protocol + environment IO
# ------------------------------

@dataclass
class EnvReply:
    obs: List[float]
    action_mask: List[int]
    reward: float
    done: bool
    info: Dict[str, Any]


def validate_env_reply_shape(
    reply: EnvReply,
    obs_dim: int,
    action_dim: int,
    *,
    worker_id: int,
    stage: str,
    allow_empty_mask: bool,
) -> None:
    if len(reply.obs) != obs_dim:
        fail(
            f"obs dim mismatch from bridge: worker={worker_id}, stage={stage}, "
            f"got={len(reply.obs)}, expected={obs_dim}"
        )
    if len(reply.action_mask) != action_dim:
        fail(
            f"action mask dim mismatch from bridge: worker={worker_id}, stage={stage}, "
            f"got={len(reply.action_mask)}, expected={action_dim}"
        )

    for idx, raw in enumerate(reply.obs):
        try:
            v = float(raw)
        except Exception:
            fail(f"obs value is not numeric: worker={worker_id}, stage={stage}, index={idx}, value={raw}")
        if not math.isfinite(v):
            fail(f"obs value is non-finite: worker={worker_id}, stage={stage}, index={idx}, value={v}")

    valid_actions = 0
    for idx, raw in enumerate(reply.action_mask):
        try:
            v = int(raw)
        except Exception:
            fail(
                f"action mask value is not integer-like: worker={worker_id}, stage={stage}, index={idx}, value={raw}"
            )
        if v not in (0, 1):
            fail(
                f"action mask must be binary 0/1: worker={worker_id}, stage={stage}, index={idx}, value={v}"
            )
        if v == 1:
            valid_actions += 1
    if not allow_empty_mask and valid_actions <= 0:
        fail(f"empty action mask from bridge: worker={worker_id}, stage={stage}")


class BridgeEnv:
    def __init__(self, runtime: Dict[str, Any], worker_id: int, repo_root: str):
        self.runtime = runtime
        self.worker_id = int(worker_id)
        seed_base = f"{runtime['seed']}|worker={self.worker_id}"
        cmd = [
            runtime["node_bin"],
            runtime["env_bridge_script"],
            "--training-mode",
            runtime["training_mode"],
            "--phase",
            str(runtime["phase"]),
            "--seed-base",
            seed_base,
            "--rule-key",
            runtime["rule_key"],
            "--max-steps",
            str(runtime["max_episode_steps"]),
            "--reward-scale",
            str(runtime["reward_scale"]),
            "--downside-penalty-scale",
            str(runtime["downside_penalty_scale"]),
            "--terminal-bonus-scale",
            str(runtime["terminal_bonus_scale"]),
            "--terminal-win-bonus",
            str(runtime["terminal_win_bonus"]),
            "--terminal-loss-penalty",
            str(runtime["terminal_loss_penalty"]),
            "--go-action-bonus",
            str(runtime["go_action_bonus"]),
            "--first-turn-policy",
            runtime["first_turn_policy"],
            "--fixed-first-turn",
            runtime["fixed_first_turn"],
        ]
        if runtime["training_mode"] == "single_actor":
            cmd.extend(
                [
                    "--control-actor",
                    runtime["control_actor"],
                    "--opponent-policy",
                    runtime["opponent_policy"],
                ]
            )
        self.proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            fail(f"failed to open stdio for worker={self.worker_id}")

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.proc.poll() is not None:
            stderr_text = ""
            if self.proc.stderr is not None:
                try:
                    stderr_text = self.proc.stderr.read()
                except Exception:
                    stderr_text = ""
            fail(
                f"bridge process already exited: worker={self.worker_id}, code={self.proc.returncode}, stderr={stderr_text}"
            )

        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.proc.stdin.flush()

        noise_samples: List[str] = []
        noise_count = 0
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr_text = ""
                if self.proc.stderr is not None:
                    try:
                        stderr_text = self.proc.stderr.read()
                    except Exception:
                        stderr_text = ""
                fail(
                    f"bridge response missing: worker={self.worker_id}, code={self.proc.poll()}, stderr={stderr_text}"
                )

            raw_line = line.strip()
            try:
                resp = json.loads(line)
            except Exception:
                noise_count += 1
                if len(noise_samples) < 3:
                    noise_samples.append(raw_line[:300])
                if noise_count >= 20:
                    fail(
                        "bridge protocol noise overflow: "
                        f"worker={self.worker_id}, cmd={payload.get('cmd')}, samples={noise_samples}"
                    )
                continue

            if not isinstance(resp, dict) or "ok" not in resp:
                noise_count += 1
                if len(noise_samples) < 3:
                    noise_samples.append(raw_line[:300])
                if noise_count >= 20:
                    fail(
                        "bridge protocol object noise overflow: "
                        f"worker={self.worker_id}, cmd={payload.get('cmd')}, samples={noise_samples}"
                    )
                continue

            if not resp.get("ok", False):
                fail(f"bridge error: worker={self.worker_id}, detail={resp.get('error')}")
            return resp

    def reset(self, episode: int) -> EnvReply:
        resp = self._request({"cmd": "reset", "episode": int(episode)})
        obs = resp.get("obs")
        mask = resp.get("action_mask")
        info = resp.get("info") or {}
        if not isinstance(obs, list) or not isinstance(mask, list):
            fail(f"invalid reset payload from bridge: worker={self.worker_id}")
        return EnvReply(obs=obs, action_mask=mask, reward=0.0, done=False, info=info)

    def step(self, action: int) -> EnvReply:
        resp = self._request({"cmd": "step", "action": int(action)})
        obs = resp.get("obs")
        mask = resp.get("action_mask")
        reward = float(resp.get("reward", 0.0))
        done = bool(resp.get("done", False))
        info = resp.get("info") or {}
        if not isinstance(obs, list) or not isinstance(mask, list):
            fail(f"invalid step payload from bridge: worker={self.worker_id}")
        if not math.isfinite(reward):
            fail(f"non-finite reward from bridge: worker={self.worker_id}, reward={reward}")
        return EnvReply(obs=obs, action_mask=mask, reward=reward, done=done, info=info)

    def close(self) -> None:
        try:
            self._request({"cmd": "close"})
        except Exception:
            pass
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ------------------------------
# PPO model / tensor helpers
# ------------------------------

class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        return self.policy_head(h), self.value_head(h)


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.distributions.Categorical:
    if logits.ndim != 2 or action_mask.ndim != 2:
        fail(f"masked_categorical expects 2D tensors, got logits={tuple(logits.shape)}, mask={tuple(action_mask.shape)}")
    if logits.shape != action_mask.shape:
        fail(f"logits/mask shape mismatch: logits={tuple(logits.shape)}, mask={tuple(action_mask.shape)}")

    valid_counts = action_mask.sum(dim=1)
    if torch.any(valid_counts <= 0):
        bad_rows = torch.nonzero(valid_counts <= 0).view(-1).tolist()
        fail(f"empty action mask rows: {bad_rows}")

    masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)


def tensor_from_rows(rows: List[List[float]], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if len(rows) <= 0:
        fail("tensor_from_rows got empty rows")
    t = torch.tensor(rows, dtype=dtype, device=device)
    if t.ndim != 2:
        fail(f"tensor_from_rows expected 2D result, got shape={tuple(t.shape)}")
    if not torch.isfinite(t).all():
        fail("tensor_from_rows produced non-finite tensor")
    return t


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------------
# Checkpoint / metric helpers
# ------------------------------

def load_resume_payload(path: str, device: torch.device) -> Dict[str, Any]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        fail(f"resume checkpoint not found: {abs_path}")
    payload = torch.load(abs_path, map_location=device)
    if not isinstance(payload, dict):
        fail(f"resume checkpoint payload is not object: {abs_path}")
    if "model_state_dict" not in payload or not isinstance(payload["model_state_dict"], dict):
        fail(f"resume checkpoint missing model_state_dict: {abs_path}")
    if "optimizer_state_dict" not in payload or not isinstance(payload["optimizer_state_dict"], dict):
        fail(f"resume checkpoint missing optimizer_state_dict: {abs_path}")
    update_raw = payload.get("update", None)
    try:
        update = int(update_raw)
    except Exception:
        fail(f"invalid resume update value: {update_raw}")
    if update <= 0:
        fail(f"resume update must be > 0, got={update}")
    payload["_resume_path"] = abs_path
    payload["_resume_update"] = update
    return payload


def save_checkpoint(
    output_dir: str,
    filename: str,
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    runtime: Dict[str, Any],
    update: int,
    obs_dim: int,
    action_dim: int,
    metrics: Dict[str, Any],
) -> None:
    payload = {
        "update": int(update),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "runtime": runtime,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    path = os.path.join(output_dir, filename)
    torch.save(payload, path)


def rate_at_or_below(values: List[float], threshold: float) -> float:
    if len(values) <= 0:
        return 0.0
    t = float(threshold)
    hit = 0
    for v in values:
        if float(v) <= t:
            hit += 1
    return float(hit / len(values))


def metric_as_float(metrics: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def _best_win_adjust(win_rate: float) -> float:
    dw = max(-BEST_WIN_CLAMP, min(BEST_WIN_CLAMP, float(win_rate) - BEST_WIN_BASELINE))
    if abs(dw) <= BEST_WIN_DEADZONE:
        return 0.0
    return float(BEST_WIN_TO_GOLD_SCALE * math.copysign(abs(dw) - BEST_WIN_DEADZONE, dw))


def _best_score(metrics: Dict[str, Any]) -> float:
    gold = metric_as_float(metrics, "mean_final_gold_diff_1000", -1e18)
    win = metric_as_float(metrics, "win_rate_1000", 0.0)
    return float(gold + _best_win_adjust(win))


def is_better_update(candidate: Dict[str, Any], best: Dict[str, Any]) -> bool:
    cand_cata = metric_as_float(candidate, "catastrophic_loss_rate_1000", 1.0)
    best_cata = metric_as_float(best, "catastrophic_loss_rate_1000", 1.0)
    cand_safe = cand_cata <= BEST_CATA_FILTER
    best_safe = best_cata <= BEST_CATA_FILTER
    if cand_safe and not best_safe:
        return True
    if not cand_safe and best_safe:
        return False

    cand_score = _best_score(candidate)
    best_score = _best_score(best)
    if cand_score > best_score + BEST_SCORE_MARGIN:
        return True
    if cand_score < best_score - BEST_SCORE_MARGIN:
        return False

    if cand_cata < best_cata - 1e-9:
        return True
    if cand_cata > best_cata + 1e-9:
        return False

    cand_win = metric_as_float(candidate, "win_rate_1000", 0.0)
    best_win = metric_as_float(best, "win_rate_1000", 0.0)
    if cand_win > best_win + 1e-9:
        return True
    if cand_win < best_win - 1e-9:
        return False

    cand_update = int(candidate.get("update", 0))
    best_update = int(best.get("update", 0))
    return cand_update < best_update


def train(
    runtime: Dict[str, Any],
    output_dir: str,
    repo_root: str,
    resume_payload: Optional[Dict[str, Any]] = None,
) -> None:
    # High-level flow:
    # 1) create env workers
    # 2) collect rollout
    # 3) compute GAE/returns
    # 4) PPO optimization
    # 5) save latest/best/checkpoint + metrics
    ensure_dir(output_dir)
    random_seed = int("".join(ch for ch in str(runtime["seed"]) if ch.isdigit())[:9] or "13")
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device(runtime["device"])
    num_envs = runtime["env_workers"]

    active_opponent_policy = ""
    active_policy_start_update = 0
    if runtime["training_mode"] == "single_actor":
        active_opponent_policy, active_policy_start_update = resolve_opponent_policy_for_update(
            schedule=runtime["opponent_policy_schedule"],
            update=1,
        )
        runtime["opponent_policy"] = active_opponent_policy

    envs: List[BridgeEnv] = [BridgeEnv(runtime, i, repo_root) for i in range(num_envs)]
    try:
        reset_replies = [env.reset(i) for i, env in enumerate(envs)]
        obs_rows = [r.obs for r in reset_replies]
        mask_rows = [r.action_mask for r in reset_replies]
        obs_dim = len(obs_rows[0])
        action_dim = len(mask_rows[0])
        if obs_dim <= 0 or action_dim <= 0:
            fail(f"invalid dimensions from bridge: obs_dim={obs_dim}, action_dim={action_dim}")

        for idx, r in enumerate(reset_replies):
            validate_env_reply_shape(
                r,
                obs_dim=obs_dim,
                action_dim=action_dim,
                worker_id=idx,
                stage="reset_init",
                allow_empty_mask=False,
            )

        model = PolicyValueNet(obs_dim=obs_dim, action_dim=action_dim, hidden_size=runtime["hidden_size"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=runtime["learning_rate"], eps=1e-5)

        resume_update = 0
        global_steps = 0
        finished_episode_offset = 0
        train_elapsed_offset_sec = 0.0
        opponent_episode_counts_total: Dict[str, int] = {}
        if resume_payload is not None:
            ck_path = str(resume_payload.get("_resume_path", ""))
            ck_update = int(resume_payload.get("_resume_update", 0))
            ck_obs_dim = int(resume_payload.get("obs_dim", 0))
            ck_action_dim = int(resume_payload.get("action_dim", 0))
            if ck_obs_dim != obs_dim or ck_action_dim != action_dim:
                fail(
                    f"resume dim mismatch: ck_obs={ck_obs_dim}, ck_action={ck_action_dim}, "
                    f"env_obs={obs_dim}, env_action={action_dim}, path={ck_path}"
                )
            ck_runtime = resume_payload.get("runtime") or {}
            ck_hidden = int(ck_runtime.get("hidden_size", 0))
            if ck_hidden != int(runtime["hidden_size"]):
                fail(
                    f"resume hidden_size mismatch: ck={ck_hidden}, runtime={runtime['hidden_size']}, path={ck_path}"
                )
            ck_mode = str(ck_runtime.get("training_mode", "")).strip().lower()
            rt_mode = str(runtime["training_mode"]).strip().lower()
            phase_text = str(runtime.get("phase", "")).strip()
            try:
                phase_num = int(phase_text)
            except Exception:
                phase_num = 1
            allow_mode_transition = (
                phase_num >= 2 and ck_mode == "single_actor" and rt_mode == "selfplay"
            )
            if ck_mode != rt_mode and not allow_mode_transition:
                fail(
                    f"resume training_mode mismatch: ck={ck_mode}, runtime={rt_mode}, path={ck_path}, phase={phase_num}"
                )
            if allow_mode_transition:
                print(
                    json.dumps(
                        {
                            "type": "ppo_resume_mode_transition",
                            "phase": phase_num,
                            "from_mode": ck_mode,
                            "to_mode": rt_mode,
                            "resume_path": ck_path,
                        },
                        ensure_ascii=False,
                    )
                )
            if ck_update >= int(runtime["total_updates"]):
                fail(
                    f"resume update must be smaller than total_updates: ck_update={ck_update}, "
                    f"total_updates={runtime['total_updates']}, path={ck_path}"
                )
            try:
                model.load_state_dict(resume_payload["model_state_dict"], strict=True)
            except Exception as err:
                fail(f"failed to load model_state_dict from resume checkpoint ({ck_path}): {err}")
            try:
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            except Exception as err:
                fail(f"failed to load optimizer_state_dict from resume checkpoint ({ck_path}): {err}")
            ck_metrics = resume_payload.get("metrics") or {}
            try:
                global_steps = max(0, int(ck_metrics.get("global_steps", 0)))
            except Exception:
                global_steps = 0
            try:
                finished_episode_offset = max(0, int(ck_metrics.get("episodes_finished", 0)))
            except Exception:
                finished_episode_offset = 0
            try:
                train_elapsed_offset_sec = max(0.0, float(ck_metrics.get("elapsed_sec", 0.0)))
            except Exception:
                train_elapsed_offset_sec = 0.0
            raw_opp_total = ck_metrics.get("opponent_episode_counts_total", {})
            if isinstance(raw_opp_total, dict):
                for k, v in raw_opp_total.items():
                    key = str(k).strip()
                    if not key:
                        continue
                    try:
                        opponent_episode_counts_total[key] = max(0, int(v))
                    except Exception:
                        continue
            resume_update = ck_update
            print(
                json.dumps(
                    {
                        "type": "ppo_resume",
                        "resume_path": ck_path,
                        "resume_update": resume_update,
                        "resume_global_steps": global_steps,
                        "resume_episodes_finished": finished_episode_offset,
                    },
                    ensure_ascii=False,
                )
            )

        next_obs = tensor_from_rows(obs_rows, dtype=torch.float32, device=device)
        next_mask = tensor_from_rows(mask_rows, dtype=torch.float32, device=device)

        episode_returns = [0.0 for _ in range(num_envs)]
        episode_lengths = [0 for _ in range(num_envs)]
        episode_go_opportunities = [0 for _ in range(num_envs)]
        episode_go_attempts = [0 for _ in range(num_envs)]
        episode_go_not_selected = [0 for _ in range(num_envs)]
        completed_returns: List[float] = []
        completed_lengths: List[int] = []
        completed_gold_diff: List[float] = []
        completed_go_opportunity_games: List[int] = []
        completed_go_games: List[int] = []
        completed_go_attempts: List[int] = []
        completed_go_not_selected: List[int] = []
        completed_go_failures: List[int] = []
        global_episode_index = num_envs

        train_start = time.time()
        stage_best_metrics: Optional[Dict[str, Any]] = None
        stage_best_update = 0
        stage_updates_since_best = 0
        active_stage_index = 1
        active_stage_start_update = 1

        overall_best_metrics: Optional[Dict[str, Any]] = None
        overall_best_update = 0
        overall_best_metrics_path = os.path.join(output_dir, "best_overall_metrics.json")
        legacy_best_metrics_path = os.path.join(output_dir, "best_metrics.json")

        def stage_best_metrics_path(stage_index: int) -> str:
            return os.path.join(output_dir, f"best_metrics_stage{int(stage_index)}.json")

        def stage_best_checkpoint_name(stage_index: int) -> str:
            return f"best_stage{int(stage_index)}.pt"

        if os.path.exists(overall_best_metrics_path):
            try:
                with open(overall_best_metrics_path, "r", encoding="utf-8-sig") as f:
                    best_obj = json.load(f)
                if isinstance(best_obj, dict) and "mean_final_gold_diff_1000" in best_obj:
                    overall_best_metrics = dict(best_obj)
                    overall_best_update = int(overall_best_metrics.get("update", 0))
            except Exception as err:
                fail(f"failed to read best_overall_metrics.json: {overall_best_metrics_path}, err={err}")
        elif os.path.exists(legacy_best_metrics_path):
            # Legacy pre-stage file is treated as overall-best snapshot.
            try:
                with open(legacy_best_metrics_path, "r", encoding="utf-8-sig") as f:
                    best_obj = json.load(f)
                if isinstance(best_obj, dict) and "mean_final_gold_diff_1000" in best_obj:
                    overall_best_metrics = dict(best_obj)
                    overall_best_update = int(overall_best_metrics.get("update", 0))
            except Exception as err:
                fail(f"failed to read legacy best_metrics.json: {legacy_best_metrics_path}, err={err}")

        if runtime["training_mode"] == "single_actor":
            start_update = int(resume_update + 1)
            target_policy, target_start = resolve_opponent_policy_for_update(
                schedule=runtime["opponent_policy_schedule"],
                update=start_update,
            )
            policy_changed = target_policy != active_opponent_policy
            stage_changed = int(target_start) != int(active_policy_start_update)
            if policy_changed:
                for env in envs:
                    env.close()
                runtime["opponent_policy"] = target_policy
                envs = [BridgeEnv(runtime, i, repo_root) for i in range(num_envs)]
                reset_replies = [env.reset(i) for i, env in enumerate(envs)]
                for idx, r in enumerate(reset_replies):
                    validate_env_reply_shape(
                        r,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        worker_id=idx,
                        stage="reset_schedule_init",
                        allow_empty_mask=False,
                    )
                obs_rows = [r.obs for r in reset_replies]
                mask_rows = [r.action_mask for r in reset_replies]
                next_obs = tensor_from_rows(obs_rows, dtype=torch.float32, device=device)
                next_mask = tensor_from_rows(mask_rows, dtype=torch.float32, device=device)
                episode_returns = [0.0 for _ in range(num_envs)]
                episode_lengths = [0 for _ in range(num_envs)]
                episode_go_opportunities = [0 for _ in range(num_envs)]
                episode_go_attempts = [0 for _ in range(num_envs)]
                episode_go_not_selected = [0 for _ in range(num_envs)]
                global_episode_index = num_envs
            if policy_changed or stage_changed:
                active_opponent_policy = target_policy
                active_policy_start_update = int(target_start)

            active_stage_start_update = int(active_policy_start_update)
            active_stage_index = resolve_stage_index_for_start_update(
                schedule=runtime["opponent_policy_schedule"],
                start_update=active_stage_start_update,
            )
        else:
            active_stage_start_update = 1
            active_stage_index = 1

        active_stage_metrics_path = stage_best_metrics_path(active_stage_index)
        if os.path.exists(active_stage_metrics_path):
            try:
                with open(active_stage_metrics_path, "r", encoding="utf-8-sig") as f:
                    best_obj = json.load(f)
                if isinstance(best_obj, dict) and "mean_final_gold_diff_1000" in best_obj:
                    stage_best_metrics = dict(best_obj)
                    stage_best_update = int(stage_best_metrics.get("update", 0))
            except Exception as err:
                fail(
                    "failed to read stage best metrics: "
                    f"path={active_stage_metrics_path}, stage={active_stage_index}, err={err}"
                )

        if resume_payload is not None:
            ck_metrics = resume_payload.get("metrics") or {}
            if overall_best_metrics is None and "mean_final_gold_diff_1000" in ck_metrics:
                overall_best_metrics = dict(ck_metrics)
                try:
                    overall_best_update = int(overall_best_metrics.get("update", resume_update))
                except Exception:
                    overall_best_update = resume_update

            if stage_best_metrics is None and "mean_final_gold_diff_1000" in ck_metrics:
                ck_stage_index_raw = ck_metrics.get("stage_index", 0)
                ck_stage_start_raw = ck_metrics.get("stage_start_update", ck_metrics.get("active_opponent_policy_stage_start_update", 0))
                try:
                    ck_stage_index = int(ck_stage_index_raw)
                except Exception:
                    ck_stage_index = 0
                try:
                    ck_stage_start = int(ck_stage_start_raw)
                except Exception:
                    ck_stage_start = 0
                same_stage = (
                    (ck_stage_index > 0 and ck_stage_index == int(active_stage_index))
                    or (ck_stage_start > 0 and ck_stage_start == int(active_stage_start_update))
                )
                if same_stage:
                    stage_best_metrics = dict(ck_metrics)
                    try:
                        stage_best_update = int(stage_best_metrics.get("update", resume_update))
                    except Exception:
                        stage_best_update = resume_update

        if stage_best_update > 0:
            stage_updates_since_best = max(0, resume_update - stage_best_update)
        else:
            stage_updates_since_best = 0

        for update in range(resume_update + 1, runtime["total_updates"] + 1):
            if runtime["training_mode"] == "single_actor":
                target_policy, target_start = resolve_opponent_policy_for_update(
                    schedule=runtime["opponent_policy_schedule"],
                    update=update,
                )
                target_start = int(target_start)
                policy_changed = target_policy != active_opponent_policy
                stage_changed = target_start != int(active_policy_start_update)

                if policy_changed:
                    for env in envs:
                        env.close()
                    runtime["opponent_policy"] = target_policy
                    envs = [BridgeEnv(runtime, i, repo_root) for i in range(num_envs)]
                    reset_replies = [env.reset(global_episode_index + i) for i, env in enumerate(envs)]
                    for idx, r in enumerate(reset_replies):
                        validate_env_reply_shape(
                            r,
                            obs_dim=obs_dim,
                            action_dim=action_dim,
                            worker_id=idx,
                            stage="reset_policy_switch",
                            allow_empty_mask=False,
                        )
                    obs_rows = [r.obs for r in reset_replies]
                    mask_rows = [r.action_mask for r in reset_replies]
                    next_obs = tensor_from_rows(obs_rows, dtype=torch.float32, device=device)
                    next_mask = tensor_from_rows(mask_rows, dtype=torch.float32, device=device)
                    episode_returns = [0.0 for _ in range(num_envs)]
                    episode_lengths = [0 for _ in range(num_envs)]
                    episode_go_opportunities = [0 for _ in range(num_envs)]
                    episode_go_attempts = [0 for _ in range(num_envs)]
                    episode_go_not_selected = [0 for _ in range(num_envs)]
                    global_episode_index += num_envs
                    print(
                        json.dumps(
                            {
                                "type": "ppo_opponent_policy_switch",
                                "update": update,
                                "from_policy": active_opponent_policy,
                                "to_policy": target_policy,
                                "stage_start_update": target_start,
                            },
                            ensure_ascii=False,
                        )
                    )

                if policy_changed or stage_changed:
                    prev_stage_index = int(active_stage_index)
                    prev_stage_start = int(active_stage_start_update)
                    active_opponent_policy = target_policy
                    active_policy_start_update = target_start
                    active_stage_start_update = target_start
                    active_stage_index = resolve_stage_index_for_start_update(
                        schedule=runtime["opponent_policy_schedule"],
                        start_update=active_stage_start_update,
                    )
                    stage_best_metrics = None
                    stage_best_update = 0
                    stage_updates_since_best = 0
                    switched_stage_metrics_path = stage_best_metrics_path(active_stage_index)
                    if os.path.exists(switched_stage_metrics_path):
                        try:
                            with open(switched_stage_metrics_path, "r", encoding="utf-8-sig") as f:
                                best_obj = json.load(f)
                            if isinstance(best_obj, dict) and "mean_final_gold_diff_1000" in best_obj:
                                stage_best_metrics = dict(best_obj)
                                stage_best_update = int(stage_best_metrics.get("update", 0))
                                stage_updates_since_best = max(0, (update - 1) - stage_best_update)
                        except Exception as err:
                            fail(
                                "failed to read stage best metrics on stage switch: "
                                f"path={switched_stage_metrics_path}, update={update}, stage={active_stage_index}, err={err}"
                            )
                    if stage_changed:
                        print(
                            json.dumps(
                                {
                                    "type": "ppo_stage_switch",
                                    "update": update,
                                    "from_stage_index": prev_stage_index,
                                    "from_stage_start_update": prev_stage_start,
                                    "to_stage_index": int(active_stage_index),
                                    "to_stage_start_update": int(active_stage_start_update),
                                    "to_policy": target_policy,
                                },
                                ensure_ascii=False,
                            )
                        )
                else:
                    active_policy_start_update = target_start
                    active_stage_start_update = target_start

            if runtime["training_mode"] != "single_actor":
                active_stage_index = 1
                active_stage_start_update = 1

            if runtime["total_updates"] <= 1:
                lr_now = float(runtime["learning_rate_final"])
            else:
                progress = float(update - 1) / float(runtime["total_updates"] - 1)
                lr_now = float(runtime["learning_rate"] + progress * (runtime["learning_rate_final"] - runtime["learning_rate"]))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # 1) Rollout buffers for this update.
            rollout_obs: List[torch.Tensor] = []
            rollout_masks: List[torch.Tensor] = []
            rollout_actions: List[torch.Tensor] = []
            rollout_logprobs: List[torch.Tensor] = []
            rollout_values: List[torch.Tensor] = []
            rollout_rewards: List[torch.Tensor] = []
            rollout_dones: List[torch.Tensor] = []
            opponent_episode_counts_update: Dict[str, int] = {}

            for _ in range(runtime["rollout_steps"]):
                rollout_obs.append(next_obs.detach().clone())
                rollout_masks.append(next_mask.detach().clone())

                with torch.no_grad():
                    logits, value = model(next_obs)
                    dist = masked_categorical(logits, next_mask)
                    sampled_actions = dist.sample()

                action_list = [int(x) for x in sampled_actions.detach().cpu().tolist()]
                for env_idx in range(num_envs):
                    go_mask_raw = float(next_mask[env_idx, GO_ACTION_INDEX].item())
                    go_mask = int(go_mask_raw)
                    if go_mask not in (0, 1):
                        fail(
                            f"invalid go action mask value: worker={env_idx}, update={update}, "
                            f"step={global_steps}, value={go_mask_raw}"
                        )
                    if go_mask != 1:
                        continue
                    episode_go_opportunities[env_idx] += 1
                    if action_list[env_idx] == GO_ACTION_INDEX:
                        episode_go_attempts[env_idx] += 1
                        continue
                    if random.random() < runtime["go_explore_prob"]:
                        action_list[env_idx] = GO_ACTION_INDEX
                        episode_go_attempts[env_idx] += 1
                    else:
                        episode_go_not_selected[env_idx] += 1

                actions = torch.tensor(action_list, dtype=torch.int64, device=device)
                with torch.no_grad():
                    logprob = dist.log_prob(actions)

                rollout_actions.append(actions.detach().clone())
                rollout_logprobs.append(logprob.detach().clone())
                rollout_values.append(value.squeeze(-1).detach().clone())

                step_rewards: List[float] = []
                step_dones: List[float] = []
                next_obs_rows: List[List[float]] = []
                next_mask_rows: List[List[float]] = []

                for env_idx, env in enumerate(envs):
                    reply = env.step(int(action_list[env_idx]))
                    validate_env_reply_shape(
                        reply,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        worker_id=env_idx,
                        stage="step",
                        allow_empty_mask=bool(reply.done),
                    )
                    done = bool(reply.done)
                    reward = float(reply.reward)
                    if done:
                        final_gold_diff = float(reply.info.get("final_gold_diff", 0.0))
                        if final_gold_diff <= runtime["catastrophic_loss_threshold"]:
                            cat_gap = runtime["catastrophic_loss_threshold"] - final_gold_diff
                            cat_penalty = cat_gap * runtime["catastrophic_penalty_scale"]
                            reward -= cat_penalty

                    episode_returns[env_idx] += reward
                    episode_lengths[env_idx] += 1
                    global_steps += 1

                    if done:
                        completed_returns.append(episode_returns[env_idx])
                        completed_lengths.append(episode_lengths[env_idx])
                        final_gold_diff_done = float(reply.info.get("final_gold_diff", 0.0))
                        completed_gold_diff.append(final_gold_diff_done)
                        go_attempts_ep = int(episode_go_attempts[env_idx])
                        go_opportunities_ep = int(episode_go_opportunities[env_idx])
                        go_not_selected_ep = int(episode_go_not_selected[env_idx])
                        go_game = 1 if go_attempts_ep > 0 else 0
                        go_opportunity_game = 1 if go_opportunities_ep > 0 else 0
                        go_failure = 1 if (go_game == 1 and final_gold_diff_done < 0.0) else 0
                        completed_go_attempts.append(go_attempts_ep)
                        completed_go_not_selected.append(go_not_selected_ep)
                        completed_go_games.append(go_game)
                        completed_go_opportunity_games.append(go_opportunity_game)
                        completed_go_failures.append(go_failure)
                        if runtime["training_mode"] == "single_actor":
                            opp_policy = str(reply.info.get("opponent_policy", "")).strip()
                            if not opp_policy:
                                fail(
                                    "missing opponent_policy in bridge done info "
                                    f"(worker={env_idx}, update={update}, step={global_steps})"
                                )
                            opponent_episode_counts_update[opp_policy] = (
                                int(opponent_episode_counts_update.get(opp_policy, 0)) + 1
                            )
                            opponent_episode_counts_total[opp_policy] = (
                                int(opponent_episode_counts_total.get(opp_policy, 0)) + 1
                            )
                        episode_returns[env_idx] = 0.0
                        episode_lengths[env_idx] = 0
                        episode_go_opportunities[env_idx] = 0
                        episode_go_attempts[env_idx] = 0
                        episode_go_not_selected[env_idx] = 0

                        reset_reply = env.reset(global_episode_index)
                        validate_env_reply_shape(
                            reset_reply,
                            obs_dim=obs_dim,
                            action_dim=action_dim,
                            worker_id=env_idx,
                            stage="reset_after_done",
                            allow_empty_mask=False,
                        )
                        global_episode_index += 1
                        next_obs_rows.append(reset_reply.obs)
                        next_mask_rows.append(reset_reply.action_mask)
                        step_dones.append(1.0)
                    else:
                        next_obs_rows.append(reply.obs)
                        next_mask_rows.append(reply.action_mask)
                        step_dones.append(0.0)

                    step_rewards.append(reward)

                rollout_rewards.append(torch.tensor(step_rewards, dtype=torch.float32, device=device))
                rollout_dones.append(torch.tensor(step_dones, dtype=torch.float32, device=device))
                next_obs = tensor_from_rows(next_obs_rows, dtype=torch.float32, device=device)
                next_mask = tensor_from_rows(next_mask_rows, dtype=torch.float32, device=device)

            # 2) Build batch tensors + GAE/returns.
            with torch.no_grad():
                _, next_value = model(next_obs)
                next_value = next_value.squeeze(-1)

            obs_t = torch.stack(rollout_obs, dim=0)
            mask_t = torch.stack(rollout_masks, dim=0)
            actions_t = torch.stack(rollout_actions, dim=0)
            old_logprob_t = torch.stack(rollout_logprobs, dim=0)
            values_t = torch.stack(rollout_values, dim=0)
            rewards_t = torch.stack(rollout_rewards, dim=0)
            dones_t = torch.stack(rollout_dones, dim=0)

            advantages = torch.zeros_like(rewards_t, device=device)
            last_gae = torch.zeros(num_envs, dtype=torch.float32, device=device)
            for t in reversed(range(runtime["rollout_steps"])):
                if t == runtime["rollout_steps"] - 1:
                    next_values = next_value
                else:
                    next_values = values_t[t + 1]
                nonterminal = 1.0 - dones_t[t]
                delta = rewards_t[t] + runtime["gamma"] * next_values * nonterminal - values_t[t]
                last_gae = delta + runtime["gamma"] * runtime["gae_lambda"] * nonterminal * last_gae
                advantages[t] = last_gae
            returns_t = advantages + values_t

            b_obs = obs_t.reshape(-1, obs_dim)
            b_mask = mask_t.reshape(-1, action_dim)
            b_actions = actions_t.reshape(-1)
            b_old_logprob = old_logprob_t.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_returns = returns_t.reshape(-1)
            b_values = values_t.reshape(-1)

            adv_mean = b_adv.mean()
            adv_std = b_adv.std(unbiased=False)
            if float(adv_std.item()) <= 1e-8:
                b_adv = b_adv - adv_mean
            else:
                b_adv = (b_adv - adv_mean) / (adv_std + 1e-8)

            batch_size = b_obs.shape[0]
            mb_size = runtime["minibatch_size"]
            if mb_size <= 0 or mb_size > batch_size:
                fail(f"invalid minibatch size: {mb_size} (batch_size={batch_size})")

            last_approx_kl = 0.0
            approx_kl_sum = 0.0
            approx_kl_count = 0
            early_stop_kl = False

            # 3) PPO optimization epochs.
            model.train()
            for _ in range(runtime["ppo_epochs"]):
                indices = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, mb_size):
                    end = min(start + mb_size, batch_size)
                    mb_idx = indices[start:end]

                    logits, value_pred = model(b_obs[mb_idx])
                    dist = masked_categorical(logits, b_mask[mb_idx])
                    new_logprob = dist.log_prob(b_actions[mb_idx])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logprob - b_old_logprob[mb_idx])
                    adv_mb = b_adv[mb_idx]
                    pg_loss_1 = -adv_mb * ratio
                    pg_loss_2 = -adv_mb * torch.clamp(
                        ratio, 1.0 - runtime["clip_coef"], 1.0 + runtime["clip_coef"]
                    )
                    policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                    v_pred = value_pred.squeeze(-1)
                    v_old = b_values[mb_idx]
                    if runtime["value_clip_coef"] > 0:
                        v_pred_clipped = v_old + torch.clamp(
                            v_pred - v_old,
                            -runtime["value_clip_coef"],
                            runtime["value_clip_coef"],
                        )
                        value_loss_unclipped = (v_pred - b_returns[mb_idx]) ** 2
                        value_loss_clipped = (v_pred_clipped - b_returns[mb_idx]) ** 2
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * ((v_pred - b_returns[mb_idx]) ** 2).mean()

                    loss = (
                        policy_loss
                        + runtime["value_coef"] * value_loss
                        - runtime["entropy_coef"] * entropy
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), runtime["max_grad_norm"])
                    optimizer.step()

                    with torch.no_grad():
                        approx_kl_val = float((b_old_logprob[mb_idx] - new_logprob).mean().item())
                        last_approx_kl = approx_kl_val
                        approx_kl_sum += approx_kl_val
                        approx_kl_count += 1
                        if approx_kl_val > runtime["target_kl"]:
                            early_stop_kl = True
                            break
                if early_stop_kl:
                    break

            if approx_kl_count > 0:
                last_approx_kl = float(approx_kl_sum / approx_kl_count)

            # 4) Rolling metrics over latest N finished games.
            window_gold_diff = completed_gold_diff[-ROLLING_METRIC_GAMES:]
            mean_gold_diff = float(sum(window_gold_diff) / max(1, len(window_gold_diff)))
            wins_1000 = sum(1 for v in window_gold_diff if float(v) > 0)
            denom_1000 = max(1, len(window_gold_diff))
            catastrophic_rate_1000 = rate_at_or_below(window_gold_diff, runtime["catastrophic_loss_threshold"])
            window_go_opportunity_games = completed_go_opportunity_games[-ROLLING_METRIC_GAMES:]
            window_go_games = completed_go_games[-ROLLING_METRIC_GAMES:]
            window_go_attempts = completed_go_attempts[-ROLLING_METRIC_GAMES:]
            window_go_not_selected = completed_go_not_selected[-ROLLING_METRIC_GAMES:]
            window_go_failures = completed_go_failures[-ROLLING_METRIC_GAMES:]
            go_opportunity_games = int(sum(window_go_opportunity_games))
            go_games = int(sum(window_go_games))
            go_attempts = int(sum(window_go_attempts))
            go_not_selected = int(sum(window_go_not_selected))
            go_failures = int(sum(window_go_failures))
            episodes_finished_total = int(finished_episode_offset + len(completed_returns))
            metric_window_games = len(window_gold_diff)
            best_eval_ready = metric_window_games >= MIN_METRIC_WINDOW_GAMES
            stage_update_index = int(update - int(active_stage_start_update) + 1)

            metrics = {
                "update": update,
                "global_steps": global_steps,
                "episodes_finished": episodes_finished_total,
                "elapsed_sec": train_elapsed_offset_sec + (time.time() - train_start),
                "mean_final_gold_diff_1000": mean_gold_diff,
                "win_rate_1000": float(wins_1000 / denom_1000),
                "catastrophic_loss_rate_1000": catastrophic_rate_1000,
                "metric_window_games": metric_window_games,
                "stage_index": int(active_stage_index),
                "stage_start_update": int(active_stage_start_update),
                "stage_update_index": int(stage_update_index),
                "best_update_so_far": int(stage_best_update),
                "updates_since_best": int(stage_updates_since_best),
                "stage_best_update_so_far": int(stage_best_update),
                "stage_updates_since_best": int(stage_updates_since_best),
                "overall_best_update_so_far": int(overall_best_update),
                "early_stop_patience_triggered": False,
                "go_opportunity_games": go_opportunity_games,
                "go_games": go_games,
                "go_attempts": go_attempts,
                "go_not_selected": go_not_selected,
                "go_failures": go_failures,
                "approx_kl": last_approx_kl,
                "active_opponent_policy": active_opponent_policy if runtime["training_mode"] == "single_actor" else "",
                "active_opponent_policy_stage_start_update": int(active_policy_start_update)
                if runtime["training_mode"] == "single_actor"
                else 0,
                "opponent_episode_counts_update": {
                    k: int(opponent_episode_counts_update[k])
                    for k in sorted(opponent_episode_counts_update.keys())
                },
                "opponent_episode_counts_total": {
                    k: int(opponent_episode_counts_total[k])
                    for k in sorted(opponent_episode_counts_total.keys())
                },
            }

            # 5) Stage-best / overall-best tracking + stage-scoped early-stop gate.
            is_stage_best_update = False
            is_overall_best_update = False
            if best_eval_ready:
                if stage_best_metrics is None or is_better_update(candidate=metrics, best=stage_best_metrics):
                    stage_best_update = int(update)
                    stage_updates_since_best = 0
                    is_stage_best_update = True
                else:
                    stage_updates_since_best += 1

                if overall_best_metrics is None or is_better_update(candidate=metrics, best=overall_best_metrics):
                    overall_best_update = int(update)
                    is_overall_best_update = True

            metrics["best_update_so_far"] = int(stage_best_update)
            metrics["updates_since_best"] = int(stage_updates_since_best)
            metrics["stage_best_update_so_far"] = int(stage_best_update)
            metrics["stage_updates_since_best"] = int(stage_updates_since_best)
            metrics["overall_best_update_so_far"] = int(overall_best_update)

            if is_stage_best_update:
                stage_best_metrics = dict(metrics)
            if is_overall_best_update:
                overall_best_metrics = dict(metrics)

            stop_by_patience = (
                best_eval_ready
                and
                stage_update_index >= runtime["early_stop_min_updates"]
                and stage_updates_since_best >= runtime["early_stop_patience_updates"]
            )
            metrics["early_stop_patience_triggered"] = bool(stop_by_patience)

            if update == 1 or update % runtime["log_every_updates"] == 0 or update == runtime["total_updates"]:
                print(
                    json.dumps(
                        {
                            "type": "ppo_update",
                            "phase": runtime["phase"],
                            "seed": runtime["seed"],
                            "training_mode": runtime["training_mode"],
                            **metrics,
                        },
                        ensure_ascii=False,
                    )
                )

            if is_stage_best_update:
                save_checkpoint(
                    output_dir=output_dir,
                    filename="best.pt",
                    model=model,
                    optimizer=optimizer,
                    runtime=runtime,
                    update=update,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    metrics=metrics,
                )
                save_checkpoint(
                    output_dir=output_dir,
                    filename=stage_best_checkpoint_name(active_stage_index),
                    model=model,
                    optimizer=optimizer,
                    runtime=runtime,
                    update=update,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    metrics=metrics,
                )
                with open(os.path.join(output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                with open(stage_best_metrics_path(active_stage_index), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)

            if is_overall_best_update:
                save_checkpoint(
                    output_dir=output_dir,
                    filename="best_overall.pt",
                    model=model,
                    optimizer=optimizer,
                    runtime=runtime,
                    update=update,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    metrics=metrics,
                )
                with open(overall_best_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)

            save_checkpoint(
                output_dir=output_dir,
                filename="latest.pt",
                model=model,
                optimizer=optimizer,
                runtime=runtime,
                update=update,
                obs_dim=obs_dim,
                action_dim=action_dim,
                metrics=metrics,
            )

            if update % runtime["save_every_updates"] == 0 or update == runtime["total_updates"] or stop_by_patience:
                save_checkpoint(
                    output_dir=output_dir,
                    filename=f"checkpoint_update_{update}.pt",
                    model=model,
                    optimizer=optimizer,
                    runtime=runtime,
                    update=update,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    metrics=metrics,
                )
                with open(
                    os.path.join(output_dir, f"metrics_update_{update}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)

            if stop_by_patience:
                best_mean_gold = metric_as_float(
                    stage_best_metrics or {},
                    "mean_final_gold_diff_1000",
                    metrics["mean_final_gold_diff_1000"],
                )
                print(
                    json.dumps(
                        {
                            "type": "ppo_early_stop",
                            "reason": "no_improvement_patience",
                            "update": update,
                            "stage_index": int(active_stage_index),
                            "stage_start_update": int(active_stage_start_update),
                            "stage_update_index": int(stage_update_index),
                            "best_update": stage_best_update,
                            "updates_since_best": stage_updates_since_best,
                            "best_mean_final_gold_diff_1000": best_mean_gold,
                            "best_score": _best_score(stage_best_metrics or metrics),
                            "patience_updates": runtime["early_stop_patience_updates"],
                            "min_updates": runtime["early_stop_min_updates"],
                        },
                        ensure_ascii=False,
                    )
                )
                break

    finally:
        for env in envs:
            env.close()


# ------------------------------
# CLI entrypoint
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked PPO trainer for Matgo")
    parser.add_argument(
        "--runtime-config",
        required=True,
        help="Path to runtime JSON config",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional override for runtime output_dir",
    )
    parser.add_argument(
        "--seed",
        default="",
        help="Optional override for runtime seed",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Optional resume checkpoint path (overrides runtime resume_checkpoint)",
    )
    parser.add_argument(
        "--total-updates",
        type=int,
        default=None,
        help="Optional override for runtime total_updates (>0)",
    )
    parser.add_argument(
        "--log-every-updates",
        type=int,
        default=None,
        help="Optional override for runtime log_every_updates (>0)",
    )
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=None,
        help="Optional override for runtime save_every_updates (>0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(os.getcwd())
    runtime_path = os.path.abspath(args.runtime_config)
    if not os.path.exists(runtime_path):
        fail(f"runtime config not found: {runtime_path}")

    runtime_raw = read_json(runtime_path)
    runtime = normalize_runtime(runtime_raw, runtime_path)

    if str(args.seed).strip():
        runtime["seed"] = str(args.seed).strip()

    if args.total_updates is not None:
        if int(args.total_updates) <= 0:
            fail(f"--total-updates must be > 0, got={args.total_updates}")
        runtime["total_updates"] = int(args.total_updates)
    if args.log_every_updates is not None:
        if int(args.log_every_updates) <= 0:
            fail(f"--log-every-updates must be > 0, got={args.log_every_updates}")
        runtime["log_every_updates"] = int(args.log_every_updates)
    if args.save_every_updates is not None:
        if int(args.save_every_updates) <= 0:
            fail(f"--save-every-updates must be > 0, got={args.save_every_updates}")
        runtime["save_every_updates"] = int(args.save_every_updates)
    if runtime["early_stop_min_updates"] > runtime["total_updates"]:
        fail(
            "early_stop_min_updates must be <= total_updates "
            f"({runtime['early_stop_min_updates']} > {runtime['total_updates']})"
        )

    resume_path = str(args.resume_checkpoint).strip() or str(runtime.get("resume_checkpoint", "")).strip()
    phase_raw = str(runtime.get("phase", "")).strip()
    try:
        phase_num = int(phase_raw)
    except Exception:
        phase_num = 1
    if phase_num >= 2 and not resume_path:
        fail(
            f"phase{phase_num} requires resume checkpoint from previous phase "
            f"(set runtime 'resume_checkpoint' or --resume-checkpoint)"
        )

    resume_payload = None
    if resume_path:
        resume_payload = load_resume_payload(resume_path, torch.device(runtime["device"]))

    output_dir = str(args.output_dir).strip() or runtime["output_dir"]
    output_dir = os.path.abspath(output_dir)

    ensure_dir(output_dir)
    run_meta = {
        "runtime_config": runtime_path,
        "output_dir": output_dir,
        "started_at_unix": time.time(),
        "runtime": runtime,
        "resume_checkpoint": str(resume_path or ""),
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    train(runtime=runtime, output_dir=output_dir, repo_root=repo_root, resume_payload=resume_payload)


if __name__ == "__main__":
    main()
