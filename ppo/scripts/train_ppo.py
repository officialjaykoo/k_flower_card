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
    "best_gold_tolerance",
    "early_stop_patience_updates",
    "early_stop_min_updates",
    "hidden_size",
    "device",
    "log_every_updates",
    "save_every_updates",
    "output_dir",
]


def fail(msg: str) -> None:
    raise RuntimeError(str(msg))


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
        if runtime["control_actor"] not in ("human", "ai"):
            fail(f"invalid control_actor: {runtime['control_actor']}")
    else:
        control_actor = str(cfg.get("control_actor", "")).strip()
        opponent_policy = str(cfg.get("opponent_policy", "")).strip()
        if control_actor:
            fail("control_actor must be empty in selfplay mode")
        if opponent_policy:
            fail("opponent_policy must be empty in selfplay mode")

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
    if abs(runtime["reward_scale"]) <= 0:
        fail("reward_scale must be non-zero")

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
    runtime["best_gold_tolerance"] = as_nonneg_float(cfg, "best_gold_tolerance")
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
            stderr=subprocess.PIPE,
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
        try:
            resp = json.loads(line)
        except Exception as err:
            fail(f"bridge returned invalid JSON: worker={self.worker_id}, line={line.strip()}, err={err}")
        if not isinstance(resp, dict):
            fail(f"bridge response is not object: worker={self.worker_id}, response={resp}")
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
            if self.proc.stderr:
                self.proc.stderr.close()
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


def quantile(values: List[float], q: float) -> float:
    if len(values) <= 0:
        return 0.0
    s = sorted(float(v) for v in values)
    qn = max(0.0, min(1.0, float(q)))
    idx = int(math.floor((len(s) - 1) * qn))
    return float(s[idx])


def tail_mean(values: List[float], ratio: float) -> float:
    if len(values) <= 0:
        return 0.0
    r = max(0.01, min(1.0, float(ratio)))
    s = sorted(float(v) for v in values)
    n = max(1, int(math.ceil(len(s) * r)))
    return float(sum(s[:n]) / n)


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


def is_better_update(candidate: Dict[str, Any], best: Dict[str, Any], gold_tolerance: float) -> bool:
    cand_gold = metric_as_float(candidate, "mean_final_gold_diff_50", -1e18)
    best_gold = metric_as_float(best, "mean_final_gold_diff_50", -1e18)
    tol = max(0.0, float(gold_tolerance))

    if cand_gold > best_gold + tol:
        return True
    if cand_gold < best_gold - tol:
        return False

    cand_cata = metric_as_float(candidate, "catastrophic_loss_rate_50", 1.0)
    best_cata = metric_as_float(best, "catastrophic_loss_rate_50", 1.0)
    if cand_cata < best_cata - 1e-9:
        return True
    if cand_cata > best_cata + 1e-9:
        return False

    cand_win = metric_as_float(candidate, "win_rate_50", 0.0)
    best_win = metric_as_float(best, "win_rate_50", 0.0)
    if cand_win > best_win + 1e-9:
        return True
    if cand_win < best_win - 1e-9:
        return False

    cand_update = int(candidate.get("update", 0))
    best_update = int(best.get("update", 0))
    return cand_update < best_update


def explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if float(var_y.item()) <= 1e-12:
        return 0.0
    return float((1.0 - torch.var(y_true - y_pred) / var_y).item())


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
        completed_returns: List[float] = []
        completed_lengths: List[int] = []
        completed_gold_diff: List[float] = []
        opponent_episode_counts_total: Dict[str, int] = {}
        global_episode_index = num_envs

        train_start = time.time()
        best_metrics: Optional[Dict[str, Any]] = None
        best_update = 0
        updates_since_best = 0
        best_metrics_path = os.path.join(output_dir, "best_metrics.json")

        if os.path.exists(best_metrics_path):
            try:
                with open(best_metrics_path, "r", encoding="utf-8-sig") as f:
                    best_obj = json.load(f)
                if isinstance(best_obj, dict) and "mean_final_gold_diff_50" in best_obj:
                    best_metrics = dict(best_obj)
                    best_update = int(best_metrics.get("update", 0))
            except Exception as err:
                fail(f"failed to read best_metrics.json: {best_metrics_path}, err={err}")

        if resume_payload is not None:
            ck_metrics = resume_payload.get("metrics") or {}
            if best_metrics is None and "mean_final_gold_diff_50" in ck_metrics:
                best_metrics = dict(ck_metrics)
                try:
                    best_update = int(best_metrics.get("update", resume_update))
                except Exception:
                    best_update = resume_update
        updates_since_best = max(0, resume_update - best_update)

        for update in range(resume_update + 1, runtime["total_updates"] + 1):
            if runtime["total_updates"] <= 1:
                lr_now = float(runtime["learning_rate_final"])
            else:
                progress = float(update - 1) / float(runtime["total_updates"] - 1)
                lr_now = float(runtime["learning_rate"] + progress * (runtime["learning_rate_final"] - runtime["learning_rate"]))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            rollout_obs: List[torch.Tensor] = []
            rollout_masks: List[torch.Tensor] = []
            rollout_actions: List[torch.Tensor] = []
            rollout_logprobs: List[torch.Tensor] = []
            rollout_values: List[torch.Tensor] = []
            rollout_rewards: List[torch.Tensor] = []
            rollout_dones: List[torch.Tensor] = []
            update_cat_penalty_total = 0.0
            update_cat_event_count = 0
            opponent_episode_counts_update: Dict[str, int] = {}

            update_begin = time.time()
            for _ in range(runtime["rollout_steps"]):
                rollout_obs.append(next_obs.detach().clone())
                rollout_masks.append(next_mask.detach().clone())

                with torch.no_grad():
                    logits, value = model(next_obs)
                    dist = masked_categorical(logits, next_mask)
                    actions = dist.sample()
                    logprob = dist.log_prob(actions)

                rollout_actions.append(actions.detach().clone())
                rollout_logprobs.append(logprob.detach().clone())
                rollout_values.append(value.squeeze(-1).detach().clone())

                step_rewards: List[float] = []
                step_dones: List[float] = []
                next_obs_rows: List[List[float]] = []
                next_mask_rows: List[List[float]] = []

                action_list = actions.detach().cpu().tolist()
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
                            update_cat_penalty_total += float(cat_penalty)
                            update_cat_event_count += 1

                    episode_returns[env_idx] += reward
                    episode_lengths[env_idx] += 1
                    global_steps += 1

                    if done:
                        completed_returns.append(episode_returns[env_idx])
                        completed_lengths.append(episode_lengths[env_idx])
                        completed_gold_diff.append(float(reply.info.get("final_gold_diff", 0.0)))
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

            last_policy_loss = 0.0
            last_value_loss = 0.0
            last_entropy = 0.0
            last_approx_kl = 0.0
            approx_kl_sum = 0.0
            approx_kl_count = 0
            epochs_ran = 0
            early_stop_kl = False

            model.train()
            for epoch_idx in range(runtime["ppo_epochs"]):
                epochs_ran = epoch_idx + 1
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
                        last_policy_loss = float(policy_loss.item())
                        last_value_loss = float(value_loss.item())
                        last_entropy = float(entropy.item())
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

            with torch.no_grad():
                _, predicted_values_post = model(b_obs)
                ev = explained_variance(b_returns, predicted_values_post.squeeze(-1))

            wall = max(1e-6, time.time() - update_begin)
            fps = int((runtime["rollout_steps"] * num_envs) / wall)

            mean_return = float(sum(completed_returns[-50:]) / max(1, len(completed_returns[-50:])))
            mean_len = float(sum(completed_lengths[-50:]) / max(1, len(completed_lengths[-50:])))
            mean_gold_diff = float(sum(completed_gold_diff[-50:]) / max(1, len(completed_gold_diff[-50:])))
            window_gold_diff = completed_gold_diff[-50:]
            wins_50 = sum(1 for v in window_gold_diff if float(v) > 0)
            losses_50 = sum(1 for v in window_gold_diff if float(v) < 0)
            draws_50 = len(window_gold_diff) - wins_50 - losses_50
            denom_50 = max(1, len(window_gold_diff))
            p10_gold_diff = quantile(window_gold_diff, 0.1)
            cvar10_gold_diff = tail_mean(window_gold_diff, 0.1)
            catastrophic_rate_50 = rate_at_or_below(window_gold_diff, runtime["catastrophic_loss_threshold"])

            metrics = {
                "update": update,
                "global_steps": global_steps,
                "episodes_finished": finished_episode_offset + len(completed_returns),
                "fps": fps,
                "policy_loss": last_policy_loss,
                "value_loss": last_value_loss,
                "entropy": last_entropy,
                "learning_rate": lr_now,
                "learning_rate_final": runtime["learning_rate_final"],
                "approx_kl": last_approx_kl,
                "target_kl": runtime["target_kl"],
                "early_stop_kl": early_stop_kl,
                "ppo_epochs_ran": epochs_ran,
                "explained_variance": ev,
                "mean_return_50": mean_return,
                "mean_ep_len_50": mean_len,
                "mean_final_gold_diff_50": mean_gold_diff,
                "win_rate_50": float(wins_50 / denom_50),
                "loss_rate_50": float(losses_50 / denom_50),
                "draw_rate_50": float(draws_50 / denom_50),
                "p10_final_gold_diff_50": p10_gold_diff,
                "cvar10_final_gold_diff_50": cvar10_gold_diff,
                "catastrophic_loss_threshold": runtime["catastrophic_loss_threshold"],
                "catastrophic_loss_rate_50": catastrophic_rate_50,
                "catastrophic_penalty_scale": runtime["catastrophic_penalty_scale"],
                "catastrophic_penalty_total_update": update_cat_penalty_total,
                "catastrophic_event_count_update": update_cat_event_count,
                "value_clip_coef": runtime["value_clip_coef"],
                "opponent_episode_counts_update": {
                    k: int(opponent_episode_counts_update[k])
                    for k in sorted(opponent_episode_counts_update.keys())
                },
                "opponent_episode_counts_total": {
                    k: int(opponent_episode_counts_total[k])
                    for k in sorted(opponent_episode_counts_total.keys())
                },
                "elapsed_sec": train_elapsed_offset_sec + (time.time() - train_start),
            }

            is_best_update = False
            if best_metrics is None or is_better_update(
                candidate=metrics,
                best=best_metrics,
                gold_tolerance=runtime["best_gold_tolerance"],
            ):
                best_metrics = dict(metrics)
                best_update = int(update)
                updates_since_best = 0
                is_best_update = True
            else:
                updates_since_best += 1

            metrics["is_best_update"] = is_best_update
            metrics["best_update_so_far"] = int(best_update)
            metrics["best_mean_final_gold_diff_50_so_far"] = metric_as_float(
                best_metrics or {},
                "mean_final_gold_diff_50",
                metrics["mean_final_gold_diff_50"],
            )
            metrics["updates_since_best"] = int(updates_since_best)
            metrics["best_gold_tolerance"] = runtime["best_gold_tolerance"]

            stop_by_patience = (
                update >= runtime["early_stop_min_updates"]
                and updates_since_best >= runtime["early_stop_patience_updates"]
            )
            metrics["early_stop_patience_triggered"] = bool(stop_by_patience)
            metrics["early_stop_patience_updates"] = runtime["early_stop_patience_updates"]
            metrics["early_stop_min_updates"] = runtime["early_stop_min_updates"]

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

            if is_best_update:
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
                with open(os.path.join(output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
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
                print(
                    json.dumps(
                        {
                            "type": "ppo_early_stop",
                            "reason": "no_improvement_patience",
                            "update": update,
                            "best_update": best_update,
                            "updates_since_best": updates_since_best,
                            "best_mean_final_gold_diff_50": metric_as_float(
                                best_metrics or {},
                                "mean_final_gold_diff_50",
                                metrics["mean_final_gold_diff_50"],
                            ),
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
