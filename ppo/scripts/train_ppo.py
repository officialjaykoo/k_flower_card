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
    "env_workers",
    "total_updates",
    "rollout_steps",
    "minibatch_size",
    "ppo_epochs",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "learning_rate",
    "entropy_coef",
    "value_coef",
    "max_grad_norm",
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
    runtime["terminal_bonus_scale"] = as_finite_float(cfg, "terminal_bonus_scale")
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
    runtime["entropy_coef"] = as_nonneg_float(cfg, "entropy_coef")
    runtime["value_coef"] = as_nonneg_float(cfg, "value_coef")
    runtime["max_grad_norm"] = as_nonneg_float(cfg, "max_grad_norm")
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
            if len(r.obs) != obs_dim:
                fail(f"obs dim mismatch at reset worker={idx}: {len(r.obs)} vs {obs_dim}")
            if len(r.action_mask) != action_dim:
                fail(f"mask dim mismatch at reset worker={idx}: {len(r.action_mask)} vs {action_dim}")

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
        global_episode_index = num_envs

        train_start = time.time()

        for update in range(resume_update + 1, runtime["total_updates"] + 1):
            rollout_obs: List[torch.Tensor] = []
            rollout_masks: List[torch.Tensor] = []
            rollout_actions: List[torch.Tensor] = []
            rollout_logprobs: List[torch.Tensor] = []
            rollout_values: List[torch.Tensor] = []
            rollout_rewards: List[torch.Tensor] = []
            rollout_dones: List[torch.Tensor] = []

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
                    reward = float(reply.reward)
                    done = bool(reply.done)

                    episode_returns[env_idx] += reward
                    episode_lengths[env_idx] += 1
                    global_steps += 1

                    if done:
                        completed_returns.append(episode_returns[env_idx])
                        completed_lengths.append(episode_lengths[env_idx])
                        completed_gold_diff.append(float(reply.info.get("final_gold_diff", 0.0)))
                        episode_returns[env_idx] = 0.0
                        episode_lengths[env_idx] = 0

                        reset_reply = env.reset(global_episode_index)
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
                    value_loss = ((v_pred - b_returns[mb_idx]) ** 2).mean()

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
                        last_approx_kl = float((b_old_logprob[mb_idx] - new_logprob).mean().item())

            with torch.no_grad():
                predicted_values = b_values
                ev = explained_variance(b_returns, predicted_values)

            wall = max(1e-6, time.time() - update_begin)
            fps = int((runtime["rollout_steps"] * num_envs) / wall)

            mean_return = float(sum(completed_returns[-50:]) / max(1, len(completed_returns[-50:])))
            mean_len = float(sum(completed_lengths[-50:]) / max(1, len(completed_lengths[-50:])))
            mean_gold_diff = float(sum(completed_gold_diff[-50:]) / max(1, len(completed_gold_diff[-50:])))

            metrics = {
                "update": update,
                "global_steps": global_steps,
                "episodes_finished": finished_episode_offset + len(completed_returns),
                "fps": fps,
                "policy_loss": last_policy_loss,
                "value_loss": last_value_loss,
                "entropy": last_entropy,
                "approx_kl": last_approx_kl,
                "explained_variance": ev,
                "mean_return_50": mean_return,
                "mean_ep_len_50": mean_len,
                "mean_final_gold_diff_50": mean_gold_diff,
                "elapsed_sec": train_elapsed_offset_sec + (time.time() - train_start),
            }

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

            if update % runtime["save_every_updates"] == 0 or update == runtime["total_updates"]:
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
