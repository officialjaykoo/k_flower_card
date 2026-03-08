#!/usr/bin/env python3
from __future__ import annotations

"""
duel_ppo.py
- Evaluate PPO checkpoint vs heuristic v5(H-CL) with multi-worker bridge env.
- Strict runtime config parsing + fail-fast errors with context.

Execution Flow Map:
1) parse_args()/main(): runtime bootstrap + override processing
2) load_model(): checkpoint load and shape validation
3) run_block(): multi-worker duel rollout for one fixed seat
4) summarize_block(): aggregate duel metrics

File Layout Map (top-down):
1) runtime parsing/normalization helpers
2) bridge env + model inference helpers
3) duel block runner + summary aggregation
4) CLI entrypoint
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Execution order (duel):
# run_duel_ppo.ps1 -> duel_ppo.py -> BridgeEnv(node) -> ppo_env_bridge.mjs -> engine/ai runtime.

try:
    import torch
    import torch.nn as nn
except Exception:
    print("torch is required. install first: pip install torch")
    sys.exit(1)


# ------------------------------
# Runtime schema / constants
# ------------------------------

REQUIRED_KEYS = [
    "format_version",
    "seed",
    "node_bin",
    "env_bridge_script",
    "checkpoint_path",
    "rule_key",
    "opponent_policy",
    "games",
    "workers",
    "max_episode_steps",
    "first_turn_policy",
    "fixed_first_turn",
    "switch_seats",
    "policy_mode",
    "temperature",
    "reward_scale",
    "downside_penalty_scale",
    "terminal_bonus_scale",
    "terminal_win_bonus",
    "terminal_loss_penalty",
    "go_action_bonus",
    "catastrophic_loss_threshold",
    "device",
    "result_out",
]

PLAY_SLOTS = 10
MATCH_SLOTS = 8
GO_ACTION_INDEX = PLAY_SLOTS + MATCH_SLOTS  # OPTION_ORDER[0] == "go" in ppo_env_bridge.mjs


def fail(msg: str) -> None:
    raise RuntimeError(str(msg))


# ------------------------------
# Runtime parsing / validation
# ------------------------------

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        fail(f"runtime config must be object: {path}")
    return obj


def require_keys(cfg: Dict[str, Any], keys: List[str], cfg_path: str) -> None:
    for k in keys:
        if k not in cfg:
            fail(f"runtime config missing required key '{k}': {cfg_path}")


def as_non_empty_str(cfg: Dict[str, Any], key: str) -> str:
    v = str(cfg.get(key, "")).strip()
    if not v:
        fail(f"runtime key '{key}' must be non-empty string")
    return v


def as_positive_int(cfg: Dict[str, Any], key: str) -> int:
    raw = cfg.get(key, None)
    try:
        n = int(raw)
    except Exception:
        fail(f"runtime key '{key}' must be integer, got={raw}")
    if n <= 0:
        fail(f"runtime key '{key}' must be > 0, got={n}")
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


def parse_bool_12(cfg: Dict[str, Any], key: str) -> bool:
    raw = str(cfg.get(key, "")).strip()
    if raw == "1":
        return True
    if raw == "2":
        return False
    fail(f"runtime key '{key}' must be '1' or '2', got={raw}")


def quantile(values: List[float], q: float) -> float:
    if len(values) <= 0:
        return 0.0
    s = sorted(values)
    idx = int(math.floor((len(s) - 1) * max(0.0, min(1.0, float(q)))))
    return float(s[idx])


def mean(values: List[float]) -> float:
    if len(values) <= 0:
        return 0.0
    return float(sum(values) / len(values))


def tail_mean(values: List[float], ratio: float) -> float:
    if len(values) <= 0:
        return 0.0
    r = max(0.01, min(1.0, float(ratio)))
    s = sorted(values)
    n = max(1, int(math.ceil(len(s) * r)))
    return float(sum(s[:n]) / n)


def rate_at_or_below(values: List[float], threshold: float) -> float:
    if len(values) <= 0:
        return 0.0
    hit = 0
    t = float(threshold)
    for v in values:
        if float(v) <= t:
            hit += 1
    return float(hit / len(values))


def other_actor(actor: str) -> str:
    if actor == "human":
        return "ai"
    if actor == "ai":
        return "human"
    fail(f"invalid actor: {actor}")


def normalize_runtime(cfg: Dict[str, Any], cfg_path: str) -> Dict[str, Any]:
    require_keys(cfg, REQUIRED_KEYS, cfg_path)
    rt: Dict[str, Any] = {}

    rt["format_version"] = as_non_empty_str(cfg, "format_version")
    if rt["format_version"] != "ppo_duel_runtime_v1":
        fail(f"invalid format_version: {rt['format_version']} (expected ppo_duel_runtime_v1)")

    rt["seed"] = as_non_empty_str(cfg, "seed")
    rt["node_bin"] = as_non_empty_str(cfg, "node_bin")
    rt["env_bridge_script"] = as_non_empty_str(cfg, "env_bridge_script")
    rt["checkpoint_path"] = as_non_empty_str(cfg, "checkpoint_path")
    rt["rule_key"] = as_non_empty_str(cfg, "rule_key")
    rt["opponent_policy"] = as_non_empty_str(cfg, "opponent_policy")
    rt["games"] = as_positive_int(cfg, "games")
    rt["workers"] = as_positive_int(cfg, "workers")
    rt["max_episode_steps"] = as_positive_int(cfg, "max_episode_steps")
    rt["first_turn_policy"] = as_non_empty_str(cfg, "first_turn_policy").lower()
    rt["fixed_first_turn"] = as_non_empty_str(cfg, "fixed_first_turn").lower()
    rt["switch_seats"] = parse_bool_12(cfg, "switch_seats")
    rt["policy_mode"] = as_non_empty_str(cfg, "policy_mode").lower()
    rt["temperature"] = as_finite_float(cfg, "temperature")
    rt["reward_scale"] = as_finite_float(cfg, "reward_scale")
    rt["downside_penalty_scale"] = as_finite_float(cfg, "downside_penalty_scale")
    rt["terminal_bonus_scale"] = as_finite_float(cfg, "terminal_bonus_scale")
    rt["terminal_win_bonus"] = as_finite_float(cfg, "terminal_win_bonus")
    rt["terminal_loss_penalty"] = as_finite_float(cfg, "terminal_loss_penalty")
    rt["go_action_bonus"] = as_finite_float(cfg, "go_action_bonus")
    rt["catastrophic_loss_threshold"] = as_finite_float(cfg, "catastrophic_loss_threshold")
    rt["device"] = as_non_empty_str(cfg, "device")
    rt["result_out"] = as_non_empty_str(cfg, "result_out")

    if rt["first_turn_policy"] not in ("alternate", "fixed"):
        fail(f"invalid first_turn_policy: {rt['first_turn_policy']}")
    if rt["fixed_first_turn"] not in ("human", "ai"):
        fail(f"invalid fixed_first_turn: {rt['fixed_first_turn']}")
    if rt["policy_mode"] not in ("greedy", "sample"):
        fail(f"invalid policy_mode: {rt['policy_mode']} (allowed: greedy|sample)")
    if rt["policy_mode"] == "sample" and rt["temperature"] <= 0:
        fail(f"temperature must be > 0 for sample mode, got={rt['temperature']}")
    if rt["temperature"] <= 0:
        fail(f"temperature must be > 0, got={rt['temperature']}")
    if abs(rt["reward_scale"]) <= 0:
        fail("reward_scale must be non-zero")
    if rt["downside_penalty_scale"] < 0:
        fail("downside_penalty_scale must be >= 0")
    if rt["terminal_win_bonus"] < 0:
        fail("terminal_win_bonus must be >= 0")
    if rt["terminal_loss_penalty"] < 0:
        fail("terminal_loss_penalty must be >= 0")
    if rt["go_action_bonus"] < 0:
        fail("go_action_bonus must be >= 0")

    bridge_abs = os.path.abspath(rt["env_bridge_script"])
    if not os.path.exists(bridge_abs):
        fail(f"env_bridge_script not found: {rt['env_bridge_script']}")
    rt["env_bridge_script"] = bridge_abs

    ck_abs = os.path.abspath(rt["checkpoint_path"])
    if not os.path.exists(ck_abs):
        fail(f"checkpoint_path not found: {rt['checkpoint_path']}")
    rt["checkpoint_path"] = ck_abs

    rt["result_out"] = os.path.abspath(rt["result_out"])
    os.makedirs(os.path.dirname(rt["result_out"]), exist_ok=True)
    return rt


# ------------------------------
# Bridge protocol + model helpers
# ------------------------------

@dataclass
class EnvReply:
    obs: List[float]
    action_mask: List[int]
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]


@dataclass
class SlotState:
    obs: List[float]
    action_mask: List[int]
    episode_id: int
    step_count: int
    seed_tag: str
    go_attempt_count: int
    go_attempted: bool
    go_opportunity_count: int
    go_opportunity_seen: bool


class BridgeEnv:
    def __init__(self, runtime: Dict[str, Any], repo_root: str, worker_id: int, control_actor: str):
        self.runtime = runtime
        self.worker_id = int(worker_id)
        self.control_actor = str(control_actor)
        seed_base = f"{runtime['seed']}|duel|actor={self.control_actor}|worker={self.worker_id}"
        cmd = [
            runtime["node_bin"],
            runtime["env_bridge_script"],
            "--training-mode",
            "single_actor",
            "--phase",
            "0",
            "--seed-base",
            seed_base,
            "--rule-key",
            runtime["rule_key"],
            "--control-actor",
            self.control_actor,
            "--opponent-policy",
            runtime["opponent_policy"],
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
            fail(f"failed to open stdio: worker={self.worker_id}, actor={self.control_actor}")

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.proc.poll() is not None:
            stderr_text = ""
            if self.proc.stderr is not None:
                try:
                    stderr_text = self.proc.stderr.read()
                except Exception:
                    stderr_text = ""
            fail(
                f"bridge exited early: actor={self.control_actor}, worker={self.worker_id}, "
                f"code={self.proc.returncode}, stderr={stderr_text}"
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
                f"bridge response missing: actor={self.control_actor}, worker={self.worker_id}, "
                f"code={self.proc.poll()}, stderr={stderr_text}"
            )
        try:
            resp = json.loads(line)
        except Exception as err:
            fail(
                f"bridge invalid json: actor={self.control_actor}, worker={self.worker_id}, "
                f"line={line.strip()}, err={err}"
            )
        if not isinstance(resp, dict):
            fail(f"bridge response not object: actor={self.control_actor}, worker={self.worker_id}, resp={resp}")
        if not bool(resp.get("ok", False)):
            fail(
                f"bridge error: actor={self.control_actor}, worker={self.worker_id}, error={resp.get('error')}"
            )
        return resp

    def reset(self, episode: int) -> EnvReply:
        resp = self._request({"cmd": "reset", "episode": int(episode)})
        obs = resp.get("obs")
        mask = resp.get("action_mask")
        info = resp.get("info") or {}
        if not isinstance(obs, list) or not isinstance(mask, list):
            fail(f"invalid reset payload: actor={self.control_actor}, worker={self.worker_id}")
        return EnvReply(obs=obs, action_mask=mask, reward=0.0, done=False, truncated=False, info=info)

    def step(self, action: int) -> EnvReply:
        resp = self._request({"cmd": "step", "action": int(action)})
        obs = resp.get("obs")
        mask = resp.get("action_mask")
        reward = float(resp.get("reward", 0.0))
        done = bool(resp.get("done", False))
        truncated = bool(resp.get("truncated", False))
        info = resp.get("info") or {}
        if not isinstance(obs, list) or not isinstance(mask, list):
            fail(f"invalid step payload: actor={self.control_actor}, worker={self.worker_id}")
        if not math.isfinite(reward):
            fail(f"non-finite reward: actor={self.control_actor}, worker={self.worker_id}, reward={reward}")
        return EnvReply(obs=obs, action_mask=mask, reward=reward, done=done, truncated=truncated, info=info)

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


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        *,
        gru_num_layers: int,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        if self.gru_num_layers <= 0:
            fail(f"gru_num_layers must be > 0, got={self.gru_num_layers}")
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.gru_num_layers,
            batch_first=False,
        )

        self.policy_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_hidden_size = max(32, self.hidden_size // 2)
        self.value_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.value_hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(self.value_hidden_size, 1)

    def init_recurrent_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        b = int(batch_size)
        if b <= 0:
            fail(f"batch_size must be > 0 for recurrent state init, got={b}")
        return torch.zeros(self.gru_num_layers, b, self.hidden_size, dtype=torch.float32, device=device)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fail("forward(obs) is disabled in GRU-only duel model; use forward_gru_step")
        return torch.empty(0), torch.empty(0)

    def forward_gru_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim != 2:
            fail(f"GRU step expects obs shape [B, obs_dim], got={tuple(obs.shape)}")
        if hidden.ndim != 3:
            fail(f"GRU hidden rank mismatch: expected 3D, got={tuple(hidden.shape)}")
        b = obs.shape[0]
        if hidden.shape[0] != self.gru_num_layers or hidden.shape[1] != b or hidden.shape[2] != self.hidden_size:
            fail(
                "GRU hidden shape mismatch: "
                f"got={tuple(hidden.shape)}, expected=({self.gru_num_layers},{b},{self.hidden_size})"
            )
        x = self.obs_encoder(obs)
        out, next_hidden = self.gru(x.unsqueeze(0), hidden)
        feat = out.squeeze(0)
        return self.policy_head(feat), self.value_head(self.value_mlp(feat)), next_hidden


def select_actions(
    model: PolicyValueNet,
    obs_rows: List[List[float]],
    mask_rows: List[List[int]],
    policy_mode: str,
    temperature: float,
    device: torch.device,
    recurrent_state: torch.Tensor,
) -> Tuple[List[int], torch.Tensor]:
    if len(obs_rows) <= 0:
        fail("select_actions called with empty obs_rows")
    obs = torch.tensor(obs_rows, dtype=torch.float32, device=device)
    mask = torch.tensor(mask_rows, dtype=torch.float32, device=device)
    if obs.ndim != 2 or mask.ndim != 2:
        fail(f"tensor rank invalid: obs={tuple(obs.shape)}, mask={tuple(mask.shape)}")
    if obs.shape[0] != mask.shape[0]:
        fail(f"batch mismatch: obs={tuple(obs.shape)}, mask={tuple(mask.shape)}")
    if not torch.isfinite(obs).all():
        fail("obs tensor contains non-finite")
    if torch.any(mask.sum(dim=1) <= 0):
        bad = torch.nonzero(mask.sum(dim=1) <= 0).view(-1).tolist()
        fail(f"mask has empty legal actions at rows={bad}")

    with torch.no_grad():
        logits, _, next_recurrent_state = model.forward_gru_step(obs, recurrent_state)
        if logits.shape != mask.shape:
            fail(f"logits/mask shape mismatch: logits={tuple(logits.shape)}, mask={tuple(mask.shape)}")
        masked_logits = logits.masked_fill(mask <= 0, -1e9)

        if policy_mode == "greedy":
            acts = torch.argmax(masked_logits, dim=1)
        else:
            probs = torch.softmax(masked_logits / float(temperature), dim=1)
            dist = torch.distributions.Categorical(probs=probs)
            acts = dist.sample()
    return [int(x) for x in acts.cpu().tolist()], next_recurrent_state


def is_go_legal(action_mask: List[int]) -> bool:
    if len(action_mask) <= GO_ACTION_INDEX:
        fail(
            f"action_mask too short for GO index check: len={len(action_mask)}, go_index={GO_ACTION_INDEX}"
        )
    raw = action_mask[GO_ACTION_INDEX]
    try:
        v = int(raw)
    except Exception:
        fail(f"GO action mask value is not int-like: index={GO_ACTION_INDEX}, value={raw}")
    if v not in (0, 1):
        fail(f"GO action mask value must be 0 or 1: index={GO_ACTION_INDEX}, value={v}")
    return v == 1


def summarize_block(results: List[Dict[str, Any]], catastrophic_threshold: float) -> Dict[str, Any]:
    diffs = [float(r["gold_diff"]) for r in results]
    wins = sum(1 for v in diffs if v > 0)
    losses = sum(1 for v in diffs if v < 0)
    draws = len(diffs) - wins - losses
    truncated_games = sum(1 for r in results if bool(r.get("truncated", False)))
    go_attempts = sum(int(r.get("go_attempts", 0)) for r in results)
    go_attempt_games = sum(1 for r in results if bool(r.get("go_attempted", False)))
    go_failures = sum(1 for r in results if bool(r.get("go_failure", False)))
    go_opportunities = sum(int(r.get("go_opportunities", 0)) for r in results)
    go_opportunity_games = sum(1 for r in results if bool(r.get("go_opportunity_game", False)))
    wr = float(wins / len(diffs)) if diffs else 0.0
    lr = float(losses / len(diffs)) if diffs else 0.0
    dr = float(draws / len(diffs)) if diffs else 0.0
    go_failure_rate = float(go_failures / go_attempt_games) if go_attempt_games > 0 else 0.0
    return {
        "games": len(diffs),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wr,
        "loss_rate": lr,
        "draw_rate": dr,
        "truncated_games": int(truncated_games),
        "mean_gold_delta": mean(diffs),
        "std_gold_delta": float(statistics.pstdev(diffs)) if len(diffs) > 0 else 0.0,
        "p10_gold_delta": quantile(diffs, 0.1),
        "p50_gold_delta": quantile(diffs, 0.5),
        "p90_gold_delta": quantile(diffs, 0.9),
        "cvar10_gold_delta": tail_mean(diffs, 0.1),
        "catastrophic_loss_rate": rate_at_or_below(diffs, catastrophic_threshold),
        "catastrophic_loss_threshold": float(catastrophic_threshold),
        "go_attempts": int(go_attempts),
        "go_attempt_games": int(go_attempt_games),
        "go_failures": int(go_failures),
        "go_opportunities": int(go_opportunities),
        "go_opportunity_games": int(go_opportunity_games),
        "go_failure_rate": go_failure_rate,
    }


# ------------------------------
# Duel block execution
# ------------------------------

def run_block(
    runtime: Dict[str, Any],
    model: PolicyValueNet,
    device: torch.device,
    repo_root: str,
    control_actor: str,
    games: int,
) -> Dict[str, Any]:
    # Run one seat block (control_actor fixed) with multi-worker parallel environments.
    if games <= 0:
        return {"results": [], "summary": summarize_block([], runtime["catastrophic_loss_threshold"])}

    worker_count = min(int(runtime["workers"]), games)
    if worker_count <= 1:
        fail(f"workers must be >=2 for simulation runs, got worker_count={worker_count}")

    envs = [BridgeEnv(runtime, repo_root, worker_id=i, control_actor=control_actor) for i in range(worker_count)]
    slots: List[Optional[SlotState]] = [None for _ in range(worker_count)]
    slot_hidden: List[Optional[torch.Tensor]] = [None for _ in range(worker_count)]
    try:
        next_episode_id = 0
        completed = 0
        results: List[Dict[str, Any]] = []
        started_at = time.time()
        last_log_completed = 0

        for wi in range(worker_count):
            if next_episode_id >= games:
                break
            reply = envs[wi].reset(next_episode_id)
            slots[wi] = SlotState(
                obs=reply.obs,
                action_mask=reply.action_mask,
                episode_id=next_episode_id,
                step_count=0,
                seed_tag=str(reply.info.get("seed", "")),
                go_attempt_count=0,
                go_attempted=False,
                go_opportunity_count=0,
                go_opportunity_seen=False,
            )
            slot_hidden[wi] = model.init_recurrent_state(batch_size=1, device=device)
            next_episode_id += 1

        while completed < games:
            active_idx = [i for i, slot in enumerate(slots) if slot is not None]
            if len(active_idx) <= 0:
                fail(
                    f"no active env while duel not finished: actor={control_actor}, completed={completed}, games={games}"
                )

            obs_rows = [slots[i].obs for i in active_idx if slots[i] is not None]
            mask_rows = [slots[i].action_mask for i in active_idx if slots[i] is not None]
            hidden_batch_parts: List[torch.Tensor] = []
            for wi in active_idx:
                h = slot_hidden[wi]
                if h is None:
                    fail(f"missing recurrent hidden state for active slot: actor={control_actor}, env_i={wi}")
                hidden_batch_parts.append(h)
            recurrent_batch = torch.cat(hidden_batch_parts, dim=1)

            actions, recurrent_next_batch = select_actions(
                model=model,
                obs_rows=obs_rows,
                mask_rows=mask_rows,
                policy_mode=runtime["policy_mode"],
                temperature=runtime["temperature"],
                device=device,
                recurrent_state=recurrent_batch,
            )
            if recurrent_next_batch.shape[1] != len(active_idx):
                fail(
                    "recurrent_next_batch batch mismatch: "
                    f"got={recurrent_next_batch.shape[1]}, expected={len(active_idx)}"
                )

            for local_i, env_i in enumerate(active_idx):
                slot = slots[env_i]
                if slot is None:
                    fail(f"internal slot error: actor={control_actor}, env_i={env_i}")
                slot_hidden[env_i] = recurrent_next_batch[:, local_i : local_i + 1, :].detach().clone()
                if is_go_legal(slot.action_mask):
                    slot.go_opportunity_count += 1
                    slot.go_opportunity_seen = True
                action = actions[local_i]
                if action == GO_ACTION_INDEX:
                    slot.go_attempt_count += 1
                    slot.go_attempted = True
                reply = envs[env_i].step(action)
                slot.step_count += 1

                if reply.done:
                    final_diff = float(reply.info.get("final_gold_diff", 0.0))
                    seat = control_actor
                    opp = other_actor(control_actor)
                    results.append(
                        {
                            "episode_id": slot.episode_id,
                            "seed": slot.seed_tag,
                            "control_actor": seat,
                            "opponent_actor": opp,
                            "steps": int(reply.info.get("step", slot.step_count)),
                            "gold_diff": final_diff,
                            "control_gold": float(reply.info.get("control_gold", 0.0)),
                            "opponent_gold": float(reply.info.get("opponent_gold", 0.0)),
                            "win": final_diff > 0,
                            "loss": final_diff < 0,
                            "draw": final_diff == 0,
                            "truncated": bool(reply.truncated),
                            "go_attempts": int(slot.go_attempt_count),
                            "go_attempted": bool(slot.go_attempted),
                            "go_failure": bool(slot.go_attempted and final_diff < 0),
                            "go_opportunities": int(slot.go_opportunity_count),
                            "go_opportunity_game": bool(slot.go_opportunity_seen),
                        }
                    )
                    completed += 1

                    if next_episode_id < games:
                        rr = envs[env_i].reset(next_episode_id)
                        slots[env_i] = SlotState(
                            obs=rr.obs,
                            action_mask=rr.action_mask,
                            episode_id=next_episode_id,
                            step_count=0,
                            seed_tag=str(rr.info.get("seed", "")),
                            go_attempt_count=0,
                            go_attempted=False,
                            go_opportunity_count=0,
                            go_opportunity_seen=False,
                        )
                        slot_hidden[env_i] = model.init_recurrent_state(batch_size=1, device=device)
                        next_episode_id += 1
                    else:
                        slots[env_i] = None
                        slot_hidden[env_i] = None
                else:
                    slots[env_i] = SlotState(
                        obs=reply.obs,
                        action_mask=reply.action_mask,
                        episode_id=slot.episode_id,
                        step_count=slot.step_count,
                        seed_tag=slot.seed_tag,
                        go_attempt_count=slot.go_attempt_count,
                        go_attempted=slot.go_attempted,
                        go_opportunity_count=slot.go_opportunity_count,
                        go_opportunity_seen=slot.go_opportunity_seen,
                    )

            if completed - last_log_completed >= 100 or completed == games:
                elapsed = max(1e-6, time.time() - started_at)
                fps = int(sum((s.step_count if s is not None else 0) for s in slots if s is not None) / elapsed)
                print(
                    json.dumps(
                        {
                            "type": "duel_progress",
                            "actor": control_actor,
                            "completed_games": completed,
                            "target_games": games,
                            "elapsed_sec": elapsed,
                            "workers": worker_count,
                            "active_envs": len([s for s in slots if s is not None]),
                            "loop_fps_hint": fps,
                        },
                        ensure_ascii=False,
                    )
                )
                last_log_completed = completed

        return {
            "results": results,
            "summary": summarize_block(results, runtime["catastrophic_loss_threshold"]),
        }
    finally:
        for env in envs:
            env.close()


# ------------------------------
# Checkpoint load + CLI entrypoint
# ------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[PolicyValueNet, Dict[str, Any]]:
    obj = torch.load(checkpoint_path, map_location=device)
    if not isinstance(obj, dict):
        fail(f"checkpoint payload is not object: {checkpoint_path}")
    obs_dim = int(obj.get("obs_dim", 0))
    action_dim = int(obj.get("action_dim", 0))
    runtime = obj.get("runtime") or {}
    if obs_dim <= 0 or action_dim <= 0:
        fail(f"invalid checkpoint dims: obs_dim={obs_dim}, action_dim={action_dim}")
    hidden_size = int(runtime.get("hidden_size", 0))
    if hidden_size <= 0:
        fail(f"invalid hidden_size in checkpoint runtime: {hidden_size}")
    if "gru_num_layers" not in runtime:
        fail(f"checkpoint runtime missing gru_num_layers: {checkpoint_path}")
    gru_num_layers_raw = runtime.get("gru_num_layers")
    try:
        gru_num_layers = int(gru_num_layers_raw)
    except Exception:
        fail(f"invalid gru_num_layers in checkpoint runtime: {gru_num_layers_raw}")
    if gru_num_layers <= 0:
        fail(f"gru_num_layers must be > 0 in checkpoint runtime, got={gru_num_layers}")
    sd = obj.get("model_state_dict")
    if not isinstance(sd, dict):
        fail(f"model_state_dict missing in checkpoint: {checkpoint_path}")
    cl_keys = {"obs_encoder.0.weight", "value_mlp.0.weight"}
    missing_cl = sorted(list(cl_keys - set(sd.keys())))
    if len(missing_cl) > 0:
        fail(
            "not a CL checkpoint (missing CL architecture keys): "
            f"path={checkpoint_path}, missing={missing_cl}"
        )

    model = PolicyValueNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        gru_num_layers=gru_num_layers,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_size": hidden_size,
        "gru_num_layers": gru_num_layers,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO vs v5(H-CL) duel runner")
    p.add_argument("--runtime-config", required=True, help="Path to duel runtime JSON")
    p.add_argument("--seed", default="", help="Optional override for runtime seed")
    p.add_argument(
        "--checkpoint-path",
        default="",
        help="Optional override for runtime checkpoint_path",
    )
    p.add_argument(
        "--result-out",
        default="",
        help="Optional override for runtime result_out",
    )
    p.add_argument("--games", type=int, default=None, help="Optional override for runtime games (>0)")
    p.add_argument("--workers", type=int, default=None, help="Optional override for runtime workers (>0)")
    p.add_argument("--opponent-policy", default="", help="Optional override for runtime opponent_policy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(os.getcwd())
    cfg_path = os.path.abspath(args.runtime_config)
    if not os.path.exists(cfg_path):
        fail(f"runtime config not found: {cfg_path}")
    cfg = read_json(cfg_path)
    runtime = normalize_runtime(cfg, cfg_path)

    if str(args.seed).strip():
        runtime["seed"] = str(args.seed).strip()
    if str(args.checkpoint_path).strip():
        ck_abs = os.path.abspath(str(args.checkpoint_path).strip())
        if not os.path.exists(ck_abs):
            fail(f"checkpoint_path not found: {ck_abs}")
        runtime["checkpoint_path"] = ck_abs
    if str(args.result_out).strip():
        result_abs = os.path.abspath(str(args.result_out).strip())
        result_dir = os.path.dirname(result_abs)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        runtime["result_out"] = result_abs
    if args.games is not None:
        if int(args.games) <= 0:
            fail(f"--games must be > 0, got={args.games}")
        runtime["games"] = int(args.games)
    if args.workers is not None:
        if int(args.workers) <= 0:
            fail(f"--workers must be > 0, got={args.workers}")
        runtime["workers"] = int(args.workers)
    if str(args.opponent_policy).strip():
        runtime["opponent_policy"] = str(args.opponent_policy).strip()

    random_seed = int("".join(ch for ch in runtime["seed"] if ch.isdigit())[:9] or "13")
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device(runtime["device"])

    model, model_meta = load_model(runtime["checkpoint_path"], device)
    total_games = int(runtime["games"])
    if runtime["switch_seats"]:
        human_games = total_games // 2
        ai_games = total_games - human_games
    else:
        human_games = total_games
        ai_games = 0

    print(
        json.dumps(
            {
                "type": "duel_start",
                "seed": runtime["seed"],
                "games": total_games,
                "workers": runtime["workers"],
                "opponent_policy": runtime["opponent_policy"],
                "policy_mode": runtime["policy_mode"],
                "switch_seats": runtime["switch_seats"],
                "go_action_bonus": runtime["go_action_bonus"],
                "checkpoint": runtime["checkpoint_path"],
                "model_meta": model_meta,
            },
            ensure_ascii=False,
        )
    )

    started_at = time.time()
    block_human = run_block(
        runtime=runtime,
        model=model,
        device=device,
        repo_root=repo_root,
        control_actor="human",
        games=human_games,
    )
    block_ai = run_block(
        runtime=runtime,
        model=model,
        device=device,
        repo_root=repo_root,
        control_actor="ai",
        games=ai_games,
    )

    all_results = list(block_human["results"]) + list(block_ai["results"])
    overall = summarize_block(all_results, runtime["catastrophic_loss_threshold"])
    elapsed = time.time() - started_at

    out = {
        "format_version": "ppo_duel_result_v1",
        "seed": runtime["seed"],
        "checkpoint_path": runtime["checkpoint_path"],
        "opponent_policy": runtime["opponent_policy"],
        "policy_mode": runtime["policy_mode"],
        "temperature": runtime["temperature"],
        "switch_seats": runtime["switch_seats"],
        "go_action_bonus": runtime["go_action_bonus"],
        "games": total_games,
        "workers": runtime["workers"],
        "elapsed_sec": elapsed,
        "summary": overall,
        "by_actor": {
            "human": block_human["summary"],
            "ai": block_ai["summary"],
        },
        "results": all_results,
    }

    with open(runtime["result_out"], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({"type": "duel_done", "result_out": runtime["result_out"], "summary": overall}, ensure_ascii=False))


if __name__ == "__main__":
    main()
