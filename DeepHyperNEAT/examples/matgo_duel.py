from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deep_hyperneat.genome import Genome
from deep_hyperneat.decode import decode
from deep_hyperneat.matgo_runtime import (
    MATGO_INPUT_COUNT,
    MATGO_OUTPUT_COUNT,
    evaluate_substrate,
    load_runtime_settings,
    save_runtime_model,
    substrate_to_runtime_model,
)
from deep_hyperneat.phenomes import FeedForwardCPPN as CPPN


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "matgo_duel_smoke.json"
DUEL_CONFIG_FORMAT = "deep_hyperneat_matgo_duel_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
    )
    return parser.parse_args()


def _resolve_path(path_value: str, *, config_dir: Path) -> Path:
    raw = str(path_value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = (config_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def load_duel_config(path_value: str | Path) -> dict[str, object]:
    config_path = Path(path_value).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if str(payload.get("format_version") or "").strip() != DUEL_CONFIG_FORMAT:
        raise RuntimeError(f"invalid duel config format: expected {DUEL_CONFIG_FORMAT}")

    config_dir = config_path.parent
    runtime_path = _resolve_path(str(payload.get("runtime") or ""), config_dir=config_dir)
    out_path = _resolve_path(str(payload.get("out") or ""), config_dir=config_dir)
    return {
        "config_path": config_path,
        "runtime": runtime_path,
        "games": max(1, int(payload.get("games", 8) or 8)),
        "seed": str(payload.get("seed") or "deep_hyperneat_matgo_duel").strip() or "deep_hyperneat_matgo_duel",
        "sheet_width": max(1, int(payload.get("sheet_width", 26) or 26)),
        "out": out_path,
    }


def main() -> None:
    args = parse_args()
    duel_config = load_duel_config(args.config)
    runtime_settings = load_runtime_settings(duel_config["runtime"])

    genome = Genome(0)
    cppn = CPPN.create(genome)
    substrate = decode(cppn, [1, MATGO_INPUT_COUNT], MATGO_OUTPUT_COUNT, [1, int(duel_config["sheet_width"])])

    fitness, summary, meta = evaluate_substrate(
        substrate,
        REPO_ROOT,
        runtime_settings,
        games=int(duel_config["games"]),
        seed=str(duel_config["seed"]),
    )
    save_runtime_model(substrate_to_runtime_model(substrate), duel_config["out"])

    print(f"fitness={fitness:.6f}")
    print(f"win_rate={float(summary.get('win_rate_a', 0.0) or 0.0):.6f}")
    print(f"mean_gold_delta={float(summary.get('mean_gold_delta_a', 0.0) or 0.0):.6f}")
    print(f"config_path={Path(duel_config['config_path']).resolve()}")
    print(f"runtime_path={Path(duel_config['out']).resolve()}")
    _ = meta


if __name__ == "__main__":
    main()
