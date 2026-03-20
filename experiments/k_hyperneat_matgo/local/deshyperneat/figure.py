from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _edge_weight(edge: Any) -> float:
    if hasattr(edge, "edge"):
        return float(getattr(edge, "edge"))
    if isinstance(edge, dict):
        return float(edge.get("edge", 0.0))
    return float(edge or 0.0)


def save_fig_to_file(connections: Any, fname: str | Path, scale: float, size: float) -> None:
    get_nodes = getattr(connections, "get_all_nodes", None)
    get_connections = getattr(connections, "get_all_connections", None)
    all_nodes = list(get_nodes() if callable(get_nodes) else [])
    all_connections = list(get_connections() if callable(get_connections) else [])
    max_weight = max((abs(_edge_weight(connection)) for connection in all_connections), default=1.0)

    payload = {
        "scale": float(scale),
        "size": float(size),
        "max_weight": float(max_weight),
        "nodes": [repr(node) for node in all_nodes],
        "connections": [
            {
                "from": repr(getattr(connection, "from", None) if not isinstance(connection, dict) else connection.get("from")),
                "to": repr(getattr(connection, "to", None) if not isinstance(connection, dict) else connection.get("to")),
                "edge": float(_edge_weight(connection)),
            }
            for connection in all_connections
        ],
    }
    Path(fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
