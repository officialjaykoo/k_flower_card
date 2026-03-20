from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

from .substrate import SubstrateRef, SubstrateSpec


@dataclass
class Node:
    ref: SubstrateRef
    spec: SubstrateSpec
    role: str
    seed: bool
    cppn: Any
    innovation: int
    depth: int
    enabled: bool
    state_key: str
    origin: str
    split_from_link_key: str | None = None


def new(
    *,
    ref: SubstrateRef,
    spec: SubstrateSpec,
    role: str,
    seed: bool,
    cppn: Any,
    innovation: int,
    depth: int,
    enabled: bool,
    state_key: str,
    origin: str,
    split_from_link_key: str | None = None,
) -> Node:
    return Node(
        ref=ref,
        spec=copy.deepcopy(spec),
        role=str(role),
        seed=bool(seed),
        cppn=cppn,
        innovation=int(innovation),
        depth=int(depth),
        enabled=bool(enabled),
        state_key=str(state_key),
        origin=str(origin),
        split_from_link_key=None if split_from_link_key is None else str(split_from_link_key),
    )


def crossover(self_node: Node, other_node: Node, cppn, *, fitness: float, other_fitness: float) -> Node:
    return Node(
        ref=self_node.ref,
        spec=copy.deepcopy(self_node.spec),
        role=str(self_node.role),
        seed=bool(self_node.seed),
        cppn=cppn,
        innovation=int(self_node.innovation),
        depth=int(self_node.depth if random.random() < 0.5 else other_node.depth),
        enabled=bool(self_node.enabled if random.random() < 0.75 else other_node.enabled),
        state_key=str(self_node.state_key),
        origin="crossover",
        split_from_link_key=self_node.split_from_link_key or other_node.split_from_link_key,
    )


def distance(self_node: Node, other_node: Node, cppn_distance: float) -> float:
    value = 0.0
    if bool(self_node.enabled) != bool(other_node.enabled):
        value += 1.0
    if bool(self_node.seed) != bool(other_node.seed):
        value += 0.5
    if bool(self_node.split_from_link_key) != bool(other_node.split_from_link_key):
        value += 0.25
    value += 0.8 * float(cppn_distance)
    value += 0.2 * math.tanh(abs(int(self_node.depth) - int(other_node.depth)))
    return float(value)


def export_node(gene: Node, export_cppn_component) -> dict[str, Any]:
    return {
        "ref": {"kind": str(gene.ref.kind), "index": int(gene.ref.index)},
        "role": str(gene.role),
        "seed": bool(gene.seed),
        "spec": {
            "depth": int(gene.spec.depth),
            "reverse_search": bool(gene.spec.reverse_search),
            "allow_identity_links": bool(gene.spec.allow_identity_links),
            "seed_point_count": int(len(gene.spec.seed_points)),
        },
        "innovation": int(gene.innovation),
        "depth": int(gene.depth),
        "enabled": bool(gene.enabled),
        "state_key": str(gene.state_key),
        "origin": str(gene.origin),
        "dynamic": not bool(gene.seed),
        "split_from_link_key": None if gene.split_from_link_key is None else str(gene.split_from_link_key),
        "cppn": export_cppn_component(gene.cppn),
    }
