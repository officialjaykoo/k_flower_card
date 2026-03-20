from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

from .substrate import DevelopmentEdge


def edge_key(source, target) -> str:
    return f"{source.kind}->{target.kind}"


@dataclass
class Link:
    edge: DevelopmentEdge
    seed: bool
    cppn: Any
    innovation: int
    depth: int
    enabled: bool
    outer_weight: float
    identity_mapping_enabled: bool
    state_key: str
    state_redirect_key: str | None
    origin: str
    cloned_from_key: str | None
    split_hidden_kind: str | None = None


def new(
    *,
    edge: DevelopmentEdge,
    seed: bool,
    cppn: Any,
    innovation: int,
    depth: int,
    enabled: bool,
    outer_weight: float,
    identity_mapping_enabled: bool,
    state_key: str,
    state_redirect_key: str | None,
    origin: str,
    cloned_from_key: str | None,
    split_hidden_kind: str | None = None,
) -> Link:
    return Link(
        edge=copy.deepcopy(edge),
        seed=bool(seed),
        cppn=cppn,
        innovation=int(innovation),
        depth=int(depth),
        enabled=bool(enabled),
        outer_weight=float(outer_weight),
        identity_mapping_enabled=bool(identity_mapping_enabled),
        state_key=str(state_key),
        state_redirect_key=None if state_redirect_key is None else str(state_redirect_key),
        origin=str(origin),
        cloned_from_key=None if cloned_from_key is None else str(cloned_from_key),
        split_hidden_kind=None if split_hidden_kind is None else str(split_hidden_kind),
    )


def identity(parent_link: Link, *, state_key: str, redirect_key: str | None) -> Link:
    return Link(
        edge=copy.deepcopy(parent_link.edge),
        seed=bool(parent_link.seed),
        cppn=copy.deepcopy(parent_link.cppn),
        innovation=int(parent_link.innovation),
        depth=int(parent_link.depth),
        enabled=True,
        outer_weight=float(parent_link.outer_weight),
        identity_mapping_enabled=bool(parent_link.identity_mapping_enabled),
        state_key=str(state_key),
        state_redirect_key=None if redirect_key is None else str(redirect_key),
        origin="identity",
        cloned_from_key=None if redirect_key is None else str(redirect_key),
        split_hidden_kind=parent_link.split_hidden_kind,
    )


def clone_with(parent_link: Link, *, edge, state_key: str, redirect_key: str | None, origin: str) -> Link:
    return Link(
        edge=copy.deepcopy(edge),
        seed=bool(parent_link.seed),
        cppn=copy.deepcopy(parent_link.cppn),
        innovation=int(parent_link.innovation),
        depth=int(parent_link.depth),
        enabled=True,
        outer_weight=float(parent_link.outer_weight),
        identity_mapping_enabled=bool(parent_link.identity_mapping_enabled),
        state_key=str(state_key),
        state_redirect_key=None if redirect_key is None else str(redirect_key),
        origin=str(origin),
        cloned_from_key=None if redirect_key is None else str(redirect_key),
        split_hidden_kind=parent_link.split_hidden_kind,
    )


def crossover(self_link: Link, other_link: Link, cppn, *, fitness: float, other_fitness: float) -> Link:
    return Link(
        edge=copy.deepcopy(self_link.edge),
        seed=bool(self_link.seed),
        cppn=cppn,
        innovation=int(self_link.innovation),
        depth=int(self_link.depth if random.random() < 0.5 else other_link.depth),
        enabled=bool(self_link.enabled if random.random() < 0.75 else other_link.enabled),
        outer_weight=float(self_link.outer_weight if random.random() < 0.75 else other_link.outer_weight),
        identity_mapping_enabled=bool(self_link.identity_mapping_enabled or other_link.identity_mapping_enabled),
        state_key=str(self_link.state_key),
        state_redirect_key=self_link.state_redirect_key or other_link.state_redirect_key,
        origin="crossover",
        cloned_from_key=self_link.cloned_from_key or other_link.cloned_from_key,
        split_hidden_kind=self_link.split_hidden_kind or other_link.split_hidden_kind,
    )


def distance(self_link: Link, other_link: Link, cppn_distance: float) -> float:
    value = abs(float(self_link.outer_weight) - float(other_link.outer_weight))
    if bool(self_link.enabled) != bool(other_link.enabled):
        value += 1.0
    if bool(self_link.identity_mapping_enabled) != bool(other_link.identity_mapping_enabled):
        value += 0.5
    if bool(self_link.seed) != bool(other_link.seed):
        value += 0.5
    value = (0.5 * value) + (0.4 * float(cppn_distance)) + (0.1 * math.tanh(abs(int(self_link.depth) - int(other_link.depth))))
    return float(value)


def export_link(gene: Link, export_cppn_component) -> dict[str, Any]:
    return {
        "source": {"kind": str(gene.edge.source.kind), "index": int(gene.edge.source.index)},
        "target": {"kind": str(gene.edge.target.kind), "index": int(gene.edge.target.index)},
        "seed": bool(gene.seed),
        "innovation": int(gene.innovation),
        "depth": int(gene.depth),
        "enabled": bool(gene.enabled),
        "outer_weight": float(gene.outer_weight),
        "identity_mapping_enabled": bool(gene.identity_mapping_enabled),
        "state_key": str(gene.state_key),
        "state_redirect_key": None if gene.state_redirect_key is None else str(gene.state_redirect_key),
        "origin": str(gene.origin),
        "cloned_from_key": None if gene.cloned_from_key is None else str(gene.cloned_from_key),
        "dynamic": not bool(gene.seed),
        "split_hidden_kind": None if gene.split_hidden_kind is None else str(gene.split_hidden_kind),
        "cppn": export_cppn_component(gene.cppn),
    }
