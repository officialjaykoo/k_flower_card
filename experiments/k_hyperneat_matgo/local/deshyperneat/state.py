from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InitConfig:
    inputs: int = 0
    outputs: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, inputs: int = 0, outputs: int = 0, metadata: dict[str, Any] | None = None) -> "InitConfig":
        return cls(inputs=int(inputs), outputs=int(outputs), metadata=dict(metadata or {}))


@dataclass
class ComponentRegistry:
    node_states: dict[str, str] = field(default_factory=dict)
    link_states: dict[str, str] = field(default_factory=dict)
    redirects: dict[str, str] = field(default_factory=dict)


@dataclass
class State:
    engine: dict[str, object] = field(default_factory=dict)
    components: ComponentRegistry = field(default_factory=ComponentRegistry)

    def register_component_state(
        self,
        *,
        state_key: str,
        component_kind: str,
        redirect_key: str | None = None,
    ) -> str:
        key = str(state_key)
        kind = str(component_kind).strip().lower()
        if redirect_key is not None and str(redirect_key).strip():
            self.components.redirects[key] = str(redirect_key)
        root = self.resolve_component_state_key(key)
        if kind == "node":
            self.components.node_states[root] = root
        else:
            self.components.link_states[root] = root
        return root

    def resolve_component_state_key(self, state_key: str) -> str:
        current = str(state_key)
        seen: set[str] = set()
        redirects = self.components.redirects
        while current in redirects and current not in seen:
            seen.add(current)
            current = str(redirects[current])
        return current

    def export_component_snapshot(
        self,
        *,
        node_state_keys: list[str],
        link_state_keys: list[str],
    ) -> dict[str, dict[str, str]]:
        unique_node = {
            str(key): str(self.components.node_states[key])
            for key in sorted(set(node_state_keys))
            if key in self.components.node_states
        }
        resolved_link_roots = {self.resolve_component_state_key(key) for key in set(link_state_keys)}
        unique_link = {
            str(key): str(self.components.link_states[key])
            for key in sorted(resolved_link_roots)
            if key in self.components.link_states
        }
        redirects = {
            str(key): str(value)
            for key, value in sorted(self.components.redirects.items())
            if key in set(link_state_keys) or value in resolved_link_roots
        }
        return {
            "node_component_states": unique_node,
            "link_component_states": unique_link,
            "component_state_redirects": redirects,
        }

__all__ = [
    "ComponentRegistry",
    "InitConfig",
    "State",
]
