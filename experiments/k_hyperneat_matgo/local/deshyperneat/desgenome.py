from __future__ import annotations

from typing import Protocol

from .cppn import CppnModel
from .substrate import SubstrateRef, SubstrateTopology


class DesGenome(Protocol):
    def topology(self) -> SubstrateTopology:
        ...

    def get_node_cppn(self, substrate: SubstrateRef) -> CppnModel:
        ...

    def get_link_cppn(self, source: SubstrateRef, target: SubstrateRef) -> CppnModel:
        ...

    def get_depth(self, substrate: SubstrateRef) -> int:
        ...

    def get_link_outer_weight(self, source: SubstrateRef, target: SubstrateRef) -> float:
        ...

    def get_link_identity_mapping(self, source: SubstrateRef, target: SubstrateRef) -> bool:
        ...

    def is_substrate_enabled(self, substrate: SubstrateRef) -> bool:
        ...

    def is_link_enabled(self, source: SubstrateRef, target: SubstrateRef) -> bool:
        ...


__all__ = ["DesGenome"]
