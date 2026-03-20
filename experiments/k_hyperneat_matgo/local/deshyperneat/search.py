from __future__ import annotations

"""ES-HyperNEAT-style local substrate search helpers.

This module explores connection regions with a quadtree-like subdivision pass,
filters low-signal regions, and returns discovered points plus weighted
connections for developer-side phenotype growth.
"""

from dataclasses import dataclass, field
from statistics import median

from .config import SearchConfig
from .coordinates import Point2D
from .cppn import CppnModel, query_cppn_link


@dataclass(frozen=True)
class SearchConnection:
    source: Point2D
    target: Point2D
    weight: float


@dataclass
class SearchResult:
    discovered_points: list[Point2D]
    connections: list[SearchConnection]


@dataclass
class QuadPoint:
    point: Point2D
    width: float
    depth: int
    weight: float
    variance: float = 0.0
    children: list["QuadPoint"] = field(default_factory=list)

    def collect_leaf_weights(self, *, include_root: bool, include_internal: bool) -> list[float]:
        weights: list[float] = []
        if (include_root and not include_internal) or not self.children:
            weights.append(self.weight)
        for child in self.children:
            weights.extend(child.collect_leaf_weights(include_root=include_internal, include_internal=include_internal))
        return weights


class Search:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self._find_connections_cache: dict[tuple[int, float, float, bool], tuple[SearchConnection, ...]] = {}

    def _query_raw(
        self,
        cppn: CppnModel,
        source: Point2D,
        target: Point2D,
        *,
        reverse: bool,
    ) -> float:
        if reverse:
            values = target.as_cppn_input(source)
        else:
            values = source.as_cppn_input(target)
        output = cppn.activate(values)
        return float(output[0]) if output else 0.0

    def _create_children(
        self,
        node: QuadPoint,
        cppn: CppnModel,
        source: Point2D,
        *,
        reverse: bool,
    ) -> tuple[float, float]:
        width = node.width / 2.0
        child_centers = [
            Point2D(node.point.x - width, node.point.y - width),
            Point2D(node.point.x - width, node.point.y + width),
            Point2D(node.point.x + width, node.point.y + width),
            Point2D(node.point.x + width, node.point.y - width),
        ]
        children: list[QuadPoint] = []
        for center in child_centers:
            children.append(
                QuadPoint(
                    point=center,
                    width=width,
                    depth=node.depth + 1,
                    weight=self._query_raw(cppn, source, center, reverse=reverse),
                )
            )
        node.children = children
        weights = [child.weight for child in children]
        return min(weights), max(weights)

    def _calc_variance(self, node: QuadPoint, *, delta_weight: float, include_root: bool, branch_mode: bool) -> float:
        if delta_weight == 0.0:
            node.variance = 0.0
            return 0.0
        weights = node.collect_leaf_weights(include_root=include_root, include_internal=branch_mode)
        if not weights:
            node.variance = 0.0
            return 0.0
        centroid = median(weights) if self.config.median_variance else (sum(weights) / float(len(weights)))
        divisor = delta_weight if self.config.relative_variance else 1.0
        if divisor == 0.0:
            divisor = 1.0
        squares = [((centroid - value) / divisor) ** 2 for value in weights]
        node.variance = max(squares) if self.config.max_variance else (sum(squares) / float(len(squares)))
        return node.variance

    def _expand_children(
        self,
        node: QuadPoint,
        *,
        delta_weight: float,
    ) -> list[QuadPoint]:
        should_expand = (
            node.depth + 1 < self.config.initial_resolution
            or (
                node.depth + 1 < self.config.max_resolution
                and self._calc_variance(
                    node,
                    delta_weight=delta_weight,
                    include_root=True,
                    branch_mode=True,
                ) > self.config.division_threshold
            )
        )
        return list(node.children) if should_expand else []

    def _extract_connections(
        self,
        node: QuadPoint,
        cppn: CppnModel,
        source: Point2D,
        *,
        reverse: bool,
        delta_weight: float,
        connections: list[SearchConnection],
    ) -> list[QuadPoint]:
        leaves: list[QuadPoint] = []
        width = node.width
        for child in node.children:
            variance = self._calc_variance(
                child,
                delta_weight=delta_weight,
                include_root=False,
                branch_mode=True,
            )
            if variance <= self.config.variance_threshold:
                if self.config.band_threshold > 0.0:
                    d_left = abs(child.weight - self._query_raw(cppn, source, Point2D(child.point.x - width, child.point.y), reverse=reverse))
                    d_right = abs(child.weight - self._query_raw(cppn, source, Point2D(child.point.x + width, child.point.y), reverse=reverse))
                    d_up = abs(child.weight - self._query_raw(cppn, source, Point2D(child.point.x, child.point.y - width), reverse=reverse))
                    d_down = abs(child.weight - self._query_raw(cppn, source, Point2D(child.point.x, child.point.y + width), reverse=reverse))
                    band_value = max(min(d_up, d_down), min(d_left, d_right))
                else:
                    band_value = 0.0
                if band_value >= self.config.band_threshold:
                    final_weight = query_cppn_link(
                        cppn,
                        source if not reverse else child.point,
                        child.point if not reverse else source,
                        max_weight=self.config.max_weight,
                        threshold=self.config.weight_threshold,
                        leo_enabled=self.config.leo_enabled,
                        leo_threshold=self.config.leo_threshold,
                    )
                    if final_weight != 0.0:
                        if reverse:
                            connections.append(
                                SearchConnection(
                                    source=child.point,
                                    target=source,
                                    weight=final_weight,
                                )
                            )
                        else:
                            connections.append(
                                SearchConnection(
                                    source=source,
                                    target=child.point,
                                    weight=final_weight,
                                )
                            )
            if child.variance > self.config.variance_threshold:
                leaves.append(child)
        return leaves

    def find_connections(
        self,
        source: Point2D,
        cppn: CppnModel,
        *,
        reverse: bool = False,
    ) -> list[SearchConnection]:
        cache_key = (id(cppn), float(source.x), float(source.y), bool(reverse))
        cached = self._find_connections_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        root = QuadPoint(
            point=Point2D(0.0, 0.0),
            width=1.0,
            depth=1,
            weight=self._query_raw(cppn, source, Point2D(0.0, 0.0), reverse=reverse),
        )
        min_weight = root.weight
        max_weight = root.weight

        leaves = [root]
        while leaves:
            next_leaves: list[QuadPoint] = []
            for leaf in leaves:
                child_min, child_max = self._create_children(leaf, cppn, source, reverse=reverse)
                min_weight = min(min_weight, child_min)
                max_weight = max(max_weight, child_max)
            delta_weight = max_weight - min_weight
            for leaf in leaves:
                next_leaves.extend(self._expand_children(leaf, delta_weight=delta_weight))
            leaves = next_leaves

        if min_weight == max_weight:
            return []

        connections: list[SearchConnection] = []
        leaves = [root]
        delta_weight = max_weight - min_weight
        while leaves and (
            self.config.max_discoveries <= 0
            or len(connections) < self.config.max_discoveries
        ):
            next_leaves: list[QuadPoint] = []
            for leaf in leaves:
                next_leaves.extend(
                    self._extract_connections(
                        leaf,
                        cppn,
                        source,
                        reverse=reverse,
                        delta_weight=delta_weight,
                        connections=connections,
                    )
                )
            leaves = next_leaves

        for leaf in leaves:
            final_weight = query_cppn_link(
                cppn,
                source if not reverse else leaf.point,
                leaf.point if not reverse else source,
                max_weight=self.config.max_weight,
                threshold=self.config.weight_threshold,
                leo_enabled=self.config.leo_enabled,
                leo_threshold=self.config.leo_threshold,
            )
            if final_weight == 0.0:
                continue
            if reverse:
                connections.append(
                    SearchConnection(source=leaf.point, target=source, weight=final_weight)
                )
            else:
                connections.append(
                    SearchConnection(source=source, target=leaf.point, weight=final_weight)
                )

        if self.config.max_outgoing > 0 and len(connections) > self.config.max_outgoing:
            connections.sort(key=lambda item: abs(item.weight), reverse=True)
            connections = connections[: self.config.max_outgoing]

        unique: dict[tuple[float, float, float, float], SearchConnection] = {}
        for connection in connections:
            key = (
                connection.source.x,
                connection.source.y,
                connection.target.x,
                connection.target.y,
            )
            unique[key] = connection
        result = tuple(unique.values())
        self._find_connections_cache[cache_key] = result
        return list(result)

    def explore_substrate(
        self,
        *,
        seed_points: list[Point2D],
        cppn: CppnModel,
        output_points: list[Point2D] | None = None,
        blocked_points: list[Point2D] | None = None,
        depth: int,
        reverse: bool,
        allow_target_links: bool,
    ) -> SearchResult:
        outputs = set(output_points or [])
        visited = set(blocked_points or [])
        if not allow_target_links:
            visited.update(seed_points)

        layers: list[list[Point2D]] = [list(seed_points)]
        connections: list[SearchConnection] = []
        discovered_points: set[Point2D] = set()
        iteration_limit = max(0, int(depth))
        if iteration_limit <= 0:
            iteration_limit = max(0, int(self.config.iteration_level or 0))

        for layer_index in range(iteration_limit):
            if layer_index >= len(layers):
                break
            current_layer = layers[layer_index]
            if not current_layer:
                break

            next_points: list[Point2D] = []
            for source in current_layer:
                for connection in self.find_connections(source, cppn, reverse=reverse):
                    discovered_point = connection.source if reverse else connection.target
                    if discovered_point in visited:
                        continue
                    connections.append(connection)
                    discovered_points.add(discovered_point)
                    if discovered_point not in outputs:
                        next_points.append(discovered_point)

            unique_next: list[Point2D] = []
            seen_next: set[Point2D] = set()
            for point in next_points:
                if point in seen_next:
                    continue
                seen_next.add(point)
                unique_next.append(point)
            if not unique_next:
                break
            visited.update(unique_next)
            layers.append(unique_next)

        return SearchResult(
            discovered_points=sorted(discovered_points, key=lambda point: (point.y, point.x)),
            connections=connections,
        )


__all__ = ["QuadPoint", "Search", "SearchConnection", "SearchResult"]
