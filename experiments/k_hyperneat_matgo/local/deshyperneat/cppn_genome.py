from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _parse_options(value: Any, default: list[str]) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [token for token in text.split() if token]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(float(value), float(max_value)), float(min_value))


def _sample_float(*, mean: float, stdev: float, init_type: str, min_value: float, max_value: float) -> float:
    kind = str(init_type or "gaussian").strip().lower()
    if "uniform" in kind:
        low = max(min_value, mean - (2.0 * stdev))
        high = min(max_value, mean + (2.0 * stdev))
        if high < low:
            high = low
        return random.uniform(low, high)
    return _clamp(random.gauss(mean, stdev), min_value, max_value)


def _mutate_float(
    value: float,
    *,
    mutate_rate: float,
    mutate_power: float,
    replace_rate: float,
    init_mean: float,
    init_stdev: float,
    init_type: str,
    min_value: float,
    max_value: float,
) -> float:
    roll = random.random()
    if roll < float(mutate_rate):
        return _clamp(float(value) + random.gauss(0.0, float(mutate_power)), min_value, max_value)
    if roll < float(mutate_rate) + float(replace_rate):
        return _sample_float(
            mean=float(init_mean),
            stdev=float(init_stdev),
            init_type=str(init_type),
            min_value=float(min_value),
            max_value=float(max_value),
        )
    return float(value)


def _mutate_choice(value: str, *, mutate_rate: float, options: list[str], default: str) -> str:
    if not options:
        return str(value or default)
    if random.random() < float(mutate_rate):
        return str(random.choice(options))
    return str(value or default)


def _mutate_bool(
    value: bool,
    *,
    mutate_rate: float,
    rate_to_true_add: float,
    rate_to_false_add: float,
) -> bool:
    threshold = float(mutate_rate)
    if value:
        threshold += float(rate_to_false_add)
    else:
        threshold += float(rate_to_true_add)
    if random.random() < threshold:
        return bool(random.random() < 0.5)
    return bool(value)


def _activation_fn(name: str):
    key = str(name or "identity").strip().lower()
    if key == "tanh":
        return math.tanh
    if key == "sin":
        return math.sin
    if key == "gauss":
        return lambda x: math.exp(-(float(x) * float(x)))
    if key == "abs":
        return lambda x: abs(float(x))
    return lambda x: float(x)


def _aggregation_fn(name: str):
    key = str(name or "sum").strip().lower()
    if key == "sum":
        return lambda values: float(sum(values))
    return lambda values: float(sum(values))


@dataclass
class StandaloneCppnGenomeConfig:
    num_inputs: int
    num_outputs: int
    num_hidden: int
    feed_forward: bool
    initial_connection: str
    conn_add_prob: float
    conn_delete_prob: float
    node_add_prob: float
    node_delete_prob: float
    activation_default: str
    activation_mutate_rate: float
    activation_options: list[str]
    aggregation_default: str
    aggregation_mutate_rate: float
    aggregation_options: list[str]
    bias_init_mean: float
    bias_init_stdev: float
    bias_init_type: str
    bias_max_value: float
    bias_min_value: float
    bias_mutate_power: float
    bias_mutate_rate: float
    bias_replace_rate: float
    response_init_mean: float
    response_init_stdev: float
    response_init_type: str
    response_max_value: float
    response_min_value: float
    response_mutate_power: float
    response_mutate_rate: float
    response_replace_rate: float
    enabled_default: str
    enabled_mutate_rate: float
    enabled_rate_to_true_add: float
    enabled_rate_to_false_add: float
    compatibility_disjoint_coefficient: float
    compatibility_weight_coefficient: float
    weight_init_mean: float
    weight_init_stdev: float
    weight_init_type: str
    weight_max_value: float
    weight_min_value: float
    weight_mutate_power: float
    weight_mutate_rate: float
    weight_replace_rate: float
    input_keys: list[int] = field(default_factory=list)
    output_keys: list[int] = field(default_factory=list)
    connection_innovations: dict[tuple[int, int], int] = field(default_factory=dict)
    next_connection_innovation: int = 0

    def __post_init__(self) -> None:
        if not self.input_keys:
            self.input_keys = [-(index + 1) for index in range(int(self.num_inputs))]
        if not self.output_keys:
            self.output_keys = [index for index in range(int(self.num_outputs))]

    def get_connection_innovation(self, key: tuple[int, int]) -> int:
        existing = self.connection_innovations.get(key)
        if existing is not None:
            return int(existing)
        innovation = int(self.next_connection_innovation)
        self.connection_innovations[key] = innovation
        self.next_connection_innovation = innovation + 1
        return innovation

    def save(self, handle) -> None:
        lines = [
            f"num_inputs = {self.num_inputs}",
            f"num_outputs = {self.num_outputs}",
            f"num_hidden = {self.num_hidden}",
            f"feed_forward = {self.feed_forward}",
            f"initial_connection = {self.initial_connection}",
            f"conn_add_prob = {self.conn_add_prob}",
            f"conn_delete_prob = {self.conn_delete_prob}",
            f"node_add_prob = {self.node_add_prob}",
            f"node_delete_prob = {self.node_delete_prob}",
            f"activation_default = {self.activation_default}",
            f"activation_mutate_rate = {self.activation_mutate_rate}",
            f"activation_options = {' '.join(self.activation_options)}",
            f"aggregation_default = {self.aggregation_default}",
            f"aggregation_mutate_rate = {self.aggregation_mutate_rate}",
            f"aggregation_options = {' '.join(self.aggregation_options)}",
            f"compatibility_disjoint_coefficient = {self.compatibility_disjoint_coefficient}",
            f"compatibility_weight_coefficient = {self.compatibility_weight_coefficient}",
        ]
        handle.write("\n".join(lines) + "\n")


@dataclass
class StandaloneCppnNodeGene:
    key: int
    bias: float
    response: float
    activation: str
    aggregation: str

    @classmethod
    def create(cls, key: int, config: StandaloneCppnGenomeConfig) -> "StandaloneCppnNodeGene":
        return cls(
            key=int(key),
            bias=_sample_float(
                mean=config.bias_init_mean,
                stdev=config.bias_init_stdev,
                init_type=config.bias_init_type,
                min_value=config.bias_min_value,
                max_value=config.bias_max_value,
            ),
            response=_sample_float(
                mean=config.response_init_mean,
                stdev=config.response_init_stdev,
                init_type=config.response_init_type,
                min_value=config.response_min_value,
                max_value=config.response_max_value,
            ),
            activation=str(config.activation_default),
            aggregation=str(config.aggregation_default),
        )

    def copy(self) -> "StandaloneCppnNodeGene":
        return StandaloneCppnNodeGene(
            key=int(self.key),
            bias=float(self.bias),
            response=float(self.response),
            activation=str(self.activation),
            aggregation=str(self.aggregation),
        )

    def crossover(
        self,
        other: "StandaloneCppnNodeGene",
        config: StandaloneCppnGenomeConfig,
    ) -> "StandaloneCppnNodeGene":
        return StandaloneCppnNodeGene(
            key=int(self.key),
            bias=float(self.bias if random.random() < 0.5 else other.bias),
            response=float(self.response if random.random() < 0.5 else other.response),
            activation=str(self.activation if random.random() < 0.5 else other.activation),
            aggregation=str(self.aggregation if random.random() < 0.5 else other.aggregation),
        )

    def mutate(self, config: StandaloneCppnGenomeConfig) -> None:
        self.bias = _mutate_float(
            self.bias,
            mutate_rate=config.bias_mutate_rate,
            mutate_power=config.bias_mutate_power,
            replace_rate=config.bias_replace_rate,
            init_mean=config.bias_init_mean,
            init_stdev=config.bias_init_stdev,
            init_type=config.bias_init_type,
            min_value=config.bias_min_value,
            max_value=config.bias_max_value,
        )
        self.response = _mutate_float(
            self.response,
            mutate_rate=config.response_mutate_rate,
            mutate_power=config.response_mutate_power,
            replace_rate=config.response_replace_rate,
            init_mean=config.response_init_mean,
            init_stdev=config.response_init_stdev,
            init_type=config.response_init_type,
            min_value=config.response_min_value,
            max_value=config.response_max_value,
        )
        self.activation = _mutate_choice(
            self.activation,
            mutate_rate=config.activation_mutate_rate,
            options=config.activation_options,
            default=config.activation_default,
        )
        self.aggregation = _mutate_choice(
            self.aggregation,
            mutate_rate=config.aggregation_mutate_rate,
            options=config.aggregation_options,
            default=config.aggregation_default,
        )

    def distance(self, other: "StandaloneCppnNodeGene", config: StandaloneCppnGenomeConfig) -> float:
        delta = abs(float(self.bias) - float(other.bias)) + abs(float(self.response) - float(other.response))
        if str(self.activation) != str(other.activation):
            delta += 1.0
        if str(self.aggregation) != str(other.aggregation):
            delta += 1.0
        return delta * float(config.compatibility_weight_coefficient)


@dataclass
class StandaloneCppnConnectionGene:
    key: tuple[int, int]
    innovation: int
    weight: float
    enabled: bool

    @classmethod
    def create(
        cls,
        key: tuple[int, int],
        config: StandaloneCppnGenomeConfig,
        *,
        weight: float | None = None,
        enabled: bool | None = None,
    ) -> "StandaloneCppnConnectionGene":
        return cls(
            key=(int(key[0]), int(key[1])),
            innovation=int(config.get_connection_innovation((int(key[0]), int(key[1])))),
            weight=float(
                _sample_float(
                    mean=config.weight_init_mean,
                    stdev=config.weight_init_stdev,
                    init_type=config.weight_init_type,
                    min_value=config.weight_min_value,
                    max_value=config.weight_max_value,
                )
                if weight is None
                else weight
            ),
            enabled=bool(
                _parse_bool(config.enabled_default, True) if enabled is None else enabled
            ),
        )

    def copy(self) -> "StandaloneCppnConnectionGene":
        return StandaloneCppnConnectionGene(
            key=(int(self.key[0]), int(self.key[1])),
            innovation=int(self.innovation),
            weight=float(self.weight),
            enabled=bool(self.enabled),
        )

    def crossover(
        self,
        other: "StandaloneCppnConnectionGene",
        config: StandaloneCppnGenomeConfig,
    ) -> "StandaloneCppnConnectionGene":
        enabled = bool(self.enabled if random.random() < 0.5 else other.enabled)
        if (not self.enabled) or (not other.enabled):
            if random.random() < 0.75:
                enabled = False
        return StandaloneCppnConnectionGene(
            key=(int(self.key[0]), int(self.key[1])),
            innovation=int(self.innovation),
            weight=float(self.weight if random.random() < 0.5 else other.weight),
            enabled=enabled,
        )

    def mutate(self, config: StandaloneCppnGenomeConfig) -> None:
        self.weight = _mutate_float(
            self.weight,
            mutate_rate=config.weight_mutate_rate,
            mutate_power=config.weight_mutate_power,
            replace_rate=config.weight_replace_rate,
            init_mean=config.weight_init_mean,
            init_stdev=config.weight_init_stdev,
            init_type=config.weight_init_type,
            min_value=config.weight_min_value,
            max_value=config.weight_max_value,
        )
        self.enabled = _mutate_bool(
            self.enabled,
            mutate_rate=config.enabled_mutate_rate,
            rate_to_true_add=config.enabled_rate_to_true_add,
            rate_to_false_add=config.enabled_rate_to_false_add,
        )

    def distance(self, other: "StandaloneCppnConnectionGene", config: StandaloneCppnGenomeConfig) -> float:
        delta = abs(float(self.weight) - float(other.weight))
        if bool(self.enabled) != bool(other.enabled):
            delta += 1.0
        return delta * float(config.compatibility_weight_coefficient)


class StandaloneCppnGenome:
    @classmethod
    def parse_config(cls, param_dict: dict[str, Any]) -> StandaloneCppnGenomeConfig:
        return StandaloneCppnGenomeConfig(
            num_inputs=int(param_dict.get("num_inputs", 0) or 0),
            num_outputs=int(param_dict.get("num_outputs", 0) or 0),
            num_hidden=int(param_dict.get("num_hidden", 0) or 0),
            feed_forward=_parse_bool(param_dict.get("feed_forward", True), True),
            initial_connection=str(param_dict.get("initial_connection", "full_direct") or "full_direct"),
            conn_add_prob=float(param_dict.get("conn_add_prob", 0.20) or 0.20),
            conn_delete_prob=float(param_dict.get("conn_delete_prob", 0.03) or 0.03),
            node_add_prob=float(param_dict.get("node_add_prob", 0.08) or 0.08),
            node_delete_prob=float(param_dict.get("node_delete_prob", 0.02) or 0.02),
            activation_default=str(param_dict.get("activation_default", "tanh") or "tanh"),
            activation_mutate_rate=float(param_dict.get("activation_mutate_rate", 0.20) or 0.20),
            activation_options=_parse_options(param_dict.get("activation_options"), ["tanh", "sin", "gauss", "identity", "abs"]),
            aggregation_default=str(param_dict.get("aggregation_default", "sum") or "sum"),
            aggregation_mutate_rate=float(param_dict.get("aggregation_mutate_rate", 0.0) or 0.0),
            aggregation_options=_parse_options(param_dict.get("aggregation_options"), ["sum"]),
            bias_init_mean=float(param_dict.get("bias_init_mean", 0.0) or 0.0),
            bias_init_stdev=float(param_dict.get("bias_init_stdev", 0.3) or 0.3),
            bias_init_type=str(param_dict.get("bias_init_type", "gaussian") or "gaussian"),
            bias_max_value=float(param_dict.get("bias_max_value", 5.0) or 5.0),
            bias_min_value=float(param_dict.get("bias_min_value", -5.0) or -5.0),
            bias_mutate_power=float(param_dict.get("bias_mutate_power", 0.2) or 0.2),
            bias_mutate_rate=float(param_dict.get("bias_mutate_rate", 0.5) or 0.5),
            bias_replace_rate=float(param_dict.get("bias_replace_rate", 0.1) or 0.1),
            response_init_mean=float(param_dict.get("response_init_mean", 1.0) or 1.0),
            response_init_stdev=float(param_dict.get("response_init_stdev", 0.0) or 0.0),
            response_init_type=str(param_dict.get("response_init_type", "gaussian") or "gaussian"),
            response_max_value=float(param_dict.get("response_max_value", 5.0) or 5.0),
            response_min_value=float(param_dict.get("response_min_value", -5.0) or -5.0),
            response_mutate_power=float(param_dict.get("response_mutate_power", 0.0) or 0.0),
            response_mutate_rate=float(param_dict.get("response_mutate_rate", 0.0) or 0.0),
            response_replace_rate=float(param_dict.get("response_replace_rate", 0.0) or 0.0),
            enabled_default=str(param_dict.get("enabled_default", "true") or "true"),
            enabled_mutate_rate=float(param_dict.get("enabled_mutate_rate", 0.015) or 0.015),
            enabled_rate_to_true_add=float(param_dict.get("enabled_rate_to_true_add", 0.0) or 0.0),
            enabled_rate_to_false_add=float(param_dict.get("enabled_rate_to_false_add", 0.0) or 0.0),
            compatibility_disjoint_coefficient=float(
                param_dict.get("compatibility_disjoint_coefficient", 1.0) or 1.0
            ),
            compatibility_weight_coefficient=float(
                param_dict.get("compatibility_weight_coefficient", 0.4) or 0.4
            ),
            weight_init_mean=float(param_dict.get("weight_init_mean", 0.0) or 0.0),
            weight_init_stdev=float(param_dict.get("weight_init_stdev", 0.5) or 0.5),
            weight_init_type=str(param_dict.get("weight_init_type", "gaussian") or "gaussian"),
            weight_max_value=float(param_dict.get("weight_max_value", 5.0) or 5.0),
            weight_min_value=float(param_dict.get("weight_min_value", -5.0) or -5.0),
            weight_mutate_power=float(param_dict.get("weight_mutate_power", 0.24) or 0.24),
            weight_mutate_rate=float(param_dict.get("weight_mutate_rate", 0.7) or 0.7),
            weight_replace_rate=float(param_dict.get("weight_replace_rate", 0.1) or 0.1),
        )

    def __init__(self, key: int):
        self.key = int(key)
        self.fitness: float | None = None
        self.nodes: dict[int, StandaloneCppnNodeGene] = {}
        self.connections: dict[tuple[int, int], StandaloneCppnConnectionGene] = {}
        self.node_order: dict[int, float] = {}
        self.input_keys: list[int] = []
        self.output_keys: list[int] = []
        self.next_node_key: int = 0

    def configure_new(self, config: StandaloneCppnGenomeConfig) -> None:
        self.input_keys = list(config.input_keys)
        self.output_keys = list(config.output_keys)
        self.nodes = {}
        self.connections = {}
        self.node_order = {int(key): 0.0 for key in self.input_keys}
        for key in self.output_keys:
            self.nodes[int(key)] = StandaloneCppnNodeGene.create(int(key), config)
            self.node_order[int(key)] = 1.0
        self.next_node_key = max(self.output_keys or [0]) + 1
        for _ in range(int(config.num_hidden)):
            hidden_key = self._new_node_key()
            self.nodes[hidden_key] = StandaloneCppnNodeGene.create(hidden_key, config)
            self.node_order[hidden_key] = 0.5
        if str(config.initial_connection).strip().lower() == "full_direct":
            for source_key in self.input_keys:
                for target_key in self.output_keys:
                    self.connections[(int(source_key), int(target_key))] = StandaloneCppnConnectionGene.create(
                        (int(source_key), int(target_key)),
                        config,
                    )
        elif str(config.initial_connection).strip().lower() == "partial_direct":
            for source_key in self.input_keys:
                for target_key in self.output_keys:
                    if random.random() < 0.5:
                        self.connections[(int(source_key), int(target_key))] = StandaloneCppnConnectionGene.create(
                            (int(source_key), int(target_key)),
                            config,
                        )

    def configure_crossover(
        self,
        genome1: "StandaloneCppnGenome",
        genome2: "StandaloneCppnGenome",
        config: StandaloneCppnGenomeConfig,
    ) -> None:
        dominant, recessive = self._select_parents(genome1, genome2)
        self.input_keys = list(config.input_keys)
        self.output_keys = list(config.output_keys)
        self.nodes = {}
        self.connections = {}
        self.node_order = {int(key): 0.0 for key in self.input_keys}

        node_keys = sorted(set(dominant.nodes) | set(recessive.nodes))
        for key in node_keys:
            left = dominant.nodes.get(key)
            right = recessive.nodes.get(key)
            if left is not None and right is not None:
                gene = left.crossover(right, config)
                order = dominant.node_order.get(key, recessive.node_order.get(key, 0.5))
            elif left is not None:
                gene = left.copy()
                order = dominant.node_order.get(key, 0.5)
            else:
                continue
            self.nodes[int(key)] = gene
            self.node_order[int(key)] = float(order)

        dominant_conn_by_innovation = {gene.innovation: gene for gene in dominant.connections.values()}
        recessive_conn_by_innovation = {gene.innovation: gene for gene in recessive.connections.values()}
        for innovation in sorted(set(dominant_conn_by_innovation) | set(recessive_conn_by_innovation)):
            left = dominant_conn_by_innovation.get(innovation)
            right = recessive_conn_by_innovation.get(innovation)
            if left is not None and right is not None:
                gene = left.crossover(right, config)
            elif left is not None:
                gene = left.copy()
            else:
                continue
            self.connections[tuple(gene.key)] = gene

        self.next_node_key = max([*self.output_keys, *self.nodes.keys(), 0]) + 1
        self._repair(config)

    def mutate(self, config: StandaloneCppnGenomeConfig) -> None:
        for gene in self.nodes.values():
            gene.mutate(config)
        for gene in self.connections.values():
            gene.mutate(config)
        if random.random() < float(config.node_add_prob):
            self._mutate_add_node(config)
        if random.random() < float(config.node_delete_prob):
            self._mutate_delete_node()
        if random.random() < float(config.conn_add_prob):
            self._mutate_add_connection(config)
        if random.random() < float(config.conn_delete_prob):
            self._mutate_delete_connection()
        self._repair(config)

    def distance(self, other: "StandaloneCppnGenome", config: StandaloneCppnGenomeConfig) -> float:
        node_keys = set(self.nodes) | set(other.nodes)
        conn_innovations = {gene.innovation for gene in self.connections.values()} | {
            gene.innovation for gene in other.connections.values()
        }

        node_delta = 0.0
        for key in node_keys:
            left = self.nodes.get(key)
            right = other.nodes.get(key)
            if left is None or right is None:
                node_delta += float(config.compatibility_disjoint_coefficient)
            else:
                node_delta += left.distance(right, config)
                node_delta += abs(float(self.node_order.get(key, 0.5)) - float(other.node_order.get(key, 0.5)))

        left_conn = {gene.innovation: gene for gene in self.connections.values()}
        right_conn = {gene.innovation: gene for gene in other.connections.values()}
        conn_delta = 0.0
        for innovation in conn_innovations:
            left = left_conn.get(innovation)
            right = right_conn.get(innovation)
            if left is None or right is None:
                conn_delta += float(config.compatibility_disjoint_coefficient)
            else:
                conn_delta += left.distance(right, config)

        node_norm = node_delta / max(1, len(node_keys))
        conn_norm = conn_delta / max(1, len(conn_innovations))
        return float(node_norm + conn_norm)

    def _select_parents(
        self,
        genome1: "StandaloneCppnGenome",
        genome2: "StandaloneCppnGenome",
    ) -> tuple["StandaloneCppnGenome", "StandaloneCppnGenome"]:
        left_fitness = float(getattr(genome1, "fitness", 0.0) or 0.0)
        right_fitness = float(getattr(genome2, "fitness", 0.0) or 0.0)
        if right_fitness > left_fitness:
            return genome2, genome1
        if right_fitness == left_fitness and int(genome2.key) < int(genome1.key):
            return genome2, genome1
        return genome1, genome2

    def _new_node_key(self) -> int:
        key = int(self.next_node_key)
        self.next_node_key = key + 1
        return key

    def _mutate_add_node(self, config: StandaloneCppnGenomeConfig) -> None:
        enabled = [gene for gene in self.connections.values() if gene.enabled]
        if not enabled:
            return
        base = random.choice(enabled)
        base.enabled = False
        hidden_key = self._new_node_key()
        self.nodes[hidden_key] = StandaloneCppnNodeGene.create(hidden_key, config)
        source_order = float(self.node_order.get(base.key[0], 0.0))
        target_order = float(self.node_order.get(base.key[1], 1.0))
        self.node_order[hidden_key] = (source_order + target_order) * 0.5
        self.connections[(int(base.key[0]), int(hidden_key))] = StandaloneCppnConnectionGene.create(
            (int(base.key[0]), int(hidden_key)),
            config,
            weight=1.0,
            enabled=True,
        )
        self.connections[(int(hidden_key), int(base.key[1]))] = StandaloneCppnConnectionGene.create(
            (int(hidden_key), int(base.key[1])),
            config,
            weight=float(base.weight),
            enabled=True,
        )

    def _mutate_delete_node(self) -> None:
        hidden_keys = [key for key in self.nodes if key not in self.output_keys]
        if not hidden_keys:
            return
        delete_key = random.choice(hidden_keys)
        self.nodes.pop(delete_key, None)
        self.node_order.pop(delete_key, None)
        for key in list(self.connections.keys()):
            if int(key[0]) == int(delete_key) or int(key[1]) == int(delete_key):
                self.connections.pop(key, None)

    def _mutate_add_connection(self, config: StandaloneCppnGenomeConfig) -> None:
        node_keys = [*self.input_keys, *sorted(self.nodes.keys())]
        candidates: list[tuple[int, int]] = []
        for source_key in node_keys:
            if source_key in self.output_keys:
                continue
            source_order = float(self.node_order.get(source_key, 0.0))
            for target_key in sorted(self.nodes.keys()):
                if target_key in self.input_keys:
                    continue
                target_order = float(self.node_order.get(target_key, 1.0))
                if source_order >= target_order:
                    continue
                pair = (int(source_key), int(target_key))
                if pair in self.connections:
                    continue
                candidates.append(pair)
        if not candidates:
            return
        pair = random.choice(candidates)
        self.connections[pair] = StandaloneCppnConnectionGene.create(pair, config)

    def _mutate_delete_connection(self) -> None:
        if not self.connections:
            return
        key = random.choice(list(self.connections.keys()))
        self.connections.pop(key, None)

    def _repair(self, config: StandaloneCppnGenomeConfig) -> None:
        for key in list(self.connections.keys()):
            source_key, target_key = key
            if source_key not in self.input_keys and source_key not in self.nodes:
                self.connections.pop(key, None)
                continue
            if target_key not in self.nodes:
                self.connections.pop(key, None)
                continue
            if float(self.node_order.get(source_key, 0.0)) >= float(self.node_order.get(target_key, 1.0)):
                self.connections.pop(key, None)
        for output_key in self.output_keys:
            if output_key not in self.nodes:
                self.nodes[output_key] = StandaloneCppnNodeGene.create(output_key, config)
                self.node_order[output_key] = 1.0


class StandaloneCppnNetwork:
    def __init__(self, input_keys: list[int], output_keys: list[int], node_evals: list[tuple[int, Any, Any, float, float, list[tuple[int, float]]]]):
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        self.node_evals = list(node_evals)
        self.values = {int(key): 0.0 for key in [*self.input_keys, *self.output_keys]}

    def activate(self, inputs: list[float]) -> list[float]:
        if len(inputs) != len(self.input_keys):
            raise RuntimeError(f"Expected {len(self.input_keys)} inputs, got {len(inputs)}")
        for key, value in zip(self.input_keys, inputs):
            self.values[int(key)] = float(value)
        for node_key, activation_fn, aggregation_fn, bias, response, links in self.node_evals:
            weighted = [self.values.get(int(source_key), 0.0) * float(weight) for source_key, weight in links]
            aggregated = aggregation_fn(weighted)
            self.values[int(node_key)] = float(activation_fn(float(bias) + (float(response) * float(aggregated))))
        return [float(self.values.get(int(key), 0.0)) for key in self.output_keys]

    @classmethod
    def create(cls, genome: StandaloneCppnGenome, config: StandaloneCppnGenomeConfig) -> "StandaloneCppnNetwork":
        node_evals: list[tuple[int, Any, Any, float, float, list[tuple[int, float]]]] = []
        ordered_nodes = sorted(
            genome.nodes.keys(),
            key=lambda key: (
                float(genome.node_order.get(key, 1.0)),
                int(key),
            ),
        )
        enabled_connections = [gene for gene in genome.connections.values() if gene.enabled]
        for node_key in ordered_nodes:
            gene = genome.nodes[node_key]
            incoming: list[tuple[int, float]] = []
            for connection in enabled_connections:
                if int(connection.key[1]) != int(node_key):
                    continue
                incoming.append((int(connection.key[0]), float(connection.weight)))
            node_evals.append(
                (
                    int(node_key),
                    _activation_fn(gene.activation),
                    _aggregation_fn(gene.aggregation),
                    float(gene.bias),
                    float(gene.response),
                    incoming,
                )
            )
        return cls(config.input_keys, config.output_keys, node_evals)


Genome = StandaloneCppnGenome
GenomeConfig = StandaloneCppnGenomeConfig
Network = StandaloneCppnNetwork
NodeGene = StandaloneCppnNodeGene
ConnectionGene = StandaloneCppnConnectionGene
