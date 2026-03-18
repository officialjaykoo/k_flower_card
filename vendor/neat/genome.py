"""Handles genomes (individuals in the population)."""
import copy
import sys
from itertools import count
from random import choice, random, shuffle

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.attributes import FloatAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import creates_cycle
from neat.graphs import required_for_output

DECISION_THRESHOLD_NAMES = (
    'go_stop_threshold',
    'shaking_threshold',
    'president_threshold',
    'gukjin_threshold',
)


def _split_head_enabled(config):
    return bool(getattr(config, '_codex_head_split_enabled', False))


def _split_head_input_keys_by_output(config):
    return getattr(config, '_codex_head_input_keys_by_output', {}) or {}


def _input_gate_enabled(config):
    return bool(getattr(config, '_codex_input_gate_enabled', False))


def _allowed_input_keys_by_output(config):
    mapping = _split_head_input_keys_by_output(config)
    if mapping:
        return {
            int(output_key): {int(item) for item in values}
            for output_key, values in mapping.items()
        }
    input_keys = {int(key) for key in getattr(config, 'input_keys', [])}
    return {
        int(output_key): set(input_keys)
        for output_key in getattr(config, 'output_keys', [])
    }


def _sanitize_input_gate_sets(gate_sets, config):
    allowed_map = _allowed_input_keys_by_output(config)
    min_map = getattr(config, '_codex_input_gate_min_active_by_output', {}) or {}
    normalized = {}
    for output_key, allowed in allowed_map.items():
        current = {
            int(item)
            for item in set((gate_sets or {}).get(int(output_key), set()) or set())
            if int(item) in allowed
        }
        minimum = max(0, int(min_map.get(int(output_key), 0) or 0))
        if minimum > len(allowed):
            minimum = len(allowed)
        if len(current) < minimum:
            missing = list(allowed - current)
            shuffle(missing)
            current.update(missing[:max(0, minimum - len(current))])
        normalized[int(output_key)] = current
    return normalized


def _ensure_input_gate_sets(genome, config):
    if not _input_gate_enabled(config):
        setattr(genome, 'input_gate_keys_by_output', {})
        return {}
    normalized = _sanitize_input_gate_sets(getattr(genome, 'input_gate_keys_by_output', {}) or {}, config)
    setattr(genome, 'input_gate_keys_by_output', normalized)
    return normalized


def _initialize_input_gate_sets(genome, config):
    if not _input_gate_enabled(config):
        setattr(genome, 'input_gate_keys_by_output', {})
        return {}
    allowed_map = _allowed_input_keys_by_output(config)
    initial_map = getattr(config, '_codex_input_gate_initial_active_by_output', {}) or {}
    gate_sets = {}
    for output_key, allowed in allowed_map.items():
        target = int(initial_map.get(int(output_key), len(allowed)) or 0)
        if target <= 0 or target >= len(allowed):
            gate_sets[int(output_key)] = set(allowed)
            continue
        pool = list(allowed)
        shuffle(pool)
        gate_sets[int(output_key)] = set(pool[:target])
    normalized = _sanitize_input_gate_sets(gate_sets, config)
    setattr(genome, 'input_gate_keys_by_output', normalized)
    return normalized


def _resolve_node_head(genome, config, node_key):
    key = int(node_key)
    if key in getattr(config, 'output_keys', []):
        mapping = _split_head_input_keys_by_output(config)
        return key if key in mapping else None
    node_heads = getattr(genome, 'node_heads', {}) or {}
    head = node_heads.get(key)
    if head is None:
        return None
    return int(head)


def _connection_allowed(genome, config, input_key, output_key):
    if int(output_key) < 0:
        return False
    if not _split_head_enabled(config):
        return True

    output_head = _resolve_node_head(genome, config, output_key)
    if output_head is None:
        return False

    input_key = int(input_key)
    if input_key in getattr(config, 'input_keys', []):
        if input_key not in _split_head_input_keys_by_output(config).get(int(output_head), set()):
            return False
        if _input_gate_enabled(config):
            return input_key in _ensure_input_gate_sets(genome, config).get(int(output_head), set())
        return True

    if input_key in getattr(config, 'output_keys', []):
        return False

    input_head = _resolve_node_head(genome, config, input_key)
    return input_head is not None and int(input_head) == int(output_head)


class DefaultGenomeConfig:
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params, section_name='DefaultGenome'):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected')]

        self.decision_threshold_attributes = [FloatAttribute(name) for name in DECISION_THRESHOLD_NAMES]
        for attr in self.decision_threshold_attributes:
            self._params += attr.get_config_params()

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params, section_name))

        self.node_gene_type.validate_attributes(self)
        self.connection_gene_type.validate_attributes(self)
        for attr in self.decision_threshold_attributes:
            attr.validate(self)

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)

        self.node_indexer = None
        
        # Innovation tracker will be set by Population/Reproduction
        # This enables same-generation deduplication per NEAT paper (Stanley & Miikkulainen, 2002)
        self.innovation_tracker = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write(f'initial_connection      = {self.initial_connection} {self.connection_fraction}\n')
        else:
            f.write(f'initial_connection      = {self.initial_connection}\n')

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if 'initial_connection' not in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            if node_dict:
                self.node_indexer = count(max(list(node_dict)) + 1)
            else:
                # No existing nodes; start from num_outputs.
                self.node_indexer = count(self.num_outputs)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)

    def __getstate__(self):
        """Prepare config for pickling by converting node_indexer to a picklable form."""
        state = self.__dict__.copy()
        # Convert the itertools.count object to an integer representing the next value
        # We peek at the value by calling next() and storing it, then the counter advances
        if self.node_indexer is not None:
            # We need to save the next value that would be returned
            # Since calling next() consumes the value, we save it and will recreate
            # the counter starting from that value when unpickling
            state['_node_indexer_next_value'] = next(self.node_indexer)
            state['node_indexer'] = None
        else:
            state['_node_indexer_next_value'] = None
        return state

    def __setstate__(self, state):
        """Restore config from pickled state, recreating node_indexer."""
        _node_indexer_next_value = state.pop('_node_indexer_next_value', None)
        self.__dict__.update(state)
        # Recreate the count object starting from the saved next value
        if _node_indexer_next_value is not None:
            # Recreate counter starting from the value we saved
            self.node_indexer = count(_node_indexer_next_value)
        else:
            self.node_indexer = None


class DefaultGenome:
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict, cls.__name__)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}
        self.node_heads = {}
        self.input_gate_keys_by_output = {}

        # Fitness results.
        self.fitness = None
        for name in DECISION_THRESHOLD_NAMES:
            setattr(self, name, 0.0)

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
        self.node_heads = {}
        self.input_gate_keys_by_output = {}
        for attr in getattr(config, 'decision_threshold_attributes', []):
            setattr(self, attr.name, float(attr.init_value(config)))

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        _initialize_input_gate_sets(self, config)

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            if _split_head_enabled(config):
                raise RuntimeError("split-head mode requires num_hidden = 0 at initialization")
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # Add connections based on initial connectivity type.

        if 'fs_neat' in config.initial_connection:
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                        "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                        "\tif not, set initial_connection = fs_neat_hidden",
                        sep='\n', file=sys.stderr)
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep='\n', file=sys.stderr)
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        f"\tif this is desired, set initial_connection = partial_nodirect {config.connection_fraction};",
                        f"\tif not, set initial_connection = partial_direct {config.connection_fraction}",
                        sep='\n', file=sys.stderr)
                self.connect_partial_nodirect(config)

    def configure_crossover(self, genome1, genome2, config):
        """
        Configure a new genome by crossover from two parent genomes.
        
        Implements NEAT paper (Stanley & Miikkulainen, 2002, p. 108) crossover:
        "When crossing over, the genes in both genomes with the same innovation
        numbers are lined up. Genes are randomly chosen from either parent at
        matching genes, whereas all excess or disjoint genes are always included
        from the more fit parent."
        """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1
        self.node_heads = {}
        self.input_gate_keys_by_output = {}

        if _input_gate_enabled(config):
            allowed_map = _allowed_input_keys_by_output(config)
            parent1_gates = _sanitize_input_gate_sets(
                getattr(parent1, 'input_gate_keys_by_output', {}) or {int(k): set(v) for k, v in allowed_map.items()},
                config,
            )
            parent2_gates = _sanitize_input_gate_sets(
                getattr(parent2, 'input_gate_keys_by_output', {}) or {int(k): set(v) for k, v in allowed_map.items()},
                config,
            )
            crossed_gates = {}
            for output_key, allowed in allowed_map.items():
                current = set()
                for input_key in allowed:
                    in_parent1 = int(input_key) in parent1_gates.get(int(output_key), set())
                    in_parent2 = int(input_key) in parent2_gates.get(int(output_key), set())
                    if in_parent1 and in_parent2:
                        current.add(int(input_key))
                    elif in_parent1 and random() < 0.75:
                        current.add(int(input_key))
                    elif in_parent2 and random() < 0.25:
                        current.add(int(input_key))
                crossed_gates[int(output_key)] = current
            self.input_gate_keys_by_output = _sanitize_input_gate_sets(crossed_gates, config)

        # Inherit connection genes by innovation number
        # Build innovation number mappings for both parents
        parent1_innovations = {cg.innovation: cg for cg in parent1.connections.values()}
        parent2_innovations = {cg.innovation: cg for cg in parent2.connections.values()}
        
        # Get all innovation numbers from both parents
        all_innovations = set(parent1_innovations.keys()) | set(parent2_innovations.keys())
        
        for innovation_num in all_innovations:
            cg1 = parent1_innovations.get(innovation_num)
            cg2 = parent2_innovations.get(innovation_num)
            
            if cg1 is not None and cg2 is not None:
                # Matching genes: homologous genes are lined up by innovation number
                # Randomly inherit from either parent
                if cg1.key != cg2.key:
                    # This can happen if innovation numbers get reused (e.g., after checkpoint restore
                    # in a very long evolution run). Treat as disjoint genes instead of matching.
                    import warnings
                    warnings.warn(
                        f"Innovation number collision: innovation {innovation_num} assigned to both "
                        f"{cg1.key} and {cg2.key}. Treating as disjoint genes.",
                        RuntimeWarning
                    )
                    # Take the gene from the fitter parent
                    new_gene = cg1.copy()
                    # For feed-forward networks, check if this connection would create a cycle
                    if config.feed_forward and creates_cycle(list(self.connections), new_gene.key):
                        continue
                    self.connections[new_gene.key] = new_gene
                else:
                    new_gene = cg1.crossover(cg2)
                    # For feed-forward networks, check if this connection would create a cycle
                    if config.feed_forward and creates_cycle(list(self.connections), new_gene.key):
                        continue
                    self.connections[new_gene.key] = new_gene
            elif cg1 is not None:
                # Disjoint or excess gene from fittest parent (parent1)
                new_gene = cg1.copy()
                # For feed-forward networks, check if this connection would create a cycle
                if config.feed_forward and creates_cycle(list(self.connections), new_gene.key):
                    continue
                self.connections[new_gene.key] = new_gene
            # Note: genes only in parent2 (less fit) are not inherited

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

        for attr in getattr(config, 'decision_threshold_attributes', []):
            value1 = getattr(parent1, attr.name, None)
            value2 = getattr(parent2, attr.name, None)
            if value1 is None and value2 is None:
                value = attr.init_value(config)
            elif value1 is None:
                value = value2
            elif value2 is None:
                value = value1
            else:
                value = value1 if random() < 0.5 else value2
            setattr(self, attr.name, float(value))

        if _split_head_enabled(config):
            parent1_heads = getattr(parent1, 'node_heads', {}) or {}
            parent2_heads = getattr(parent2, 'node_heads', {}) or {}
            for key in self.nodes:
                if key in config.output_keys:
                    continue
                head1 = parent1_heads.get(key)
                head2 = parent2_heads.get(key)
                if head1 is None and head2 is None:
                    continue
                if head1 is None:
                    self.node_heads[int(key)] = int(head2)
                elif head2 is None:
                    self.node_heads[int(key)] = int(head1)
                else:
                    self.node_heads[int(key)] = int(head1 if random() < 0.5 else head2)
            self._prune_invalid_split_connections(config)

    def _prune_invalid_split_connections(self, config):
        if not _split_head_enabled(config):
            return
        invalid_keys = []
        for key in list(self.connections):
            in_node, out_node = key
            if not _connection_allowed(self, config, in_node, out_node):
                invalid_keys.append(key)
        for key in invalid_keys:
            del self.connections[key]

    def _valid_new_connection_candidates(self, config):
        possible_outputs = list(self.nodes)
        possible_inputs = list((set(self.nodes) - set(config.output_keys)) | set(config.input_keys))
        candidates = []
        existing_connections = list(self.connections)
        for out_node in possible_outputs:
            for in_node in possible_inputs:
                key = (in_node, out_node)
                if key in self.connections:
                    continue
                if in_node in config.output_keys and out_node in config.output_keys:
                    continue
                if not _connection_allowed(self, config, in_node, out_node):
                    continue
                if config.feed_forward and creates_cycle(existing_connections, key):
                    continue
                candidates.append(key)
        return candidates

    def mutate(self, config):
        """ Mutates this genome. """

        if _input_gate_enabled(config):
            gate_sets = _ensure_input_gate_sets(self, config)
            enable_prob = float(getattr(config, '_codex_input_gate_enable_prob', 0.0) or 0.0)
            disable_prob = float(getattr(config, '_codex_input_gate_disable_prob', 0.0) or 0.0)
            min_map = getattr(config, '_codex_input_gate_min_active_by_output', {}) or {}
            for output_key, allowed in _allowed_input_keys_by_output(config).items():
                enabled_inputs = set(gate_sets.get(int(output_key), set()))
                ordered_allowed = list(allowed)
                shuffle(ordered_allowed)
                minimum = max(0, int(min_map.get(int(output_key), 0) or 0))
                for input_key in ordered_allowed:
                    if int(input_key) in enabled_inputs:
                        if len(enabled_inputs) > minimum and random() < disable_prob:
                            enabled_inputs.remove(int(input_key))
                    else:
                        if random() < enable_prob:
                            enabled_inputs.add(int(input_key))
                if len(enabled_inputs) < minimum:
                    missing = list(set(allowed) - enabled_inputs)
                    shuffle(missing)
                    enabled_inputs.update(missing[:max(0, minimum - len(enabled_inputs))])
                gate_sets[int(output_key)] = enabled_inputs
            self.input_gate_keys_by_output = _sanitize_input_gate_sets(gate_sets, config)
            for conn_key in list(self.connections):
                in_node = int(conn_key[0])
                out_node = int(conn_key[1])
                if in_node < 0 and not _connection_allowed(self, config, in_node, out_node):
                    self.connections.pop(conn_key, None)

        if config.single_structural_mutation:
            div = max(1, (config.node_add_prob + config.node_delete_prob +
                          config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob) / div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob) / div):
                self.mutate_delete_connection()
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

        for attr in getattr(config, 'decision_threshold_attributes', []):
            current_value = float(getattr(self, attr.name, attr.init_value(config)))
            setattr(self, attr.name, float(attr.mutate_value(current_value, config)))

    def mutate_add_node(self, config):
        """
        Add a new node by splitting an existing connection.
        
        Uses innovation tracking per NEAT paper (Stanley & Miikkulainen, 2002):
        If multiple genomes split the same connection in one generation, the resulting
        connections receive matching innovation numbers.
        """
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return
        
        assert config.innovation_tracker is not None, (
            "Innovation tracker must be set before genome mutations. "
            "This should be set by the reproduction module."
        )

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)

        # Make the new node as neutral as possible with respect to the
        # existing connection: start with zero bias regardless of the
        # global bias initialization distribution.
        if hasattr(ng, "bias"):
            ng.bias = 0.0

        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        new_node_head = _resolve_node_head(self, config, o) if _split_head_enabled(config) else None
        if _split_head_enabled(config) and new_node_head is None:
            conn_to_split.enabled = True
            del self.nodes[new_node_id]
            return
        if new_node_head is not None:
            self.node_heads[int(new_node_id)] = int(new_node_head)
        
        # Get innovation numbers for the two new connections
        # These are keyed by the connection being split, so multiple genomes splitting
        # the same connection get matching innovation numbers
        in_innovation = config.innovation_tracker.get_innovation_number(
            i, new_node_id, 'add_node_in'
        )
        out_innovation = config.innovation_tracker.get_innovation_number(
            new_node_id, o, 'add_node_out'
        )
        
        # Add the two new connections with their innovation numbers
        in_connection = self.add_connection(config, i, new_node_id, 1.0, True, innovation=in_innovation)
        out_connection = self.add_connection(config, new_node_id, o, conn_to_split.weight, True, innovation=out_innovation)
        if in_connection is None or out_connection is None:
            conn_to_split.enabled = True
            self.node_heads.pop(int(new_node_id), None)
            del self.nodes[new_node_id]
            if in_connection is not None:
                self.connections.pop(in_connection.key, None)
            if out_connection is not None:
                self.connections.pop(out_connection.key, None)

    def add_connection(self, config, input_key, output_key, weight, enabled, innovation=None):
        """Add a connection to this genome. If innovation is None, gets a new one from tracker."""
        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        if not _connection_allowed(self, config, input_key, output_key):
            return None
        
        key = (input_key, output_key)
        
        # Get innovation number if not provided
        if innovation is None:
            assert config.innovation_tracker is not None, "Innovation tracker must be set"
            innovation = config.innovation_tracker.get_innovation_number(
                input_key, output_key, 'add_connection'
            )
        
        connection = config.connection_gene_type(key, innovation=innovation)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection
        return connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        
        Uses innovation tracking per NEAT paper (Stanley & Miikkulainen, 2002):
        If multiple genomes in the same generation add the same connection,
        they receive the same innovation number.
        """
        assert config.innovation_tracker is not None, (
            "Innovation tracker must be set before genome mutations. "
            "This should be set by the reproduction module."
        )
        candidates = self._valid_new_connection_candidates(config)
        if not candidates:
            return
        key = choice(candidates)
        in_node, out_node = key

        # Get innovation number for this connection
        # Same connection added by multiple genomes in same generation gets same number
        innovation = config.innovation_tracker.get_innovation_number(
            in_node, out_node, 'add_connection'
        )
        cg = self.create_connection(config, in_node, out_node, innovation)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in self.nodes if k not in config.output_keys]
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]
        self.node_heads.pop(int(del_key), None)

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        threshold_distance = 0.0
        threshold_attrs = getattr(config, 'decision_threshold_attributes', [])
        if threshold_attrs:
            diffs = 0.0
            for attr in threshold_attrs:
                diffs += abs(float(getattr(self, attr.name, 0.0)) - float(getattr(other, attr.name, 0.0)))
            threshold_distance = (config.compatibility_weight_coefficient * diffs) / len(threshold_attrs)
        gate_distance = 0.0
        if _input_gate_enabled(config):
            self_gates = _ensure_input_gate_sets(self, config)
            other_gates = _ensure_input_gate_sets(other, config)
            diffs = 0.0
            total = 0
            for output_key, allowed in _allowed_input_keys_by_output(config).items():
                for input_key in allowed:
                    total += 1
                    if (int(input_key) in self_gates.get(int(output_key), set())) != (
                        int(input_key) in other_gates.get(int(output_key), set())
                    ):
                        diffs += 1.0
            if total > 0:
                gate_distance = (config.compatibility_weight_coefficient * diffs) / total

        return node_distance + connection_distance + threshold_distance + gate_distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = f"Key: {self.key}\nFitness: {self.fitness}\nNodes:"
        if DECISION_THRESHOLD_NAMES:
            s += "\nThresholds:"
            for name in DECISION_THRESHOLD_NAMES:
                s += f"\n\t{name}={getattr(self, name, 0.0)}"
        for k, ng in self.nodes.items():
            s += f"\n\t{k} {ng!s}"
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    @staticmethod
    def create_node(config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id, innovation):
        """Create a new connection gene with the given innovation number."""
        connection = config.connection_gene_type((input_id, output_id), innovation=innovation)
        connection.init_attributes(config)
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        assert config.innovation_tracker is not None, "Innovation tracker must be set"
        for output_id in config.output_keys:
            if _split_head_enabled(config):
                allowed_inputs = list(_split_head_input_keys_by_output(config).get(int(output_id), set()))
                if not allowed_inputs:
                    continue
                input_id = choice(allowed_inputs)
            else:
                input_id = choice(config.input_keys)
            if not _connection_allowed(self, config, input_id, output_id):
                continue
            innovation = config.innovation_tracker.get_innovation_number(
                input_id, output_id, 'initial_connection'
            )
            connection = self.create_connection(config, input_id, output_id, innovation)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        assert config.innovation_tracker is not None, "Innovation tracker must be set"
        others = [i for i in self.nodes if i not in config.input_keys]
        for output_id in others:
            if _split_head_enabled(config):
                target_head = _resolve_node_head(self, config, output_id)
                allowed_inputs = list(_split_head_input_keys_by_output(config).get(int(target_head), set())) if target_head is not None else []
                if not allowed_inputs:
                    continue
                input_id = choice(allowed_inputs)
            else:
                input_id = choice(config.input_keys)
            if not _connection_allowed(self, config, input_id, output_id):
                continue
            innovation = config.innovation_tracker.get_innovation_number(
                input_id, output_id, 'initial_connection'
            )
            connection = self.create_connection(config, input_id, output_id, innovation)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in self.nodes if i not in config.output_keys]
        output = [i for i in self.nodes if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    if _connection_allowed(self, config, input_id, h):
                        connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    if _connection_allowed(self, config, h, output_id):
                        connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    if _connection_allowed(self, config, input_id, output_id):
                        connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in self.nodes:
                if _connection_allowed(self, config, i, i):
                    connections.append((i, i))

        return connections

    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        assert config.innovation_tracker is not None, "Innovation tracker must be set"
        for input_id, output_id in self.compute_full_connections(config, False):
            innovation = config.innovation_tracker.get_innovation_number(
                input_id, output_id, 'initial_connection'
            )
            connection = self.create_connection(config, input_id, output_id, innovation)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        assert config.innovation_tracker is not None, "Innovation tracker must be set"
        for input_id, output_id in self.compute_full_connections(config, True):
            innovation = config.innovation_tracker.get_innovation_number(
                input_id, output_id, 'initial_connection'
            )
            connection = self.create_connection(config, input_id, output_id, innovation)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        assert config.innovation_tracker is not None, "Innovation tracker must be set"
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            innovation = config.innovation_tracker.get_innovation_number(
                input_id, output_id, 'initial_connection'
            )
            connection = self.create_connection(config, input_id, output_id, innovation)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        assert config.innovation_tracker is not None, "Innovation tracker must be set"
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            innovation = config.innovation_tracker.get_innovation_number(
                input_id, output_id, 'initial_connection'
            )
            connection = self.create_connection(config, input_id, output_id, innovation)
            self.connections[connection.key] = connection

    def get_pruned_copy(self, genome_config):
        used_node_genes, used_connection_genes = get_pruned_genes(self.nodes, self.connections,
                                                                  genome_config.input_keys, genome_config.output_keys)
        new_genome = DefaultGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        new_genome.node_heads = {
            int(key): int(value)
            for key, value in dict(getattr(self, 'node_heads', {}) or {}).items()
            if int(key) in used_node_genes
        }
        new_genome.input_gate_keys_by_output = {
            int(key): {int(item) for item in value}
            for key, value in dict(getattr(self, 'input_gate_keys_by_output', {}) or {}).items()
        }
        new_genome.fitness = self.fitness
        for name in DECISION_THRESHOLD_NAMES:
            setattr(new_genome, name, float(getattr(self, name, 0.0)))
        return new_genome


def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys):
    used_nodes = required_for_output(input_keys, output_keys, connection_genes)
    used_pins = used_nodes.union(input_keys)

    # Copy used nodes into a new genome.
    used_node_genes = {}
    for n in used_nodes:
        used_node_genes[n] = copy.deepcopy(node_genes[n])

    # Copy enabled and used connections into the new genome.
    used_connection_genes = {}
    for key, cg in connection_genes.items():
        in_node_id, out_node_id = key
        if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
            used_connection_genes[key] = copy.deepcopy(cg)

    return used_node_genes, used_connection_genes
