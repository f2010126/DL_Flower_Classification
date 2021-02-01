"""Cell search spaces which are used in the MacroGraph."""

import networkx as nx
import torch.nn as nn
import ConfigSpace

from primitives import Identity, Sum
from categorical_op import CategoricalOp


class CellA(nx.DiGraph, nn.Module):
    """CellA, as given in assignment_10.pdf."""

    def __init__(self, config=None, *args, **kwargs) -> None:
        nx.DiGraph.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self._init_config(config)
        self._build_graph()

    def _init_config(self, config: ConfigSpace.ConfigurationSpace) -> None:
        """Initialize the configuration for this cell config.

        Args:
            config: ConfigSpace.ConfigurationSpace to use for this cell.

        Returns:
            None
        """
        if config is None:
            self.config = CellA.get_configuration_space().get_default_configuration().get_dictionary()
        else:
            self.config = config.get_dictionary()

    def _build_graph(self) -> None:
        """Build the structure of the cell in networkx.

        Returns:
            None
        """
        self.add_node(0, op=Identity())
        self.add_node(1, op=CategoricalOp(self.config['op_node_1'], self.config['activation_node_1']))
        self.add_node(2, op=CategoricalOp(self.config['op_node_2'], self.config['activation_node_2']))
        self.add_node(3, op=Sum())

        self.add_edge(0, 1)
        self.add_edge(0, 2)
        self.add_edge(1, 3)
        self.add_edge(2, 3)

        for node in self.nodes:
            self.add_module(f"node{node}", self.nodes[node]['op'])

    def forward(self, x):
        x0 = self.nodes[0]['op'](x)
        x1 = self.nodes[1]['op'](x0)
        x2 = self.nodes[2]['op'](x0)
        x3 = self.nodes[3]['op']([x1, x2])
        return x3

    @staticmethod
    def get_configuration_space() -> ConfigSpace.ConfigurationSpace:
        """Create a ConfigurationSpace object which represents the configuration
           space of this cell config.

        Returns:
            ConfigSpace.ConfigurationSpace for this cell.
        """
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1', 'conv3x3', 'conv5x5', 'max3x3', 'identity']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))

        activation_choices = ['relu', 'prelu', 'gelu']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_node_1", activation_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_node_2", activation_choices))
        return cs


class CellB(nx.DiGraph, nn.Module):
    """CellB, as given in assignment_10.pdf."""

    def __init__(self, config=None, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self._init_config(config)
        self._build_graph()

    def _init_config(self, config):
        """Initialize the configuration for this cell config.

        Args:
            config: ConfigSpace.ConfigurationSpace to use for this cell.

        Returns:
            None
        """
        if config is None:
            self.config = CellB.get_configuration_space().get_default_configuration().get_dictionary()
        else:
            self.config = config.get_dictionary()

    def _build_graph(self):
        """Build the structure of the cell in networkx.

        Returns:
            None
        """
        self.add_node(0, op=Identity())

        # START TODO #################
        # nodes 1,2,3,4,5
        self.add_node(1, op=CategoricalOp(self.config['op_node_1'], self.config['activation_node_1']))
        self.add_node(2, op=CategoricalOp(self.config['op_node_2'], self.config['activation_node_2']))
        self.add_node(3, op=CategoricalOp(self.config['op_node_3'], self.config['activation_node_3']))
        self.add_node(4, op=CategoricalOp(self.config['op_node_4'], self.config['activation_node_4']))
        self.add_node(5, op=Sum())

        # edges b/w nodes
        self.add_edge(0, 1)
        self.add_edge(0, 2)
        self.add_edge(0, 3)
        self.add_edge(1, 5)
        self.add_edge(2, 4)
        self.add_edge(3, 5)
        self.add_edge(4, 5)

        for node in self.nodes:
            self.add_module(f"node{node}", self.nodes[node]['op'])

        # END TODO #################

    def forward(self, x):
        # Evaluate the graph in topological ordering
        topological_order = nx.algorithms.dag.topological_sort(self)

        # Cache the output of the nodes as node attributes
        # The first node is always the identity node, so its output is simply x
        self.nodes[0]['output'] = x

        #  Needed for node traversal in topological_order:
        top = list(topological_order)
        top.remove(0)  # node 0 has already been set
        for node in top:
            # START TODO #################
            # You can use self.predecessors(node) to get the predecessors of a node
            if isinstance(self.nodes[node]['op'], Sum):
                prev_op = [self.nodes[item]['output'] for item in self.predecessors(node)]
            else:  # hold for Identity and Conv and Max
                for item in self.predecessors(node):
                    prev_op = self.nodes[item]['output']

            self.nodes[node]['output'] = self.nodes[node]['op'](prev_op)
        last_node = top[-1]  # last item in the top sort
        # END TODO #################

        return self.nodes[last_node]['output']

    @staticmethod
    def get_configuration_space():
        """Create a ConfigurationSpace object which represents the configuration
           space of this cell config.

        Returns:
            ConfigSpace.ConfigurationSpace for this cell.
        """
        cs = ConfigSpace.ConfigurationSpace()
        ops_choices = ['conv1x1', 'conv3x3', 'conv5x5', 'max3x3', 'identity']
        activation_choices = ['relu', 'prelu', 'gelu']

        # START TODO #################
        # Please use the same naming convention as in CellA above or else the tests will not pass.
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))

        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_node_1", activation_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_node_2", activation_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_node_3", activation_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_node_4", activation_choices))
        # END TODO #################

        return cs
