import numpy as np
from mccfr.nodes import CFRNode

NUM_ACTIONS = 3  # 0=fold, 1=call, 2=raise

class StrategyManager:

    def __init__(self):
        # The strategy table: info set key -> CFRNode
        self.nodes = {}

    def get_node(self, info_set_key):
        """Retrieve or create a CFRNode for a given information set."""
        if info_set_key not in self.nodes:
            self.nodes[info_set_key] = CFRNode()
        return self.nodes[info_set_key]

    def get_current_strategy(self, info_set_key, realization_weight):
        """Get the current mixed strategy for training."""
        node = self.get_node(info_set_key)
        return node.get_strategy(realization_weight)

    def get_average_strategy(self, info_set_key):
        """Get the converged average strategy for inference."""
        node = self.get_node(info_set_key)
        return node.get_average_strategy()

    def update_regrets(self, info_set_key, regret_updates):
        """Add counterfactual regrets to the node's regret table."""
        node = self.get_node(info_set_key)
        node.regret_sum += regret_updates

    def sample_action(self, info_set_key, action_mask):
        """
        Sample an action from the average strategy,
        respecting the legal action mask from PettingZoo.
        """
        strategy = self.get_average_strategy(info_set_key)

        # Zero out illegal actions and renormalize
        masked = strategy * action_mask
        total = masked.sum()

        if total > 0:
            masked /= total
        else:
            # Fall back to uniform over legal actions if all probs zeroed out
            masked = action_mask / action_mask.sum()

        return np.random.choice(len(strategy), p=masked)

    def save(self, filepath):
        """Persist the strategy table to disk."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.nodes, f)

    def load(self, filepath):
        """Load a previously trained strategy table from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            self.nodes = pickle.load(f)