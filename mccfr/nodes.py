import numpy as np

NUM_ACTIONS = 3  # 0=fold, 1=call, 2=raise

class CFRNode:

    def __init__(self):
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)

    def get_strategy(self, realization_weight):
        """
        Applies regret matching to produce a current mixed strategy,
        then accumulates it into strategy_sum weighted by reach probability.
        """
        strategy = np.maximum(self.regret_sum, 0)
        total = strategy.sum()

        if total > 0:
            strategy /= total
        else:
            # If all regrets are zero or negative, play uniformly
            strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS

        self.strategy_sum += realization_weight * strategy
        return strategy

    def get_average_strategy(self):
        """
        Returns the average strategy over all training iterations.
        This is what the agent actually uses at inference time.
        """
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS