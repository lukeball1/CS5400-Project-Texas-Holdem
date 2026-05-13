import numpy as np

NUM_ACTIONS = 5  # 0=fold, 1=call, 2=raise-half, 3=raise-full, 4=raise-all-in

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
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        
        # Bias toward calling instead of uniform fallback
        # when this node has never been visited during training
        fallback = np.zeros(NUM_ACTIONS)
        fallback[0] = 0.00  # fold   — never fold as default
        fallback[1] = 0.60  # call   — safest default play
        fallback[2] = 0.15  # raise half
        fallback[3] = 0.15  # raise full
        fallback[4] = 0.10  # all in
        return fallback