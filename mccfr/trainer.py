import numpy as np
from mccfr.info_set import InfoSet
from mccfr.strategy import StrategyManager

NUM_ACTIONS = 5  # 0=fold, 1=call, 2=raise-half, 3=raise-full, 4=raise-all-in

class MCCFRTrainer:

    def __init__(self, environment, strategy_manager: StrategyManager):
        self.env = environment
        self.strategy = strategy_manager

    def train(self, iterations: int):
        """
        Run the MCCFR training loop for a given number of iterations.
        Each iteration simulates a hand and updates regret/strategy tables.
        """
        for i in range(iterations):
            self.env.reset()
            self._traverse(
                agent_name=self.env.current_agent(),
                action_history=[],
                reach_probs={agent: 1.0 for agent in self.env.get_agents()}
            )

            if i % 1000 == 0:
                print(f"Iteration {i}/{iterations} complete")

    def _traverse(self, agent_name, action_history, reach_probs):
        """
        Recursively traverse the game tree, updating regrets as we go.
        Returns the expected utility from the current node.
        """
        if self.env.is_done():
            _, reward, _, _, _ = self.env.last()
            return reward

        obs, action_mask = self.env.observe(agent_name)

        # Build the information set key for this decision point
        info_set = InfoSet(
            hole_cards=self._extract_hole_cards(obs),
            community_cards=self._extract_community_cards(obs),
            action_history=action_history,
            round_name=self._extract_round(obs)
        )
        key = info_set.key()

        # Get current strategy via regret matching
        strategy = self.strategy.get_current_strategy(
            key, reach_probs[agent_name]
        )

        # Mask illegal actions and renormalize
        legal_strategy = strategy * action_mask
        total = legal_strategy.sum()
        if total > 0:
            legal_strategy /= total
        else:
            legal_strategy = action_mask / action_mask.sum()

        # Sample one action (Monte Carlo: we don't explore all branches)
        action = np.random.choice(NUM_ACTIONS, p=legal_strategy)

        # Update reach probability for this agent
        new_reach_probs = reach_probs.copy()
        new_reach_probs[agent_name] *= legal_strategy[action]

        # Step the environment and recurse
        self.env.step(action)
        next_agent = self.env.current_agent()
        utility = self._traverse(
            next_agent,
            action_history + [action],
            new_reach_probs
        )

        # Compute counterfactual regrets for each action
        opponent_reach = np.prod([
            v for k, v in reach_probs.items() if k != agent_name
        ])
        regret_updates = np.zeros(NUM_ACTIONS)
        for a in range(NUM_ACTIONS):
            if action_mask[a]:
                # Regret = counterfactual value of action a - actual utility
                regret_updates[a] = opponent_reach * (
                    (1.0 if a == action else 0.0) * utility - utility
                )

        self.strategy.update_regrets(key, regret_updates)
        return utility

    def _extract_hole_cards(self, obs):
        """Extract hole card indices from the observation vector."""
        # PettingZoo encodes cards as one-hot vectors in the observation
        # The first 52 values represent the player's hole cards
        return tuple(np.where(obs[:52] == 1)[0])

    def _extract_community_cards(self, obs):
        """Extract community card indices from the observation vector."""
        # Community cards are in the main 52-card section, excluding hole cards
        hole_card_indices = set(np.where(obs[:52] == 1)[0])
        community_indices = [i for i in range(52) if i not in hole_card_indices and obs[i] == 1]
        return tuple(community_indices)

    def _extract_round(self, obs):
        """Extract the current round from the observation vector."""
        # Round indicator is a scalar at index 53
        rounds = ['preflop', 'flop', 'turn', 'river', 'showdown']
        round_idx = int(obs[53])
        if round_idx < len(rounds):
            return rounds[round_idx]
        else:
            return 'showdown'