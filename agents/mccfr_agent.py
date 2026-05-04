from agents.base_agent import BaseAgent
from mccfr.info_set import InfoSet
from mccfr.strategy import StrategyManager
from mccfr.trainer import MCCFRTrainer
import numpy as np

class MCCFRAgent(BaseAgent):
    
    def __init__(self, strategy_path=None):
        self.strategy = StrategyManager()
        super().__init__(name="MCCFRAgent")
        
        # if strategy exists, load it.
        if strategy_path:
            self.strategy.load(strategy_path)

    def train(self, environment, iterations=10000, save_path=None):
        """
        Run the MCCFR training loop against the environment.
        Call this before running matchups if no strategy_path was provided.
        """
        trainer = MCCFRTrainer(environment, self.strategy)
        trainer.train(iterations)

        # Optionally persist the trained strategy to disk
        if save_path:
            self.strategy.save(save_path)
            print(f"Strategy saved to {save_path}")
    
    def act(self, observation, action_mask):
        """
        Called by runner.py every time it's the agent's turn.
        Builds the information set from the observation, then
        samples an action from the converged average strategy.
        """
        info_set = InfoSet(
            hole_cards=self._extract_hole_cards(observation),
            community_cards=self._extract_community_cards(observation),
            action_history=self._extract_action_history(observation),
            round_name=self._extract_round(observation)
        )

        return self.strategy.sample_action(info_set.key(), action_mask)
    
    def reset(self):
        """
        Called by runner.py at the start of each new game, just clears action history
        """
        self.action_history = []

    def record_action(self, action):
        """
        Records an action taken by an agent.
        """
        self.action_history.append(action)
        
    def _extract_hole_cards(self, obs):
        return tuple(np.where(obs[:52] == 1)[0])

    def _extract_community_cards(self, obs):
        return tuple(np.where(obs[52:104] == 1)[0])

    def _extract_round(self, obs):
        rounds = ['preflop', 'flop', 'turn', 'river']
        round_vec = obs[104:108]
        idx = int(np.argmax(round_vec))
        return rounds[idx]

    def _extract_action_history(self, obs):
        """
        Returns the action history tracked internally per hand.
        This is maintained via reset() and record_action() calls
        from runner.py rather than extracted from the obs vector.
        """
        return tuple(self.action_history) if hasattr(self, 'action_history') else ()
        