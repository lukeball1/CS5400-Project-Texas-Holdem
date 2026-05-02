# Simple heuristic poker agent:
# Uses density of observation vector (number of active features) as a proxy for hand strength.
from agents.base_agent import BaseAgent
import random


class HeuristicAgent(BaseAgent):
    def __init__(self, name="HeuristicAgent"):
        super().__init__(name)

    def act(self, observation, action_mask):
        # Get all valid actions from the action mask
        legal_actions = [i for i, allowed in enumerate(action_mask) if allowed]

        if not legal_actions:
            return None

        strength = self.estimate_hand_strength(observation)
        # Map actions based on BaseAgent convention:
        FOLD = 0
        CALL = 1
        RAISE = 2

        # Decision logic:
        # High strength raise
        # Medium strength call
        # Low strength fold if possible
        if strength >= 0.6:
            if RAISE in legal_actions:
                return RAISE
            if CALL in legal_actions:
                return CALL

        elif strength >= 0.3:
            if CALL in legal_actions:
                return CALL

        else:
            if FOLD in legal_actions:
                return FOLD
            if CALL in legal_actions:
                return CALL

        return random.choice(legal_actions)

    def estimate_hand_strength(self, observation):
        obs = observation
        # Use number of active features as a rough proxy for hand strength.
        if isinstance(obs, dict):
            obs = obs.get("observation", obs)

        # Normalize to range [0, 1]
        ones = sum(obs)
        strength = ones / len(obs)

        return strength