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

        # Should account for all raise variants
        FOLD         = 0
        CALL         = 1
        RAISE_HALF   = 2
        RAISE_FULL   = 3
        ALL_IN       = 4

        if strength >= 0.85:
            # Very strong hand - be aggressive
            for action in [ALL_IN, RAISE_FULL, RAISE_HALF, CALL]:
                if action in legal_actions:
                    return action
        elif strength >= 0.6:
            for action in [RAISE_FULL, RAISE_HALF, CALL]:
                if action in legal_actions:
                    return action
        elif strength >= 0.3:
            for action in [CALL, RAISE_HALF]:
                if action in legal_actions:
                    return action
        else:
            for action in [FOLD, CALL]:
                if action in legal_actions:
                    return action

        return random.choice(legal_actions)

    def estimate_hand_strength(self, observation):
        if isinstance(observation, dict):
            observation = observation.get("observation", observation)

        if len(observation) == 0:
            return 0.5

        # Actual layout from PettingZoo (54 elements total):
        # [0:52]  → one-hot card vector (hole + community cards combined)
        # [52]    → normalized pot size
        # [53]    → normalized call amount / chip count

        card_vec  = observation[:52]
        pot_size  = observation[52]
        call_amt  = observation[53]

        # Count active cards — max is 7 (2 hole + 5 community)
        num_cards = sum(card_vec)
        card_strength = min(1.0, num_cards / 7.0)

        # Factor in pot odds — if call amount is high relative
        # to pot, we need a stronger hand to justify calling
        if pot_size > 0:
            pot_odds = call_amt / (pot_size + call_amt + 1e-6)
        else:
            pot_odds = 0.0

        # Combine card strength with pot odds adjustment
        # Higher pot odds = need stronger hand to call profitably
        strength = card_strength * (1.0 - pot_odds * 0.3)

        return max(0.0, min(1.0, strength))
    
    def reset(self):
        pass

    def record_action(self, action):
        pass