# Simple heuristic poker agent:
# Uses density of observation vector (number of active features) as a proxy for hand strength.
from agents.base_agent import BaseAgent
import numpy as np
import random


class HeuristicAgent(BaseAgent):
    def __init__(self, name="HeuristicAgent"):
        super().__init__(name)

    def act(self, observation, action_mask):
        legal_actions = [i for i, allowed in enumerate(action_mask) if allowed]

        if not legal_actions:
            return None

        FOLD       = 0
        CALL       = 1
        RAISE_HALF = 2
        RAISE_FULL = 3
        ALL_IN     = 4

        strength       = self.estimate_hand_strength(observation)
        my_chips       = observation[52]
        opponent_chips = observation[53]
        chip_pressure  = opponent_chips / (my_chips + opponent_chips + 1e-6)

        # Recalibrated thresholds based on actual evaluator output range:
        #
        # Preflop (2 cards):
        #   High card only:  0.02 - 0.20
        #   Pocket pair:     0.40 - 0.45
        #
        # Postflop (5-7 cards):
        #   High card only:  0.02 - 0.20
        #   One pair:        0.20 - 0.45
        #   Two pair:        0.35 - 0.55
        #   Three of a kind: 0.50 - 0.75
        #   Straight/Flush:  0.25 - 0.48
        #   Full house:      0.65 - 0.90
        #   Four of a kind:  0.75 - 0.95

        if strength >= 0.60:
            # Three of a kind or better — very aggressive
            for action in [ALL_IN, RAISE_FULL, RAISE_HALF, CALL]:
                if action in legal_actions:
                    return action

        elif strength >= 0.35:
            # Two pair, pocket pair, or strong draw
            # If behind on chips, push harder
            if chip_pressure > 0.6:
                for action in [RAISE_FULL, RAISE_HALF, CALL]:
                    if action in legal_actions:
                        return action
            else:
                for action in [RAISE_HALF, CALL]:
                    if action in legal_actions:
                        return action

        elif strength >= 0.10:
            # One pair, high card, or weak draw
            # Almost all preflop hands land here — always at least call
            for action in [CALL]:
                if action in legal_actions:
                    return action

        else:
            # Genuinely terrible hand — fold
            for action in [FOLD, CALL]:
                if action in legal_actions:
                    return action

        return random.choice(legal_actions)

    def estimate_hand_strength(self, observation):
        if isinstance(observation, dict):
            observation = observation.get("observation", observation)

        card_indices = np.where(np.array(observation[:52]) == 1)[0]

        if len(card_indices) == 0:
            return 0.5

        # Correct rank mapping per PettingZoo documentation:
        # 0=Ace, 1=2, 2=3 ... 9=10, 10=Jack, 11=Queen, 12=King
        # Ace needs special high value treatment
        raw_ranks = [int(idx % 13) for idx in card_indices]
        suits     = [int(idx // 13) for idx in card_indices]

        # Convert to meaningful rank values where Ace is highest
        # 0(Ace)=14, 1(2)=2, 2(3)=3 ... 12(King)=13
        def rank_value(r):
            return 14 if r == 0 else r + 1

        rank_values = [rank_value(r) for r in raw_ranks]

        strength = 0.0

        # High card contribution — Ace(14) is strongest
        max_rank = max(rank_values)
        strength += ((max_rank - 2) / 12.0) * 0.20

        # Count occurrences of each rank for made hand detection
        rank_counts = {}
        for r in rank_values:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        max_count = max(rank_counts.values())
        pairs     = sum(1 for count in rank_counts.values() if count >= 2)

        # Made hand bonuses
        if max_count == 4:
            strength += 0.75   # four of a kind
        elif max_count == 3 and pairs >= 2:
            strength += 0.65   # full house
        elif max_count == 3:
            strength += 0.50   # three of a kind
        elif max_count == 2 and pairs >= 2:
            strength += 0.35   # two pair
        elif max_count == 2:
            strength += 0.20   # one pair

        # Flush detection — 5+ cards of same suit
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suit_count = max(suit_counts.values())

        if max_suit_count >= 5:
            strength += 0.25   # flush
        elif max_suit_count >= 3:
            strength += 0.03   # flush draw

        # Straight detection
        sorted_ranks = sorted(set(rank_values))

        # Handle Ace-low straight (A-2-3-4-5)
        if 14 in sorted_ranks:
            sorted_ranks = [1] + sorted_ranks  # add low Ace

        consecutive     = 1
        max_consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i-1] + 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1

        if max_consecutive >= 5:
            strength += 0.25   # straight
        elif max_consecutive >= 4:
            strength += 0.03   # straight draw

        return min(1.0, float(strength))

    def _pot_odds(self, observation):
        """Separate helper for pot odds — used in act(), not in strength."""
        pot_size = observation[52]
        call_amt = observation[53]
        if pot_size > 0:
            return call_amt / (pot_size + call_amt + 1e-6)
        return 0.0
    
    def reset(self):
        pass

    def record_action(self, action):
        pass