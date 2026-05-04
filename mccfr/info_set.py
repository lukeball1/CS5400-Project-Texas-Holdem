#basic class for an information set in MCCFR for Texas Hold'em. 
class InfoSet:
    
    def __init__(self, hole_cards, community_cards, action_history, round_name):
        self.hole_cards = tuple(sorted(hole_cards))
        self.community_cards = tuple(community_cards)
        self.action_history = tuple(action_history)
        self.round_name = round_name

    def key(self):
        """
        Produces a hashable string key representing this information set.
        Two game states that look identical to the player will produce
        the same key, allowing the agent to reuse learned strategies.
        """
        return str((
            self.hole_cards,
            self.community_cards,
            self.action_history,
            self.round_name
        ))

    def __str__(self):
        return self.key()