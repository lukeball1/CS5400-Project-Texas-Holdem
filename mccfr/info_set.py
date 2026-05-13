# In info_set.py — rename fields accordingly
class InfoSet:
    def __init__(self, hole_cards, community_cards,
                  action_history, round_name,
                  my_chips=0.0, opponent_chips=0.0):
        self.hole_cards       = tuple(sorted(hole_cards))
        self.community_cards  = tuple(community_cards)
        self.action_history   = tuple(action_history)
        self.round_name       = round_name

        # Bucket chip counts into ranges to limit key explosion
        # 0-25 chips, 25-50, 50-75, 75-100
        self.my_chip_bucket   = (int(my_chips) // 25) * 25
        self.opp_chip_bucket  = (int(opponent_chips) // 25) * 25

    def key(self):
        return str((
            self.hole_cards,
            self.community_cards,
            self.action_history,
            self.round_name,
            self.my_chip_bucket,
            self.opp_chip_bucket
        ))

    def __str__(self):
        return self.key()