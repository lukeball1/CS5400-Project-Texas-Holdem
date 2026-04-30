from pettingzoo.classic import texas_holdem_no_limit_v6

class PokerEnvironment:

    def __init__(self, num_players=2, render_mode=None):
        self.env = texas_holdem_no_limit_v6.env(
            num_players=num_players,
            render_mode=render_mode
        )

    def reset(self):
        """Reset the environment at the start of a new game."""
        self.env.reset()

    def step(self, action):
        """Take an action for the current agent."""
        self.env.step(action)

    def observe(self, agent):
        """Get the current observation and action mask for a given agent."""
        obs = self.env.observe(agent)
        return obs['observation'], obs['action_mask']

    def current_agent(self):
        """Returns whose turn it is."""
        return self.env.agent_selection

    def is_done(self):
        """Returns True if the game is over."""
        return all(self.env.terminations.values()) or \
                all(self.env.truncations.values())

    def last(self):
        """
        PettingZoo's built-in method — returns
        (observation, reward, termination, truncation, info)
        for the current agent.
        """
        return self.env.last()

    def get_agents(self):
        """Returns the list of active agent name strings."""
        return self.env.agents

    def close(self):
        self.env.close()