from game.environment import PokerEnvironment
from agents.base_agent import BaseAgent
from simulation.logger import Logger

class Runner:
    def __init__(self, agent1: BaseAgent, agent2: BaseAgent, logger: Logger = True):
        self.env = PokerEnvironment()
        self.agents = {}
        self.logger = logger
        self.agent1 = agent1
        self.agent2 = agent2
        
    def run(self, num_games: int):
        """
        Run num_games matchups between agent1 and agent2.
        Returns a results dictionary with high level stats.
        """
        results = {
            self.agent1.name: {"wins": 0, "total_chips": 0},
            self.agent2.name: {"wins": 0, "total_chips": 0}
        }

        for game_num in range(num_games):
            outcome = self._run_single_game(game_num)

            for agent_name, stats in outcome.items():
                results[agent_name]["total_chips"] += stats["chip_delta"]
                if stats["won"]:
                    results[agent_name]["wins"] += 1

        self._print_summary(results, num_games)
        return results

    def _run_single_game(self, game_num: int):
        """
        Run a single hand of poker.
        Returns chip deltas and win/loss outcome per agent.
        """
        self.env.reset()

        # Map PettingZoo agent strings (e.g. "player_0", "player_1")
        # to our agent objects
        pz_agents = self.env.get_agents()
        self.agents = {
            pz_agents[0]: self.agent1,
            pz_agents[1]: self.agent2
        }

        # Reset all agents at the start of each hand
        for agent in self.agents.values():
            agent.reset()

        if self.logger:
            self.logger.log_game_start(game_num, self.agents)

        # Track starting chip counts to compute deltas at the end
        starting_chips = self._get_chip_counts()

        # --- Main game loop ---
        while not self.env.is_done():
            pz_agent_name = self.env.current_agent()
            agent = self.agents[pz_agent_name]

            obs, action_mask = self.env.observe(pz_agent_name)
            action = agent.act(obs, action_mask)

            # Inform all agents of the action taken so they
            # can update their internal action histories
            for a in self.agents.values():
                a.record_action(action)

            if self.logger:
                self.logger.log_action(game_num, agent.name, action)

            self.env.step(action)

        # --- Game over ---
        ending_chips = self._get_chip_counts()
        outcome = self._compute_outcome(starting_chips, ending_chips)

        if self.logger:
            self.logger.log_game_end(game_num, outcome)

        return outcome

    def _get_chip_counts(self):
        """
        Retrieve the current chip count for each agent.
        Uses the rewards accumulation from PettingZoo.
        """
        chip_counts = {}
        for pz_name, agent in self.agents.items():
            # Access rewards directly from the environment
            reward = self.env.env.rewards.get(pz_name, 0)
            chip_counts[agent.name] = reward
        return chip_counts

    def _compute_outcome(self, starting_chips, ending_chips):
        """
        Compute chip deltas and determine winner.
        """
        outcome = {}
        for agent_name in starting_chips:
            delta = ending_chips[agent_name] - starting_chips[agent_name]
            outcome[agent_name] = {
                "chip_delta": delta,
                "won": delta > 0
            }
        return outcome

    def _print_summary(self, results, num_games):
        print(f"\n--- Results after {num_games} games ---")
        for agent_name, stats in results.items():
            win_rate = stats["wins"] / num_games * 100
            avg_chips = stats["total_chips"] / num_games
            print(f"{agent_name}: {win_rate:.1f}% win rate | "
                    f"{avg_chips:.1f} avg chip delta per game")