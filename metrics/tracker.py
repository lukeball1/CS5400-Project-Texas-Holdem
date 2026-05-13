import pandas as pd
import numpy as np
import os

ACTION_LABELS = {
    0: "fold",
    1: "call",
    2: "raise_half_pot",
    3: "raise_full_pot",
    4: "all_in"
}

class Tracker:

    def __init__(self, log_dir="data/logs"):
        self.log_dir = log_dir
        self.games_df = None
        self.actions_df = None

    def load_latest(self):
        """
        Automatically find and load the most recent log files
        from the log directory based on timestamp in filename.
        """
        all_files = os.listdir(self.log_dir)

        game_files = sorted([f for f in all_files if f.endswith("_games.csv")])
        action_files = sorted([f for f in all_files if f.endswith("_actions.csv")])

        if not game_files or not action_files:
            raise FileNotFoundError("No log files found in data/logs/")

        # Most recent file is last after sorting by timestamp prefix
        latest_games   = os.path.join(self.log_dir, game_files[-1])
        latest_actions = os.path.join(self.log_dir, action_files[-1])

        self.games_df   = pd.read_csv(latest_games)
        self.actions_df = pd.read_csv(latest_actions)

        print(f"Loaded: {game_files[-1]}")
        print(f"Loaded: {action_files[-1]}")

    def load_specific(self, run_id):
        """Load log files for a specific run ID."""
        games_path   = os.path.join(self.log_dir, f"{run_id}_games.csv")
        actions_path = os.path.join(self.log_dir, f"{run_id}_actions.csv")

        self.games_df   = pd.read_csv(games_path)
        self.actions_df = pd.read_csv(actions_path)

    # ---------------------------------------------------------------
    # Win rate metrics
    # ---------------------------------------------------------------

    def win_rates(self):
        """
        Overall win rate per agent.
        Returns a dict of {agent_name: win_rate_float}
        """
        results = {}
        for agent in self.games_df["agent_name"].unique():
            agent_df = self.games_df[self.games_df["agent_name"] == agent]
            results[agent] = agent_df["won"].astype(int).mean()
        return results

    def rolling_win_rate(self, agent_name, window=100):
        """
        Win rate over time using a rolling window.
        Useful for showing convergence behavior.
        Returns a DataFrame with columns [game_num, rolling_win_rate]
        """
        agent_df = self.games_df[self.games_df["agent_name"] == agent_name].copy()
        agent_df = agent_df.sort_values("game_num").reset_index(drop=True)
        agent_df["rolling_win_rate"] = (
            agent_df["won"]
            .astype(int)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        return agent_df[["game_num", "rolling_win_rate"]]

    def cumulative_win_rate(self, agent_name):
        """
        Cumulative win rate from game 1 to game N.
        Shows long-run convergence.
        """
        agent_df = self.games_df[self.games_df["agent_name"] == agent_name].copy()
        agent_df = agent_df.sort_values("game_num").reset_index(drop=True)
        agent_df["cumulative_win_rate"] = (
            agent_df["won"]
            .astype(int)
            .expanding()
            .mean()
        )
        return agent_df[["game_num", "cumulative_win_rate"]]

    # ---------------------------------------------------------------
    # Chip metrics
    # ---------------------------------------------------------------

    def average_chip_delta(self):
        """
        Average chip gain/loss per game per agent.
        Returns a dict of {agent_name: avg_chip_delta}
        """
        results = {}
        for agent in self.games_df["agent_name"].unique():
            agent_df = self.games_df[self.games_df["agent_name"] == agent]
            results[agent] = agent_df["chip_delta"].mean()
        return results

    def cumulative_chips(self, agent_name):
        """
        Cumulative chip count over all games.
        Shows bankroll growth/decline over time.
        Returns a DataFrame with columns [game_num, cumulative_chips]
        """
        agent_df = self.games_df[self.games_df["agent_name"] == agent_name].copy()
        agent_df = agent_df.sort_values("game_num").reset_index(drop=True)
        agent_df["cumulative_chips"] = agent_df["chip_delta"].cumsum()
        return agent_df[["game_num", "cumulative_chips"]]

    def chip_variance(self):
        """
        Variance in chip delta per agent.
        High variance = unstable/risky strategy.
        Low variance = consistent strategy.
        Returns a dict of {agent_name: variance}
        """
        results = {}
        for agent in self.games_df["agent_name"].unique():
            agent_df = self.games_df[self.games_df["agent_name"] == agent]
            results[agent] = agent_df["chip_delta"].var()
        return results

    def chip_distribution(self, agent_name):
        """
        Returns raw chip deltas for an agent — useful for
        plotting histograms to visualize outcome distribution.
        """
        agent_df = self.games_df[self.games_df["agent_name"] == agent_name]
        return agent_df["chip_delta"].values

    # ---------------------------------------------------------------
    # Action distribution metrics
    # ---------------------------------------------------------------

    def action_distribution(self):
        """
        Fold/call/raise distribution per agent as percentages.
        Returns a dict of {agent_name: {action_label: percentage}}
        """
        results = {}
        for agent in self.actions_df["agent_name"].unique():
            agent_df = self.actions_df[self.actions_df["agent_name"] == agent]
            total = len(agent_df)
            dist = {}
            for action_id, label in ACTION_LABELS.items():
                count = len(agent_df[agent_df["action"] == action_id])
                dist[label] = round((count / total) * 100, 2) if total > 0 else 0.0
            results[agent] = dist
        return results

    def action_distribution_by_round(self):
        """
        Action distribution broken down by round (preflop/flop/turn/river)
        if round data is available in the action log.
        Returns a nested dict of {agent: {round: {action: pct}}}
        """
        if "round" not in self.actions_df.columns:
            print("Round data not available in action log.")
            return {}

        results = {}
        for agent in self.actions_df["agent_name"].unique():
            agent_df = self.actions_df[self.actions_df["agent_name"] == agent]
            results[agent] = {}
            for round_name in agent_df["round"].unique():
                round_df = agent_df[agent_df["round"] == round_name]
                total = len(round_df)
                dist = {}
                for action_id, label in ACTION_LABELS.items():
                    count = len(round_df[round_df["action"] == action_id])
                    dist[label] = round((count / total) * 100, 2) if total > 0 else 0.0
                results[agent][round_name] = dist
        return results

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    def full_summary(self):
        """
        Compile all metrics into a single summary dictionary.
        Used by reporter.py to generate the final report.
        """
        return {
            "win_rates":         self.win_rates(),
            "avg_chip_delta":    self.average_chip_delta(),
            "chip_variance":     self.chip_variance(),
            "action_distribution": self.action_distribution(),
        }

    def print_summary(self):
        """Print a readable summary to the console."""
        summary = self.full_summary()
        num_games = self.games_df["game_num"].nunique()

        print(f"\n{'='*50}")
        print(f"  MATCH SUMMARY — {num_games} games")
        print(f"{'='*50}")

        print("\n--- Win Rates ---")
        for agent, rate in summary["win_rates"].items():
            print(f"  {agent}: {rate * 100:.1f}%")

        print("\n--- Avg Chip Delta Per Game ---")
        for agent, delta in summary["avg_chip_delta"].items():
            print(f"  {agent}: {delta:+.1f} chips")

        print("\n--- Chip Variance (strategy stability) ---")
        for agent, var in summary["chip_variance"].items():
            print(f"  {agent}: {var:.1f}")

        print("\n--- Action Distribution ---")
        for agent, dist in summary["action_distribution"].items():
            print(f"  {agent}:")
            for action, pct in dist.items():
                print(f"    {action}: {pct}%")
        print(f"{'='*50}\n")