import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from metrics.tracker import Tracker

# Color palette — one color per agent for consistency across all charts
AGENT_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

class Reporter:

    def __init__(self, tracker: Tracker, output_dir="data/reports"):
        self.tracker = tracker
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_all(self):
        """
        Generate all charts and the summary report.
        Call this once after loading log data in tracker.
        """
        print("\nGenerating reports...")
        self.plot_cumulative_win_rate()
        self.plot_rolling_win_rate()
        self.plot_cumulative_chips()
        self.plot_chip_distribution()
        self.plot_action_distribution()
        self.plot_win_rate_bar()
        self.plot_chip_variance()
        self.save_summary_json()
        self.tracker.print_summary()
        print(f"All reports saved to {self.output_dir}/")

    # ---------------------------------------------------------------
    # Win rate charts
    # ---------------------------------------------------------------

    def plot_cumulative_win_rate(self):
        """
        Cumulative win rate from game 1 to N per agent.
        Shows long run convergence toward true win rate.
        Most useful chart for demonstrating variance washout.
        """
        agents = self.tracker.games_df["agent_name"].unique()
        plt.figure(figsize=(12, 6))

        for i, agent in enumerate(agents):
            df = self.tracker.cumulative_win_rate(agent)
            plt.plot(
                df["game_num"],
                df["cumulative_win_rate"],
                label=agent,
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                linewidth=2
            )

        plt.axhline(y=0.5, color="gray", linestyle="--",
                    linewidth=1, label="50% baseline")
        plt.xlabel("Number of Games")
        plt.ylabel("Cumulative Win Rate")
        plt.title("Cumulative Win Rate Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("cumulative_win_rate.png")

    def plot_rolling_win_rate(self, window=100):
        """
        Rolling win rate over a sliding window of games.
        Smooths out short-term variance while showing trends.
        """
        agents = self.tracker.games_df["agent_name"].unique()
        plt.figure(figsize=(12, 6))

        for i, agent in enumerate(agents):
            df = self.tracker.rolling_win_rate(agent, window=window)
            plt.plot(
                df["game_num"],
                df["rolling_win_rate"],
                label=agent,
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                linewidth=2
            )

        plt.axhline(y=0.5, color="gray", linestyle="--",
                    linewidth=1, label="50% baseline")
        plt.xlabel("Number of Games")
        plt.ylabel(f"Win Rate (rolling {window}-game window)")
        plt.title(f"Rolling Win Rate ({window}-game window)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("rolling_win_rate.png")

    def plot_win_rate_bar(self):
        """
        Simple bar chart of final overall win rate per agent.
        Good for at-a-glance comparison in the report.
        """
        win_rates = self.tracker.win_rates()
        agents = list(win_rates.keys())
        rates  = [win_rates[a] * 100 for a in agents]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(
            agents, rates,
            color=AGENT_COLORS[:len(agents)],
            edgecolor="white",
            width=0.5
        )

        # Label each bar with its percentage
        for bar, rate in zip(bars, rates):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rate:.1f}%",
                ha="center", va="bottom",
                fontweight="bold"
            )

        plt.axhline(y=50, color="gray", linestyle="--",
                    linewidth=1, label="50% baseline")
        plt.ylabel("Win Rate (%)")
        plt.title("Overall Win Rate by Agent")
        plt.ylim(0, 110)
        plt.legend()
        plt.tight_layout()
        self._save("win_rate_bar.png")

    # ---------------------------------------------------------------
    # Chip charts
    # ---------------------------------------------------------------

    def plot_cumulative_chips(self):
        """
        Cumulative chip gain/loss over all games per agent.
        Shows bankroll growth or decline over time.
        Upward slope = profitable strategy.
        """
        agents = self.tracker.games_df["agent_name"].unique()
        plt.figure(figsize=(12, 6))

        for i, agent in enumerate(agents):
            df = self.tracker.cumulative_chips(agent)
            plt.plot(
                df["game_num"],
                df["cumulative_chips"],
                label=agent,
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                linewidth=2
            )

        plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Number of Games")
        plt.ylabel("Cumulative Chip Delta")
        plt.title("Cumulative Chip Count Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("cumulative_chips.png")

    def plot_chip_distribution(self):
        """
        Histogram of chip deltas per game per agent.
        Shows the shape of outcome distribution —
        narrow = consistent, wide = high variance strategy.
        """
        agents = self.tracker.games_df["agent_name"].unique()
        plt.figure(figsize=(12, 6))

        for i, agent in enumerate(agents):
            deltas = self.tracker.chip_distribution(agent)
            plt.hist(
                deltas,
                bins=40,
                alpha=0.6,
                label=agent,
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                edgecolor="white"
            )

        plt.axvline(x=0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Chip Delta Per Game")
        plt.ylabel("Frequency")
        plt.title("Distribution of Chip Outcomes Per Game")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("chip_distribution.png")

    def plot_chip_variance(self):
        """
        Bar chart of chip delta variance per agent.
        High variance = unpredictable/risky strategy.
        Low variance = stable/consistent strategy.
        """
        variances = self.tracker.chip_variance()
        agents = list(variances.keys())
        values = [variances[a] for a in agents]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(
            agents, values,
            color=AGENT_COLORS[:len(agents)],
            edgecolor="white",
            width=0.5
        )

        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}",
                ha="center", va="bottom",
                fontweight="bold"
            )

        plt.ylabel("Chip Delta Variance")
        plt.title("Strategy Stability — Chip Variance Per Agent")
        plt.tight_layout()
        self._save("chip_variance.png")

    # ---------------------------------------------------------------
    # Action distribution charts
    # ---------------------------------------------------------------

    def plot_action_distribution(self):
        """
        Grouped bar chart of fold/call/raise distribution per agent.
        Reveals each agent's playing style —
        passive (lots of calls), aggressive (lots of raises),
        or conservative (lots of folds).
        """
        dist = self.tracker.action_distribution()
        agents  = list(dist.keys())
        actions = list(dist[agents[0]].keys())

        x = np.arange(len(actions))
        width = 0.8 / len(agents)

        plt.figure(figsize=(12, 6))

        for i, agent in enumerate(agents):
            values = [dist[agent][a] for a in actions]
            offset = (i - len(agents) / 2 + 0.5) * width
            bars = plt.bar(
                x + offset, values,
                width=width,
                label=agent,
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                edgecolor="white"
            )
            for bar, val in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8
                )

        plt.xticks(x, actions, rotation=15)
        plt.ylabel("Percentage of Actions (%)")
        plt.title("Action Distribution by Agent")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        self._save("action_distribution.png")

    # ---------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------

    def save_summary_json(self):
        """
        Save a full JSON summary of all metrics.
        Complements the per-run summary already written by logger.py
        with the additional computed metrics from tracker.py.
        """
        summary = self.tracker.full_summary()
        num_games = self.tracker.games_df["game_num"].nunique()

        report = {
            "num_games": num_games,
            "win_rates": {
                k: round(v * 100, 2)
                for k, v in summary["win_rates"].items()
            },
            "avg_chip_delta": {
                k: round(v, 2)
                for k, v in summary["avg_chip_delta"].items()
            },
            "chip_variance": {
                k: round(v, 2)
                for k, v in summary["chip_variance"].items()
            },
            "action_distribution": summary["action_distribution"]
        }

        path = os.path.join(self.output_dir, "full_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Summary JSON saved to {path}")

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _save(self, filename):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")