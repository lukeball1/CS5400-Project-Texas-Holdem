import os
import csv
import json
from datetime import datetime

class Logger:

    def __init__(self, log_dir="data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique run ID based on timestamp so logs
        # don't overwrite each other between runs
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # We maintain two separate logs:
        # 1. A per-action log for granular decision tracking
        # 2. A per-game log for high level outcome tracking
        self.action_log_path = os.path.join(log_dir, f"{self.run_id}_actions.csv")
        self.game_log_path = os.path.join(log_dir, f"{self.run_id}_games.csv")

        self._init_action_log()
        self._init_game_log()

    # --- Initialization ---

    def _init_action_log(self):
        """Create the action log CSV with headers."""
        with open(self.action_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_num",
                "agent_name",
                "action",        # 0=fold, 1=call, 2=raise
                "action_label",  # human readable
            ])

    def _init_game_log(self):
        """Create the game log CSV with headers."""
        with open(self.game_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_num",
                "agent_name",
                "chip_delta",
                "won",           # True/False
            ])

    # --- Logging methods called by runner.py ---

    def log_game_start(self, game_num, agents):
        """
        Called at the start of each hand.
        Agents is the dict mapping pz_name -> BaseAgent.
        """
        agent_names = [a.name for a in agents.values()]
        print(f"[Game {game_num}] Starting: {agent_names[0]} vs {agent_names[1]}")

    def log_action(self, game_num, agent_name, action):
        """
        Called after every decision made during a hand.
        Appends a row to the action log CSV.
        """
        action_labels = {0: "fold", 1: "call", 2: "raise"}
        with open(self.action_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                game_num,
                agent_name,
                action,
                action_labels.get(action, "unknown")
            ])

    def log_game_end(self, game_num, outcome):
        """
        Called at the end of each hand.
        Outcome is the dict returned by runner._compute_outcome().
        Appends a row per agent to the game log CSV.
        """
        with open(self.game_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for agent_name, stats in outcome.items():
                writer.writerow([
                    game_num,
                    agent_name,
                    stats["chip_delta"],
                    stats["won"]
                ])

    def log_summary(self, results, num_games):
        """
        Called at the end of a full run.
        Writes a human readable JSON summary file.
        """
        summary = {
            "run_id": self.run_id,
            "num_games": num_games,
            "agents": {}
        }

        for agent_name, stats in results.items():
            summary["agents"][agent_name] = {
                "wins": stats["wins"],
                "losses": num_games - stats["wins"],
                "win_rate": round(stats["wins"] / num_games * 100, 2),
                "total_chip_delta": stats["total_chips"],
                "avg_chip_delta_per_game": round(stats["total_chips"] / num_games, 2)
            }

        summary_path = os.path.join(
            self.log_dir, f"{self.run_id}_summary.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary written to {summary_path}")