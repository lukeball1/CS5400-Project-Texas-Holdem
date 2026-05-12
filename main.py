import argparse
from game.environment import PokerEnvironment
from agents.mccfr_agent import MCCFRAgent
from agents.heuristic_agent import HeuristicAgent
from agents.mcts_agent import MCTS_Agent
from simulation.runner import Runner
from simulation.logger import Logger

# --- Default configuration ---
DEFAULT_TRAIN_ITERATIONS = 500000
DEFAULT_NUM_GAMES = 10000
DEFAULT_STRATEGY_PATH = "data/strategy.pkl"
NUM_ACTIONS = 5  # 0=fold, 1=call, 2=raise-half, 3=raise-full, 4=raise-all-in
ACTION_LABELS = {
    0: "fold",
    1: "call",
    2: "raise_half_pot",
    3: "raise_full_pot",
    4: "all_in"
}

def train(args):
    """
    Train the MCCFR agent and save the strategy to disk.
    Run this once before any matchups.
    """
    print(f"Training MCCFR agent for {args.iterations} iterations...")
    env = PokerEnvironment()
    agent = MCCFRAgent()
    agent.train(env, iterations=args.iterations, save_path=args.strategy_path)
    print("Training complete.")

def run_matchup(agent1, agent2, num_games, log):
    """
    Run a matchup between two agents and log the results.
    """
    logger = Logger() if log else None
    runner = Runner(agent1, agent2, logger)
    results = runner.run(num_games=num_games)
    return results

def main(args):
    """
    Entry point for running matchups based on command line arguments.
    Supports three matchup modes:
        - mccfr_vs_heuristic
        - mccfr_vs_mcts
        - heuristic_vs_mcts
    """

    # --- Build agents ---
    # heuristic = HeuristicAgent()
    # mcts     = MCTS_Agent()

    matchups = {
        "mccfr_vs_heuristic": (MCCFRAgent(strategy_path=args.strategy_path, name="MCCFRAgent"), HeuristicAgent()),
        "mccfr_vs_mcts":      (MCCFRAgent(strategy_path=args.strategy_path, name="MCCFRAgent"), 
                                MCTS_Agent()),
        "heuristic_vs_mcts":  (HeuristicAgent(), MCTS_Agent()),
        "mccfr_vs_mccfr":     (MCCFRAgent(strategy_path=args.strategy_path, name="MCCFRAgent_1"), 
                                MCCFRAgent(strategy_path=args.strategy_path, name="MCCFRAgent_2"))
    }

    if args.matchup not in matchups:
        print(f"Unknown matchup '{args.matchup}'. Choose from: {list(matchups.keys())}")
        return

    agent1, agent2 = matchups[args.matchup]
    print(f"\nRunning {args.num_games} games: {agent1.name} vs {agent2.name}")
    run_matchup(agent1, agent2, args.num_games, args.log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Texas Hold'em AI - CS5400 Project"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train subcommand ---
    # Usage: python main.py train
    # Usage: python main.py train --iterations 100000
    # Usage: python main.py train --strategy_path data/my_strategy.pkl
    train_parser = subparsers.add_parser("train", help="Train the MCCFR agent")
    train_parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_TRAIN_ITERATIONS,
        help=f"Number of MCCFR training iterations (default: {DEFAULT_TRAIN_ITERATIONS})"
    )
    train_parser.add_argument(
        "--strategy_path",
        type=str,
        default=DEFAULT_STRATEGY_PATH,
        help=f"Where to save the trained strategy (default: {DEFAULT_STRATEGY_PATH})"
    )

    # --- Run subcommand ---
    # Usage: python main.py run --matchup mccfr_vs_heuristic
    # Usage: python main.py run --matchup mccfr_vs_mcts --num_games 500
    # Usage: python main.py run --matchup heuristic_vs_mcts --no_log
    run_parser = subparsers.add_parser("run", help="Run a matchup between two agents")
    run_parser.add_argument(
        "--matchup",
        type=str,
        default="mccfr_vs_heuristic",
        choices=["mccfr_vs_heuristic", "mccfr_vs_mcts", "heuristic_vs_mcts", "mccfr_vs_mccfr"],
        help="Which agents to pit against each other"
    )
    run_parser.add_argument(
        "--num_games",
        type=int,
        default=DEFAULT_NUM_GAMES,
        help=f"Number of games to simulate (default: {DEFAULT_NUM_GAMES})"
    )
    run_parser.add_argument(
        "--strategy_path",
        type=str,
        default=DEFAULT_STRATEGY_PATH,
        help=f"Path to the trained MCCFR strategy (default: {DEFAULT_STRATEGY_PATH})"
    )
    run_parser.add_argument(
        "--no_log",
        action="store_false",
        dest="log",
        default=True,
        help="Disable logging to file"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "run":
        main(args)