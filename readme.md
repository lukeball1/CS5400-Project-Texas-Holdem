# Train the MCCFR agent with default settings
python main.py train

# Train with a custom iteration count
python main.py train --iterations 100000

# Run MCCFR vs Heuristic for 1000 games
python main.py run --matchup mccfr_vs_heuristic

# Run MCCFR vs MCTS for 500 games without logging
python main.py run --matchup mccfr_vs_mcts --num_games 500 --no_log

# Run Heuristic vs MCTS
python main.py run --matchup heuristic_vs_mcts