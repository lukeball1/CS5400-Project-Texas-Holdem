from pettingzoo.classic import texas_holdem_v4
from agents.heuristic_agent import HeuristicAgent

env = texas_holdem_v4.env(render_mode=None)
env.reset()

agent = HeuristicAgent()
# Sample loop to run one episode with the heuristic agent for testing
for agent_name in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()

    print("OBS:", obs)  

    if termination or truncation:
        action = None
    else:
        action = agent.act(obs["observation"], obs["action_mask"])

    env.step(action)