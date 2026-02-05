from src.configs import Config
from src.analysis.ablation import ablation_test, print_ablation_results
from src.agents.network import ActorCritic
import pickle

config = Config()
config.evolution.enabled = False

with open("checkpoints/params.pkl", "rb") as f:
    params = pickle.load(f)

network = ActorCritic(hidden_dims=(64,64), num_actions=config.agent.num_actions)
results = ablation_test(network, params, config, num_episodes=5)
print_ablation_results(results)
