"""Quick ablation test: does the shared field help agents?

Loads trained checkpoint and compares performance across three conditions:
  - normal: field active as trained
  - zeroed: field wiped to zeros each step
  - random: field replaced with noise each step

If the field encodes useful collective knowledge, normal > zeroed.
"""

import pickle

from src.agents.network import ActorCritic
from src.analysis.ablation import ablation_test, print_ablation_results
from src.configs import Config


def main():
    # Load config matching the training run
    config = Config.from_yaml("configs/phase2.yaml")
    config.evolution.enabled = False  # disable evolution for faster ablation

    # Load trained parameters
    with open("checkpoints/params.pkl", "rb") as f:
        params = pickle.load(f)

    network = ActorCritic(
        hidden_dims=tuple(config.agent.hidden_dims),
        num_actions=6,
    )

    print("Running ablation test (5 episodes per condition, evolution disabled)...")
    print()

    results = ablation_test(network, params, config, num_episodes=5, seed=42)
    print_ablation_results(results)

    # Verdict
    if "normal" in results and "zeroed" in results:
        normal = results["normal"].mean_reward
        zeroed = results["zeroed"].mean_reward
        print()
        if normal > zeroed:
            print(f"RESULT: Field HELPS — Normal ({normal:.2f}) > Zeroed ({zeroed:.2f})")
        elif normal < zeroed:
            print(f"RESULT: Field HURTS — Normal ({normal:.2f}) < Zeroed ({zeroed:.2f})")
        else:
            print(f"RESULT: No difference — Normal ({normal:.2f}) == Zeroed ({zeroed:.2f})")


if __name__ == "__main__":
    main()
