"""Baseline methods for emergence comparison.

This module provides baseline implementations for comparing against the
field-mediated emergence system. All baselines return a standardized result dict:
    {
        "total_reward": float,
        "food_collected": float,
        "final_population": int,
        "per_agent_rewards": list[float],
    }

Baselines:
    - IPPO: Independent PPO with no field and no evolution (US-006)
    - ACO-Fixed: Hardcoded pheromone rules, no neural network (US-007)
    - ACO-Hybrid: Neural network for movement, hardcoded field writes (US-007)
    - MAPPO: Multi-Agent PPO with centralized critic (US-008)
"""
