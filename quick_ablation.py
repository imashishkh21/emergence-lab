"""Quick ablation test - minimal version."""
import jax
import jax.numpy as jnp
from src.configs import Config
from src.agents.network import ActorCritic
from src.agents.policy import get_deterministic_actions
from src.environment.env import reset, step
from src.environment.obs import get_observations, obs_dim
from src.field.field import create_field, FieldState
import pickle

def run_episode(network, params, config, key, field_mode="normal"):
    """Run single episode. field_mode: normal/zeroed/random"""
    state = reset(key, config)
    total_reward = 0.0
    
    for t in range(config.env.max_steps):
        # Modify field based on mode
        if field_mode == "zeroed":
            zero_field = create_field(config.env.grid_size, config.env.grid_size, config.field.num_channels)
            state = state.replace(field_state=zero_field)
        elif field_mode == "random":
            key, noise_key = jax.random.split(key)
            noise = jax.random.uniform(noise_key, (config.env.grid_size, config.env.grid_size, config.field.num_channels))
            state = state.replace(field_state=FieldState(values=noise))
        
        obs = get_observations(state, config)
        obs_batched = obs[None, :, :]
        actions = get_deterministic_actions(network, params, obs_batched)[0]
        
        # Only take actions for alive agents
        num_agents = config.env.num_agents
        actions = actions[:num_agents]
        
        state, rewards, done, info = step(state, actions, config)
        total_reward += float(jnp.sum(rewards))
        
        if bool(done):
            break
    
    return total_reward

# Load
config = Config()
config.evolution.enabled = False  # Disable for speed

with open("checkpoints/params.pkl", "rb") as f:
    params = pickle.load(f)

network = ActorCritic(hidden_dims=(64, 64), num_actions=config.agent.num_actions)

# Test each condition
print("Running quick ablation (3 episodes each)...")
conditions = ["normal", "zeroed", "random"]
results = {}

for cond in conditions:
    rewards = []
    for ep in range(3):
        key = jax.random.PRNGKey(ep)
        r = run_episode(network, params, config, key, cond)
        rewards.append(r)
    avg = sum(rewards) / len(rewards)
    results[cond] = avg
    print(f"  {cond}: {avg:.2f}")

print("\n" + "="*50)
print("RESULTS:")
print(f"  Normal:  {results['normal']:.2f}")
print(f"  Zeroed:  {results['zeroed']:.2f}")
print(f"  Random:  {results['random']:.2f}")
print(f"\n  Normal - Zeroed gap: {results['normal'] - results['zeroed']:+.2f}")
print(f"  Normal - Random gap: {results['normal'] - results['random']:+.2f}")
print("="*50)

if results['normal'] > results['zeroed']:
    print("\n✅ FIELD HELPS! Agents perform better with the field.")
else:
    print("\n❌ Field not helping yet - need more training.")
