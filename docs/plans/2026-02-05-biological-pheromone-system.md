# Biological Pheromone System — Implementation Plan

**Date:** 2026-02-05
**Status:** Ready to build
**Research basis:** 103 curated sources in NotebookLM (ant biology, pheromone mechanics, MARL, self-organization, swarm intelligence). 5 rounds of targeted queries.

## Why This Exists

The shared stigmergy field HURTS agents in every experiment we've run. Field ON agents die in harsh environments while Field OFF agents thrive. Root causes identified:

1. **Observation drowning:** 484 field dims out of 502 total (96% noise)
2. **Constant screaming:** Every agent writes every step, flooding the field
3. **No return trip:** Agents eat food and wander; no trail geometry forms
4. **Density/decay mismatch:** 16 agents on 1600 cells, trails vanish before anyone finds them

This plan replaces the broken system with biologically-accurate ant pheromone mechanics.

## The Complete Specification

### Action Space: 5 actions (changed from 6)

| Action | ID | Effect |
|--------|----|--------|
| Stay | 0 | No movement |
| Up | 1 | Move up |
| Down | 2 | Move down |
| Left | 3 | Move left |
| Right | 4 | Move right |

**Reproduce action removed.** Reproduction is now automatic (see Nest Mechanics below).

### Observation Vector: 45 dimensions (down from 502)

| Component | Dims | Details |
|-----------|------|---------|
| Position (normalized to [-1,1]) | 2 | row, col mapped from [0, grid_size-1] to [-1, 1] |
| Energy (normalized to [0,1]) | 1 | agent_energy / max_energy, clipped |
| Has food flag | 1 | 1.0 if carrying food, 0.0 if not |
| Nest compass (path integration) | 2 | (dx_nest, dy_nest) normalized to [-1,1] by grid_size, with distance-dependent noise |
| Field spatial (4-dir + center x 4ch) | 20 | For each of 4 channels: field value at N, S, E, W neighbors + center |
| Field temporal (dC/dt x 4ch) | 4 | current_center - previous_center per channel |
| Nearest food (K=5) | 15 | (dx, dy, available) per food slot, same as current |
| **Total** | **45** | |

### Four Field Channels

| Channel | Name | Write trigger | Write value | Diffusion | Decay |
|---------|------|---------------|-------------|-----------|-------|
| 0 | Recruitment | ONLY while has_food=True (during laden agent's write step) | 1.0 (fixed) | 0.5 (wide spread) | 0.05 (fast fade) |
| 1 | Territory | Passive, every step, every agent | +0.01 (small accumulation) | 0.01 (stays local) | 0.0001 (near permanent) |
| 2 | Reserved | Locked to 0 in Phase 1 | N/A | N/A | N/A |
| 3 | Reserved | Locked to 0 in Phase 1 | N/A | N/A | N/A |

Field values capped to [0, 1] after every write operation.

### Nest Mechanics

**Nest area:** Configurable center and radius. Default: center of grid, radius=2 (5x5 area).

**Territory channel initialization:** At reset(), set Ch1 = 1.0 in the nest 5x5 area + 1-cell border (7x7 total). Rest of grid = 0.0 for Ch1. All other channels start at 0.0.

**Agent spawn:** All agents start at random positions WITHIN the 5x5 nest area (clustered spawn).

**Food delivery:** When agent with has_food=True steps into nest area:
- has_food -> False
- Agent receives 95% of food_energy as energy
- PPO reward signal for the delivery (this is the main reward)
- laden_cooldown resets to False

**Reproduction:** Automatic. Every step, check: if agent is inside nest area AND energy > reproduce_threshold AND an empty slot exists, trigger reproduction. Agent loses reproduce_cost energy. Child spawns in nest area with mutated weights. No explicit action needed.

### Food Carrying ("Scout Sip" Model)

**Food pickup:** When agent moves adjacent to uncollected food (Chebyshev distance <= 1):
- Food item marked as collected (disappears from grid)
- has_food -> True
- Agent receives 5% of food_energy immediately (survival sip)
- laden_cooldown -> False (first action after pickup is a move)

**Laden movement penalty (alternating move/write):**
- When has_food=True, agent alternates between moving and writing
- Step with laden_cooldown=False: agent moves normally, then laden_cooldown -> True
- Step with laden_cooldown=True: agent STAYS in place (movement ignored), writes Ch0 recruitment pheromone at current position, then laden_cooldown -> False
- Effectively halves laden agent speed
- Creates physical speed difference needed for lane formation emergence

**Agent dies while carrying:** Food is lost (already removed from grid). No special recovery.

### Nest Compass (Path Integration with Noise)

```
# True vector to nest center (normalized to [-1, 1])
true_dx = (nest_center_x - agent_x) / grid_size
true_dy = (nest_center_y - agent_y) / grid_size

# Distance from nest (in grid cells)
dist = sqrt((nest_center_x - agent_x)^2 + (nest_center_y - agent_y)^2)

# Noise scales linearly with distance (10% error rate)
noise_std = 0.10 * dist / grid_size  # normalize noise too
noisy_dx = true_dx + normal(0, noise_std)
noisy_dy = true_dy + normal(0, noise_std)

# Clip to [-1, 1]
compass = clip([noisy_dx, noisy_dy], -1, 1)
```

At nest (dist=0): compass is perfect [0, 0].
Far from nest: compass has significant noise, agent must rely on Ch1 territory gradient for final approach.

### Per-Channel Diffusion and Decay

Currently `dynamics.py` uses a single `diffusion_rate` and `decay_rate` for all channels. Must change to per-channel arrays.

```python
# In FieldConfig (new):
channel_diffusion_rates: tuple[float, ...] = (0.5, 0.01, 0.0, 0.0)
channel_decay_rates: tuple[float, ...] = (0.05, 0.0001, 0.0, 0.0)
```

In `dynamics.py`:
- `diffuse()` applies different diffusion per channel
- `decay()` applies different decay per channel
- Both operations must still be JIT-compatible (use array broadcasting, not Python loops)

## Files to Modify

### src/configs.py
- Add `NestConfig` dataclass:
  ```python
  @dataclass
  class NestConfig:
      radius: int = 2  # half-width, so 5x5 area
      food_sip_fraction: float = 0.05  # immediate energy on food pickup
      compass_noise_rate: float = 0.10  # path integration error rate
  ```
- Modify `FieldConfig`:
  - Add `channel_diffusion_rates: tuple[float, ...] = (0.5, 0.01, 0.0, 0.0)`
  - Add `channel_decay_rates: tuple[float, ...] = (0.05, 0.0001, 0.0, 0.0)`
  - Keep existing `diffusion_rate` and `decay_rate` as fallbacks for backward compatibility
  - Add `field_value_cap: float = 1.0`
- Modify `EvolutionConfig`:
  - Document that reproduction now requires nest area (no config change needed, logic change in env.py)
- Add `NestConfig` to master `Config`:
  ```python
  nest: NestConfig = dataclass_field(default_factory=NestConfig)
  ```
- Update `from_yaml` and `to_yaml` to handle NestConfig

### src/environment/state.py
Add new fields to `EnvState`:
```python
has_food: jnp.ndarray          # (max_agents,) bool - carrying food
prev_field_at_pos: jnp.ndarray # (max_agents, num_channels) float32 - for temporal derivative
laden_cooldown: jnp.ndarray    # (max_agents,) bool - move/write alternation toggle
```

Note: nest_center and nest_radius come from config, not state (they're static).

### src/environment/env.py
This is the biggest change. The step() function needs restructuring:

1. **Remove reproduce action (action=5).** Action space is now 0-4.
2. **Movement phase:** Apply movement as before, BUT if agent has_food=True and laden_cooldown=True, override movement to STAY.
3. **Food pickup phase:** When agent is adjacent to uncollected food:
   - Mark food collected
   - Set has_food=True
   - Give 5% food_energy immediately
   - Set laden_cooldown=False
4. **Nest delivery phase (NEW):** When agent with has_food=True is inside nest area:
   - Set has_food=False
   - Give 95% food_energy as energy + reward
   - Set laden_cooldown=False
5. **Field write phase (RESTRUCTURED):**
   - Ch1 (territory): ALL agents write +0.01 passively at their position
   - Ch0 (recruitment): ONLY agents with has_food=True AND laden_cooldown transitioning to True (write step) write 1.0 at their position
   - Cap all field values to [0, field_value_cap]
6. **Laden cooldown toggle:** For agents with has_food=True, flip laden_cooldown each step.
7. **Energy drain:** Same as current.
8. **Death check:** Same as current. If agent dies, set has_food=False.
9. **Reproduction phase (RESTRUCTURED):** Check all agents: if alive AND in nest area AND energy > threshold AND empty slot exists -> reproduce. Remove the old reproduce-action logic entirely.
10. **Update prev_field_at_pos:** Store current field center values for next step's temporal derivative.

The `reset()` function needs:
- Spawn agents in nest area (random positions within center 5x5)
- Initialize Ch1 at 1.0 in 7x7 area around nest center
- Initialize has_food, laden_cooldown to False
- Initialize prev_field_at_pos to 0.0

### src/environment/obs.py
Complete rewrite of `get_observations()` and `obs_dim()`:

```python
def obs_dim(config: Config) -> int:
    # 2 (pos) + 1 (energy) + 1 (has_food) + 2 (compass)
    # + 5 * num_channels (field spatial: N,S,E,W,center per channel)
    # + num_channels (field temporal: dC/dt per channel)
    # + K_NEAREST_FOOD * 3 (food)
    num_ch = config.field.num_channels
    return 2 + 1 + 1 + 2 + (5 * num_ch) + num_ch + (_K_NEAREST_FOOD * 3)
```

New `get_observations()`:
1. Normalized position (same as current)
2. Normalized energy (same as current)
3. Has food flag: `state.has_food.astype(float32)`
4. Nest compass with noise (new function `_compute_compass()`)
5. Field spatial gradients: for each channel, read field at (x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1). That's 5 reads per channel. Clip to [0, 1].
6. Field temporal derivative: `current_center - state.prev_field_at_pos` per channel
7. Food observations (same `_compute_food_obs()` as current)
8. Concatenate, apply alive mask

### src/field/dynamics.py
- `diffuse()`: Accept per-channel diffusion rates array. Apply different diffusion strength per channel via broadcasting.
- `decay()`: Accept per-channel decay rates array. Apply different decay per channel via broadcasting.
- Both must remain JIT-compatible (no Python loops over channels — use array ops).

### src/field/ops.py
- `write_local()`: After writing, clip field values to [0, cap]. Or add a separate `cap_field()` function called after all writes.
- No changes to `read_local()` (it's being replaced by direct point reads in obs.py).

### src/agents/network.py
- Change input handling if obs_dim changes (it will — from 502 to 45). The network architecture (hidden dims, activation) stays the same. Only the input layer size changes automatically since it's determined by obs_dim.
- Change output action count from 6 to 5.

### src/agents/policy.py
- Update any hardcoded action count references from 6 to 5.
- The `sample_actions()` function should work automatically if it reads num_actions from the network output.

### src/agents/reproduction.py
- No changes needed. The mutation and weight copying logic is independent of WHERE reproduction happens.

### Tests

New test file: `tests/test_pheromone_system.py` covering:

1. **Nest tests:**
   - Agents spawn within nest area
   - Ch1 initialized to 1.0 in 7x7 around nest
   - Reproduction only triggers inside nest area
   - Reproduction does NOT trigger outside nest area even with enough energy

2. **Food carrying tests:**
   - Food pickup sets has_food=True
   - Scout sip gives 5% energy on pickup
   - Nest delivery gives 95% energy
   - has_food resets to False on nest delivery
   - Dead agent's has_food resets to False

3. **Laden movement tests:**
   - Laden agent alternates move/write (check positions over multiple steps)
   - Unladen agent moves every step normally
   - laden_cooldown toggles correctly

4. **Field write tests:**
   - Ch0 only written by laden agents during write step
   - Ch1 written by ALL agents every step
   - Ch2-3 remain zero
   - Field values capped to [0, 1]

5. **Observation tests:**
   - obs_dim returns 45 (with 4 channels)
   - Gradient observations correct (manually set field, verify N/S/E/W/center values)
   - Temporal derivative correct (set prev_field_at_pos, verify difference)
   - Compass noise increases with distance
   - Compass is [0,0] at nest center
   - has_food flag appears in observation

6. **Per-channel dynamics tests:**
   - Different channels decay at different rates
   - Different channels diffuse at different rates
   - Ch2-3 with rates=0.0 remain unchanged

7. **Integration tests:**
   - Full episode: agents spawn, find food, carry back, deposit trail, reproduce at nest
   - JIT compatibility for all new functions
   - Action space is 5 (not 6)

Existing test files to update:
- `tests/test_obs.py` — update expected obs_dim, field patch tests now test gradients
- `tests/test_env.py` — update action count, reproduce logic, food pickup mechanics
- `tests/test_training.py` — update for new obs_dim and action count

## Backward Compatibility

This is a BREAKING change. Old checkpoints will NOT work with the new system because:
- obs_dim changes (502 -> 45)
- Action space changes (6 -> 5)
- EnvState has new fields

This is acceptable. The old field system doesn't work anyway. We're starting fresh experiments.

To preserve the option of running the old system for comparison:
- Keep `field_obs_radius` in FieldConfig (already implemented)
- The new pheromone system is enabled via presence of NestConfig (or a boolean flag)
- Actually, simpler: just make the breaking change. Old experiments are documented in EXPERIMENT_LOG.md.

## Implementation Order

Build in this sequence to keep tests passing at each step:

1. **Config changes** — Add NestConfig, per-channel rates to FieldConfig. No behavior change yet.
2. **State changes** — Add has_food, prev_field_at_pos, laden_cooldown to EnvState. Initialize to defaults in reset().
3. **Per-channel dynamics** — Modify diffuse() and decay() to accept per-channel rates. Tests.
4. **Field capping** — Add clip after writes in ops.py. Tests.
5. **Action space reduction** — Remove reproduce action (6->5) in network.py, policy.py, env.py. Update all action-related tests.
6. **Nest spawn + territory init** — Modify reset() for clustered spawn and Ch1 pre-seeding. Tests.
7. **Food carrying mechanics** — has_food flag, scout sip, laden movement toggle. Tests.
8. **Nest delivery** — Reward on food delivery at nest. Tests.
9. **Nest reproduction** — Automatic reproduction at nest. Tests.
10. **Field write restructuring** — Ch0 success-gated, Ch1 passive, Ch2-3 locked. Tests.
11. **New observations** — Gradient sensing, temporal derivative, compass, has_food obs. Tests.
12. **Integration test** — Full episode end-to-end.

## Success Criteria

The system works if:
1. All tests pass
2. Agents learn to: find food -> carry it back to nest -> deposit trail -> other agents follow trail
3. Field ON outperforms Field OFF (the reversal we need)
4. Population sustains in harsh environment (field ON agents no longer die)

## Open Items for Phase 2 (not in this build)

- Channels 2-3: learned neural network writes (DIAL-style)
- Food quality variation (higher quality = stronger pheromone)
- Nest congestion mechanics (negative feedback when crowded)
- Lane formation visualization
- Multiple nest entrances for large populations
