# Phase 2 PRD: Evolutionary Pressure                                                                                               
                                                                                                                                      
   ## Vision                                                                                                                          
                                                                                                                                      
   Add birth, death, and reproduction to create true evolutionary dynamics. Agents that survive and reproduce pass on mutated weights 
 — natural selection in RL.                                                                                                           
                                                                                                                                      
   ## Key Technical Challenge                                                                                                         
                                                                                                                                      
   JAX requires static array shapes for JIT compilation. We cannot dynamically add/remove agents. Solution: **Use a fixed max_agents  
 array with an alive mask**.                                                                                                          
                                                                                                                                      
   ## Success Criteria                                                                                                                
                                                                                                                                      
   1. Population dynamics emerge (boom/bust cycles or equilibrium)                                                                    
   2. Lineage tracking shows dominant "families"                                                                                      
   3. Weight divergence indicates different strategies evolving                                                                       
   4. Normal field >> Zeroed field gap INCREASES (field = survival advantage)                                                         
                                                                                                                                      
   ---                                                                                                                                
                                                                                                                                      
   ## User Stories                                                                                                                    
                                                                                                                                      
   ### US-001: Add Evolution Config [x]                                                                                               
   **Task:** Add evolution-related configuration parameters to the Config system.                                                     
   **Files:** `src/configs.py`, `configs/phase2.yaml`                                                                                 
   **Changes:**                                                                                                                       
   - Add `EvolutionConfig` dataclass with: `enabled: bool = True`, `starting_energy: int = 100`, `energy_per_step: int = 1`,          
 `food_energy: int = 50`, `max_energy: int = 200`, `reproduce_threshold: int = 150`, `reproduce_cost: int = 80`, `mutation_std: float 
  = 0.01`, `max_agents: int = 32`, `min_agents: int = 2`                                                                              
   - Add `evolution: EvolutionConfig` to main `Config`                                                                                
   - Create `configs/phase2.yaml` with evolution enabled                                                                              
   **Verification:** `python -c "from src.configs import Config; c = Config(); print(c.evolution.starting_energy)"` prints `100`      
                                                                                                                                      
   ### US-002: Extend EnvState with Energy and Alive Mask [x]                                                                         
   **Task:** Add energy tracking and alive status to environment state.                                                               
   **Files:** `src/environment/state.py`                                                                                              
   **Changes:**                                                                                                                       
   - Add `agent_energy: jnp.ndarray` shape `(max_agents,)` — energy per agent                                                         
   - Add `agent_alive: jnp.ndarray` shape `(max_agents,)` — boolean mask                                                              
   - Add `agent_ids: jnp.ndarray` shape `(max_agents,)` — unique ID per agent                                                         
   - Add `agent_parent_ids: jnp.ndarray` shape `(max_agents,)` — parent ID (-1 if original)                                           
   - Add `next_agent_id: jnp.ndarray` — scalar counter for unique IDs                                                                 
   - Update `agent_positions` to shape `(max_agents, 2)` with padding                                                                 
   **Verification:** `pytest tests/test_env.py -v` passes (update tests as needed)                                                    
                                                                                                                                      
   ### US-003: Update Environment Reset for Evolution [x]                                                                             
   **Task:** Initialize evolution state on reset.                                                                                     
   **Files:** `src/environment/env.py`                                                                                                
   **Changes:**                                                                                                                       
   - Initialize `agent_energy` to `starting_energy` for first `num_agents`, 0 for rest                                                
   - Initialize `agent_alive` to True for first `num_agents`, False for rest                                                          
   - Initialize `agent_ids` to `[0, 1, ..., num_agents-1, -1, -1, ...]`                                                               
   - Initialize `agent_parent_ids` to all `-1`                                                                                        
   - Initialize `next_agent_id` to `num_agents`                                                                                       
   - Pad `agent_positions` to `(max_agents, 2)` — use (0,0) for dead agents                                                           
   **Verification:** `pytest tests/test_env.py::TestEnvReset -v` passes                                                               
                                                                                                                                      
   ### US-004: Implement Energy Drain Per Step [x]                                                                                    
   **Task:** Decrease agent energy each step.                                                                                         
   **Files:** `src/environment/env.py`                                                                                                
   **Changes:**                                                                                                                       
   - In `step()`: subtract `energy_per_step` from alive agents                                                                        
   - Only apply to agents where `agent_alive == True`                                                                                 
   - Energy cannot go below 0                                                                                                         
   **Verification:** `pytest tests/test_energy.py -v` passes. Create this test file.                                                  
                                                                                                                                      
   ### US-005: Implement Death from Starvation [x]                                                                                    
   **Task:** Kill agents when energy reaches 0.                                                                                       
   **Files:** `src/environment/env.py`                                                                                                
   **Changes:**                                                                                                                       
   - After energy drain, set `agent_alive = False` for agents with `energy <= 0`                                                      
   - Dead agents stay in arrays but are masked out of observations/actions                                                            
   - Log death count in info dict                                                                                                     
   **Verification:** `pytest tests/test_energy.py::test_death -v` passes                                                              
                                                                                                                                      
   ### US-006: Implement Food Restores Energy [x]                                                                                     
   **Task:** Collecting food adds energy.                                                                                             
   **Files:** `src/environment/env.py`                                                                                                
   **Changes:**                                                                                                                       
   - When agent collects food, add `food_energy` to that agent's energy                                                               
   - Cap at `max_energy`                                                                                                              
   - Only the closest alive agent to food gets energy (not shared)                                                                    
   - Update reward: each agent gets reward equal to energy gained (individual, not shared)                                            
   **Verification:** `pytest tests/test_energy.py::test_food_energy -v` passes                                                        
                                                                                                                                      
   ### US-007: Implement Reproduction Action [x]                                                                                      
   **Task:** Add action 5 = reproduce.                                                                                                
   **Files:** `src/environment/env.py`, `src/agents/network.py`                                                                       
   **Changes:**                                                                                                                       
   - Update `ActorCritic` to output 6 actions instead of 5                                                                            
   - Action 5 = attempt reproduction                                                                                                  
   - Reproduction only succeeds if: `energy >= reproduce_threshold` AND free slot exists                                              
   - On success: deduct `reproduce_cost` from parent                                                                                  
   **Verification:** `pytest tests/test_reproduction.py -v` passes. Create this test file.                                            
                                                                                                                                      
   ### US-008: Implement Offspring Spawning [x]                                                                                       
   **Task:** Create offspring when reproduction succeeds.                                                                             
   **Files:** `src/environment/env.py`                                                                                                
   **Changes:**                                                                                                                       
   - Find first empty slot (where `agent_alive == False`)                                                                             
   - Set offspring position to random adjacent cell of parent                                                                         
   - Set offspring energy to `reproduce_cost` (energy transferred)                                                                    
   - Set `agent_alive = True` for offspring                                                                                           
   - Set `agent_ids[slot] = next_agent_id`, increment counter                                                                         
   - Set `agent_parent_ids[slot] = parent_id`                                                                                         
   **Verification:** `pytest tests/test_reproduction.py::test_spawn -v` passes                                                        
                                                                                                                                      
   ### US-009: Implement Weight Inheritance [x]                                                                                       
   **Task:** Offspring inherits parent's neural network weights with mutation.                                                        
   **Files:** `src/agents/reproduction.py` (new), `src/environment/env.py`                                                            
   **Changes:**                                                                                                                       
   - Create `src/agents/reproduction.py` with:                                                                                        
     - `mutate_params(params, key, mutation_std)` — adds Gaussian noise to all weights                                                
     - `copy_agent_params(all_params, parent_idx, child_idx)` — copies params between agents                                          
   - Store per-agent params: shape changes from `params` to `(max_agents, params)`                                                    
   - On reproduction: `child_params = mutate_params(parent_params, key, std)`                                                         
   **Verification:** `pytest tests/test_reproduction.py::test_inheritance -v` passes                                                  
                                                                                                                                      
   ### US-010: Update Observations for Variable Population [x]                                                                        
   **Task:** Handle dead agents in observations.                                                                                      
   **Files:** `src/environment/obs.py`                                                                                                
   **Changes:**                                                                                                                       
   - Add agent's own energy (normalized to [0, 1]) to observation                                                                     
   - Dead agents get zero observations                                                                                                
   - Mask out dead agents in relative position calculations                                                                           
   **Verification:** `pytest tests/test_obs.py -v` passes                                                                             
                                                                                                                                      
   ### US-011: Update Training for Variable Population [x]                                                                            
   **Task:** Mask dead agents in training loop.                                                                                       
   **Files:** `src/training/rollout.py`, `src/training/train.py`, `src/training/ppo.py`                                               
   **Changes:**                                                                                                                       
   - Collect `alive_mask` in rollout alongside other data                                                                             
   - In PPO loss, mask out contributions from dead agents                                                                             
   - In GAE, only compute for alive agents                                                                                            
   - Handle per-agent params in forward pass                                                                                          
   **Verification:** `pytest tests/test_training.py -v` passes                                                                        
                                                                                                                                      
   ### US-012: Implement Food Respawn with Scarcity [x]                                                                               
   **Task:** Food respawns slowly to create competition.                                                                              
   **Files:** `src/environment/env.py`, `src/configs.py`                                                                              
   **Changes:**                                                                                                                       
   - Add `food_respawn_prob: float = 0.1` to EnvConfig                                                                                
   - Each step, collected food has `food_respawn_prob` chance to respawn at random location                                           
   - Total food capped at `num_food`                                                                                                  
   **Verification:** `pytest tests/test_env.py::test_food_respawn -v` passes                                                          
                                                                                                                                      
   ### US-013: Implement Lineage Tracking [x]                                                                                         
   **Task:** Track family trees for analysis.                                                                                         
   **Files:** `src/analysis/lineage.py` (new)                                                                                         
   **Changes:**                                                                                                                       
   - `LineageTracker` class that records: birth time, death time, parent, children                                                    
   - `get_lineage_depth(agent_id)` — how many generations from original                                                               
   - `get_family_tree(agent_id)` — all descendants                                                                                    
   - `get_dominant_lineages(tracker, top_k=5)` — lineages with most descendants                                                       
   **Verification:** `pytest tests/test_lineage.py -v` passes. Create this test file.                                                 
                                                                                                                                      
   ### US-014: Add Population Metrics to Training [x]                                                                                 
   **Task:** Log population dynamics during training.                                                                                 
   **Files:** `src/training/train.py`                                                                                                 
   **Changes:**                                                                                                                       
   - Track: `population_size`, `births_this_step`, `deaths_this_step`                                                                 
   - Track: `mean_energy`, `max_energy`, `min_energy` of alive agents                                                                 
   - Track: `oldest_agent_age` (steps alive)                                                                                          
   - Log to metrics dict and W&B                                                                                                      
   **Verification:** Training output shows population stats                                                                           
                                                                                                                                      
   ### US-015: Update Ablation for Evolution [ ]                                                                                      
   **Task:** Compare evolution vs no-evolution conditions.                                                                            
   **Files:** `src/analysis/ablation.py`, `scripts/run_ablation.py`                                                                   
   **Changes:**                                                                                                                       
   - Add condition: `evolution_disabled` — fixed population, no energy                                                                
   - Compare: normal (field + evolution) vs field-only vs evolution-only vs neither                                                   
   - Report: final population, total births, total deaths, survival rate                                                              
   **Verification:** `python scripts/run_ablation.py --evolution` works                                                               
                                                                                                                                      
   ### US-016: Review and Integration Test [ ]                                                                                        
   **Task:** Full system validation.                                                                                                  
   **Files:** `tests/test_integration.py`                                                                                             
   **Changes:**                                                                                                                       
   - Update integration test for Phase 2                                                                                              
   - Test: population grows, stabilizes, or crashes depending on food                                                                 
   - Test: lineage tracking works across multiple generations                                                                         
   - Test: weight divergence increases over time                                                                                      
   **Verification:** `pytest tests/test_integration.py -v` passes                                                                     
                                                                                                                                      
   ### US-017: Final Review and Cleanup [ ]                                                                                           
   **Task:** Code quality and documentation.                                                                                          
   **Files:** All modified files                                                                                                      
   **Verification:**                                                                                                                  
   - `pytest tests/ -v` — all pass                                                                                                    
   - `python -m mypy src/ --ignore-missing-imports` — no errors                                                                       
   - Training runs 1M steps without crash                                                                                             
   - README updated with Phase 2 instructions                                                                                         
                                                                                                                                      
   ---                                                                                                                                
                                                                                                                                      
   ## Technical Notes                                                                                                                 
                                                                                                                                      
   ### JAX Static Shapes Solution                                                                                                     
   ```python                                                                                                                          
   # Instead of dynamic lists, use fixed arrays with masks:                                                                           
   agent_positions = jnp.zeros((max_agents, 2))  # All slots pre-allocated                                                            
   agent_alive = jnp.array([True, True, ..., False, False])  # Mask                                                                   
                                                                                                                                      
   # In step(), use jnp.where() for conditional updates:                                                                              
   energy = jnp.where(agent_alive, energy - 1, energy)                                                                                
 ```                                                                                                                                  
                                                                                                                                      
 ### Per-Agent Parameters                                                                                                             
                                                                                                                                      
 Each agent needs its own copy of network weights for inheritance. Options:                                                           
 1. Vmapped params — store (max_agents, param_shape) and vmap forward pass                                                            
 2. Shared backbone, per-agent head — only final layer differs                                                                        
 3. Parameter server — store params outside JAX, copy on mutation                                                                     
                                                                                                                                      
 Recommend Option 1 for simplicity. May need to adjust for memory.                                                                    
                                                                                                                                      
 ### Reproduction Mechanics                                                                                                           
                                                                                                                                      
 - Action 5 = "attempt reproduce"                                                                                                     
 - Fails silently if: energy < threshold, no empty slots, at max population                                                           
 - Success: parent loses energy, child appears adjacent, inherits mutated weights                                                     
                                                                                                                                      
 ### Death Mechanics                                                                                                                  
                                                                                                                                      
 - Agents with energy <= 0 have alive set to False                                                                                    
 - They remain in arrays but are masked from all computations                                                                         
 - Their slot can be reused by new offspring                                                                                          
                                                                                                                                      
 Out of Scope (Phase 3+)                                                                                                              
                                                                                                                                      
 - Sexual reproduction (crossover)                                                                                                    
 - Speciation detection algorithms                                                                                                    
 - Explicit agent communication                                                                                                       
 - Predator/prey dynamics                                                                                                             
 - Environmental variation                                                                                                            
                                                                                                                                      
 ────────────────────────────────────────────────────────────────────────────────                                                     
                                                                                                                                      
 PRD designed for Ralph Loop autonomous execution.                                                                                    
 Each story should complete in one Claude context window.                                                                             
 Total: 17 stories expected.                                                                                                          
 ENDPRD
