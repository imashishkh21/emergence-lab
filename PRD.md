   # Phase 2 PRD: Evolutionary Pressure                                                                                               
                                                                                                                                      
   ## Vision                                                                                                                          
                                                                                                                                      
   Add birth, death, and reproduction to create true evolutionary dynamics. Agents that survive and reproduce pass on mutated weights 
 — natural selection in RL.                                                                                                           
                                                                                                                                      
   ## Success Criteria                                                                                                                
                                                                                                                                      
   1. Population dynamics emerge (boom/bust cycles or equilibrium)                                                                    
   2. Lineage tracking shows dominant "families"                                                                                      
   3. Weight divergence indicates different strategies evolving                                                                       
   4. Normal field >> Zeroed field gap INCREASES (field = survival advantage)                                                         
                                                                                                                                      
   ---                                                                                                                                
                                                                                                                                      
   ## User Stories                                                                                                                    
                                                                                                                                      
   ### US-001: Agent Energy System                                                                                                    
   **Task:** Add energy attribute to agents. Start at 100. Decrease by 1 each step.                                                   
   **Files:** `src/environment/state.py`, `src/environment/env.py`                                                                    
   **Verification:** `pytest tests/test_energy.py -v` passes. Print agent energies, confirm decrease each step.                       
                                                                                                                                      
   ### US-002: Death Mechanic                                                                                                         
   **Task:** Agents with energy <= 0 are marked dead. Dead agents don't act or receive observations. Remove from active agent list.   
   **Files:** `src/environment/state.py`, `src/environment/env.py`                                                                    
   **Verification:** `pytest tests/test_death.py -v` passes. Agent with 0 energy disappears from simulation.                          
                                                                                                                                      
   ### US-003: Food Restores Energy                                                                                                   
   **Task:** Collecting food adds +50 energy to the agent (capped at 200).                                                            
   **Files:** `src/environment/env.py`                                                                                                
   **Verification:** `pytest tests/test_energy.py -v` passes. Agent eats food, energy increases by 50.                                
                                                                                                                                      
   ### US-004: Reproduction Mechanic                                                                                                  
   **Task:** Agent with energy >= 150 can reproduce. Costs 80 energy. Spawns offspring at adjacent cell.                              
   **Files:** `src/environment/env.py`, `src/environment/state.py`                                                                    
   **Verification:** `pytest tests/test_reproduction.py -v` passes. High-energy agent spawns child.                                   
                                                                                                                                      
   ### US-005: Weight Inheritance with Mutation                                                                                       
   **Task:** Offspring inherits parent's network weights + Gaussian noise (std=0.01). Store parent_id for lineage tracking.           
   **Files:** `src/agents/reproduction.py` (new), `src/environment/env.py`                                                            
   **Verification:** `pytest tests/test_reproduction.py -v` passes. Child weights ≈ parent weights (not identical).                   
                                                                                                                                      
   ### US-006: Dynamic Agent Registry                                                                                                 
   **Task:** Replace fixed agent array with dynamic list. Support variable population (min=2, max=32).                                
   **Files:** `src/environment/state.py`, `src/environment/env.py`, `src/environment/obs.py`                                          
   **Verification:** `pytest tests/test_env.py -v` passes. Population can grow and shrink.                                            
                                                                                                                                      
   ### US-007: Lineage Tracking                                                                                                       
   **Task:** Each agent has unique ID and parent_id. Track births, deaths, lineage depth.                                             
   **Files:** `src/environment/state.py`, `src/analysis/lineage.py` (new)                                                             
   **Verification:** `pytest tests/test_lineage.py -v` passes. Can reconstruct family tree.                                           
                                                                                                                                      
   ### US-008: Population Metrics                                                                                                     
   **Task:** Track population size, births, deaths per episode. Log to metrics.                                                       
   **Files:** `src/training/train.py`, `src/analysis/emergence.py`                                                                    
   **Verification:** Training output shows population stats.                                                                          
                                                                                                                                      
   ### US-009: Update Observation Space                                                                                               
   **Task:** Add agent's own energy to observation vector. Normalize to [0, 1].                                                       
   **Files:** `src/environment/obs.py`                                                                                                
   **Verification:** `pytest tests/test_obs.py -v` passes. Observation includes energy.                                               
                                                                                                                                      
   ### US-010: Variable Batch Handling                                                                                                
   **Task:** Update training to handle variable number of agents per env. Pad/mask as needed.                                         
   **Files:** `src/training/rollout.py`, `src/training/train.py`                                                                      
   **Verification:** `pytest tests/test_training.py -v` passes with dynamic population.                                               
                                                                                                                                      
   ### US-011: Reproduction Action                                                                                                    
   **Task:** Add action 5 = "reproduce" (only works if energy >= 150).                                                                
   **Files:** `src/environment/env.py`, `src/agents/network.py`                                                                       
   **Verification:** Agent can choose to reproduce. Action fails gracefully if insufficient energy.                                   
                                                                                                                                      
   ### US-012: Carrying Capacity via Food Scarcity                                                                                    
   **Task:** Limit food respawn rate. Total food capped at num_food. Creates competition.                                             
   **Files:** `src/environment/env.py`                                                                                                
   **Verification:** Food doesn't respawn infinitely. Population limited by food availability.                                        
                                                                                                                                      
   ### US-013: New Config Options                                                                                                     
   **Task:** Add to Config: starting_energy, energy_per_step, food_energy, reproduce_threshold, reproduce_cost, mutation_std,         
 max_population.                                                                                                                      
   **Files:** `src/configs.py`, `configs/phase2.yaml`                                                                                 
   **Verification:** `python -c "from src.configs import Config; c = Config(); print(c.evolution)"` works.                            
                                                                                                                                      
   ### US-014: Evolutionary Ablation Test                                                                                             
   **Task:** Compare evolution enabled vs disabled. Does evolution improve collective performance?                                    
   **Files:** `src/analysis/ablation.py`, `scripts/run_ablation.py`                                                                   
   **Verification:** Script runs both conditions and compares.                                                                        
                                                                                                                                      
   ### US-015: Review & Refactor                                                                                                      
   **Task:** Review all Phase 2 code. Ensure clean interfaces, docstrings, type hints. Run full test suite.                           
   **Files:** All modified files                                                                                                      
   **Verification:** `pytest tests/ -v` all pass. `python -m py_compile src/**/*.py` succeeds.                                        
                                                                                                                                      
   ---                                                                                                                                
                                                                                                                                      
   ## Technical Notes                                                                                                                 
                                                                                                                                      
   - Use masking for dead agents rather than removing from arrays (JAX prefers static shapes)                                         
   - Parent weights stored separately, copied on reproduction                                                                         
   - Lineage is metadata only — doesn't affect RL, just tracking                                                                      
                                                                                                                                      
   ## Out of Scope (Phase 3+)                                                                                                         
                                                                                                                                      
   - Crossover (two-parent reproduction)                                                                                              
   - Speciation detection                                                                                                             
   - Agent communication (explicit signals)                                                                                           
   - Predator/prey dynamics                                                                                                           
