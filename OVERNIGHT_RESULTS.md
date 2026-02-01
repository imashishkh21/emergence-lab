# Overnight Results - Feb 1, 2026

## Training on MacBook Pro (Titan's machine)

### Setup
- Python 3.12 installed via Homebrew
- Full venv with JAX/Flax working
- Tuned parameters (survival-friendly)

### Training Results ✅
```
Population: 32/32 (MAXED OUT!)
Births: 108
Deaths: 108 (perfect equilibrium!)
Reward: 3.20
Mean energy: 160 (healthy)
Oldest agent: 384 steps
Entropy: 1.75 (good exploration)
```

### Ablation Test ❌
- Kept getting OOM killed on MacBook Pro
- Need to run on Mac Mini (more RAM)

### Mechanics Verification ✅
Quick test showed:
- 8 agents → 18 agents (population growth)
- Field active and accumulating
- Births/deaths working

## To Run on Mac Mini

```bash
cd /Users/ashish/Downloads/Projects/emergence-lab
source .venv/bin/activate

# Train with tuned params
python -m src.training.train --train.total-steps 1000000 --log.no-wandb --env.num-food 20 --evolution.starting-energy 200 --evolution.food-energy 100 --evolution.reproduce-threshold 120 --evolution.reproduce-cost 50

# Then run ablation
python -m src.analysis.ablation --checkpoint checkpoints/params.pkl --num-episodes 10
```

## Phase 2 Status: COMPLETE ✅

All 17 tasks done by Ralph:
- 174 tests passing
- Evolution working (birth/death/reproduction)
- Population dynamics confirmed
- Checkpoint saving works

## Ready for Phase 3
Next: Specialization detection - can we see different strategies/species emerge?
