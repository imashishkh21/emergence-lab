# Specialization Analysis Report

## Overview

- **Grid size**: 20x20
- **Max agents**: 32
- **Alive agents**: 0
- **Trajectory episodes recorded**: 5
- **Mutation std**: 0.01
- **Evolution enabled**: True

## Specialization Score

**Composite Score: 0.5771** (0 = identical, 1 = fully specialized)

| Component | Score |
|-----------|-------|
| Silhouette | 0.8069 |
| Weight Divergence | 0.0000 |
| Behavioral Variance | 0.6948 |
| Optimal Clusters (k) | 2 |

## Weight Divergence

- **Mean divergence**: 0.000000
- **Max divergence**: 0.000000
- **Agents compared**: 0

![Weight Divergence Over Time](figures/weight_divergence.png)

## Species Detection

- **Species detected**: 2
- **Silhouette**: 0.8069
- **Optimal k**: 2
- **Heredity score**: 0.0000
- **Speciation observed**: Yes

### Detected Species

| Species | Members | Heredity | Role | Key Features |
|---------|---------|----------|------|--------------|
| Cluster 0 | 12 | 1.00 | unknown | movement_entropy=0.939, food_collection_rate=0.619, distance_per_step=0.680 |
| Cluster 1 | 20 | 1.00 | unknown | movement_entropy=0.068, food_collection_rate=0.059, distance_per_step=0.049 |

![Behavior Clusters](figures/behavior_clusters_pca.png)

## Field Usage by Cluster

- **Clusters analyzed**: 2

| Cluster | Role | Write Freq | Mean Field | Movement | Spread | Field-Action Corr |
|---------|------|------------|------------|----------|--------|-------------------|
| 0 | balanced | 0.353 | 2.848 | 0.684 | 0.288 | -0.077 |
| 1 | balanced | 0.013 | 0.182 | 0.028 | 0.008 | 0.004 |

![Field Usage by Cluster](figures/field_usage.png)

## Visualizations

All figures saved to the `figures/` subdirectory:

- `behavior_clusters_pca.png` — PCA scatter of agent behaviors
- `behavior_clusters_tsne.png` — t-SNE scatter of agent behaviors
- `weight_divergence.png` — Weight divergence over training
- `field_usage.png` — Field usage metrics by cluster
- `specialization_score.png` — Specialization score over training
