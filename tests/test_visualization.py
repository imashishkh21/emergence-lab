"""Tests for specialization visualization module."""

import numpy as np
import pytest
from matplotlib.figure import Figure


class TestPlotBehaviorClusters:
    """Tests for plot_behavior_clusters."""

    def _make_data(self, n_agents=20, n_clusters=3, seed=42):
        rng = np.random.RandomState(seed)
        features = rng.randn(n_agents, 7)
        labels = rng.randint(0, n_clusters, size=n_agents)
        return features, labels

    def test_returns_figure(self):
        from src.analysis.visualization import plot_behavior_clusters

        features, labels = self._make_data()
        fig = plot_behavior_clusters(features, labels)
        assert isinstance(fig, Figure)

    def test_saves_png(self, tmp_path):
        from src.analysis.visualization import plot_behavior_clusters

        features, labels = self._make_data()
        out = tmp_path / "clusters.png"
        plot_behavior_clusters(features, labels, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_pca_axes_labels(self):
        from src.analysis.visualization import plot_behavior_clusters

        features, labels = self._make_data()
        fig = plot_behavior_clusters(features, labels, method="pca")
        ax = fig.axes[0]
        assert "PC1" in ax.get_xlabel()
        assert "PC2" in ax.get_ylabel()

    def test_tsne_method(self):
        from src.analysis.visualization import plot_behavior_clusters

        features, labels = self._make_data()
        fig = plot_behavior_clusters(features, labels, method="tsne")
        assert isinstance(fig, Figure)

    def test_single_cluster(self):
        from src.analysis.visualization import plot_behavior_clusters

        rng = np.random.RandomState(0)
        features = rng.randn(10, 7)
        labels = np.zeros(10, dtype=int)
        fig = plot_behavior_clusters(features, labels)
        assert isinstance(fig, Figure)


class TestPlotWeightDivergenceOverTime:
    """Tests for plot_weight_divergence_over_time."""

    def _make_history(self, n_steps=50, include_max=True, seed=42):
        rng = np.random.RandomState(seed)
        steps = list(range(0, n_steps * 1000, 1000))
        mean_div = np.cumsum(rng.uniform(0, 0.01, size=n_steps)).tolist()
        history = {"steps": steps, "weight_divergence": mean_div}
        if include_max:
            max_div = [m + rng.uniform(0, 0.05) for m in mean_div]
            history["max_divergence"] = max_div
        return history

    def test_returns_figure(self):
        from src.analysis.visualization import plot_weight_divergence_over_time

        history = self._make_history()
        fig = plot_weight_divergence_over_time(history)
        assert isinstance(fig, Figure)

    def test_saves_png(self, tmp_path):
        from src.analysis.visualization import plot_weight_divergence_over_time

        history = self._make_history()
        out = tmp_path / "divergence.png"
        plot_weight_divergence_over_time(history, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_with_max_divergence(self):
        from src.analysis.visualization import plot_weight_divergence_over_time

        history = self._make_history(include_max=True)
        fig = plot_weight_divergence_over_time(history)
        ax = fig.axes[0]
        # Should have 2 lines (mean + max) plus the fill
        line_labels = [line.get_label() for line in ax.get_lines()]
        assert "Mean Divergence" in line_labels
        assert "Max Divergence" in line_labels

    def test_without_max_divergence(self):
        from src.analysis.visualization import plot_weight_divergence_over_time

        history = self._make_history(include_max=False)
        fig = plot_weight_divergence_over_time(history)
        ax = fig.axes[0]
        line_labels = [line.get_label() for line in ax.get_lines()]
        assert "Mean Divergence" in line_labels
        assert "Max Divergence" not in line_labels


class TestPlotFieldUsageByCluster:
    """Tests for plot_field_usage_by_cluster."""

    def _make_usage_data(self, n_clusters=3, seed=42):
        rng = np.random.RandomState(seed)
        roles = ["writer", "reader", "balanced"]
        per_cluster = {}
        cluster_roles = {}
        for i in range(n_clusters):
            per_cluster[i] = {
                "write_frequency": rng.uniform(0, 1),
                "mean_field_value": rng.uniform(0, 1),
                "movement_rate": rng.uniform(0, 1),
                "spatial_spread": rng.uniform(0, 1),
                "field_action_correlation": rng.uniform(-1, 1),
            }
            cluster_roles[i] = roles[i % len(roles)]
        return {
            "per_cluster": per_cluster,
            "cluster_roles": cluster_roles,
            "num_clusters": n_clusters,
        }

    def test_returns_figure(self):
        from src.analysis.visualization import plot_field_usage_by_cluster

        data = self._make_usage_data()
        fig = plot_field_usage_by_cluster(data)
        assert isinstance(fig, Figure)

    def test_saves_png(self, tmp_path):
        from src.analysis.visualization import plot_field_usage_by_cluster

        data = self._make_usage_data()
        out = tmp_path / "field_usage.png"
        plot_field_usage_by_cluster(data, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_multiple_clusters(self):
        from src.analysis.visualization import plot_field_usage_by_cluster

        data = self._make_usage_data(n_clusters=3)
        fig = plot_field_usage_by_cluster(data)
        assert isinstance(fig, Figure)

    def test_roles_in_legend(self):
        from src.analysis.visualization import plot_field_usage_by_cluster

        data = self._make_usage_data(n_clusters=3)
        fig = plot_field_usage_by_cluster(data)
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        all_text = " ".join(legend_texts)
        assert "writer" in all_text
        assert "reader" in all_text
        assert "balanced" in all_text


class TestPlotSpecializationScoreOverTime:
    """Tests for plot_specialization_score_over_time."""

    def _make_scores(self, n_steps=50, include_components=True, seed=42):
        rng = np.random.RandomState(seed)
        steps = list(range(0, n_steps * 1000, 1000))
        scores_vals = np.clip(
            np.cumsum(rng.uniform(-0.01, 0.02, size=n_steps)), 0, 1
        ).tolist()
        scores = {"steps": steps, "scores": scores_vals}
        if include_components:
            scores["silhouette_component"] = np.clip(
                np.cumsum(rng.uniform(-0.01, 0.02, size=n_steps)), 0, 1
            ).tolist()
            scores["divergence_component"] = np.clip(
                np.cumsum(rng.uniform(-0.01, 0.02, size=n_steps)), 0, 1
            ).tolist()
            scores["variance_component"] = np.clip(
                np.cumsum(rng.uniform(-0.01, 0.02, size=n_steps)), 0, 1
            ).tolist()
        return scores

    def test_returns_figure(self):
        from src.analysis.visualization import plot_specialization_score_over_time

        scores = self._make_scores()
        fig = plot_specialization_score_over_time(scores)
        assert isinstance(fig, Figure)

    def test_saves_png(self, tmp_path):
        from src.analysis.visualization import plot_specialization_score_over_time

        scores = self._make_scores()
        out = tmp_path / "spec_score.png"
        plot_specialization_score_over_time(scores, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_y_axis_bounded(self):
        from src.analysis.visualization import plot_specialization_score_over_time

        scores = self._make_scores()
        fig = plot_specialization_score_over_time(scores)
        ax = fig.axes[0]
        ylim = ax.get_ylim()
        assert ylim[0] >= 0
        assert ylim[1] <= 1.1

    def test_with_components(self):
        from src.analysis.visualization import plot_specialization_score_over_time

        scores = self._make_scores(include_components=True)
        fig = plot_specialization_score_over_time(scores)
        ax = fig.axes[0]
        line_labels = [line.get_label() for line in ax.get_lines()]
        assert "Composite Score" in line_labels
        assert "Silhouette" in line_labels
        assert "Divergence" in line_labels
        assert "Variance" in line_labels

    def test_without_components(self):
        from src.analysis.visualization import plot_specialization_score_over_time

        scores = self._make_scores(include_components=False)
        fig = plot_specialization_score_over_time(scores)
        ax = fig.axes[0]
        line_labels = [line.get_label() for line in ax.get_lines()]
        assert "Composite Score" in line_labels
        assert "Silhouette" not in line_labels
        assert "Divergence" not in line_labels
        assert "Variance" not in line_labels
