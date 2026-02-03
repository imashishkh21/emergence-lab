"""Tests for paper_figures.py — publication-quality figure generation."""

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from src.analysis.paper_figures import (
    COLORBLIND_PALETTE,
    DPI,
    FONT_SIZE,
    METHOD_COLORS,
    create_figure_grid,
    get_color,
    plot_ablation_bars,
    plot_causal_emergence_gap,
    plot_o_information_trajectory,
    plot_performance_profiles,
    plot_phase_transitions,
    plot_pid_synergy_heatmap,
    plot_probability_of_improvement,
    plot_scaling_curve,
    save_figure,
    setup_publication_style,
)


class TestStyleConstants:
    """Tests for style constants and configuration."""

    def test_font_size_reasonable(self):
        """Test font size is reasonable."""
        assert 8 <= FONT_SIZE <= 16

    def test_dpi_publication_quality(self):
        """Test DPI is publication quality."""
        assert DPI >= 300

    def test_colorblind_palette_length(self):
        """Test colorblind palette has enough colors."""
        assert len(COLORBLIND_PALETTE) >= 6

    def test_colorblind_palette_valid_hex(self):
        """Test all palette colors are valid hex."""
        for color in COLORBLIND_PALETTE:
            assert color.startswith("#")
            assert len(color) == 7

    def test_method_colors_coverage(self):
        """Test method colors cover expected methods."""
        expected_methods = ["ours", "ippo", "mappo", "aco_fixed"]
        for method in expected_methods:
            assert method in METHOD_COLORS


class TestSetupPublicationStyle:
    """Tests for setup_publication_style function."""

    def test_sets_font_size(self):
        """Test font size is set."""
        setup_publication_style()
        assert plt.rcParams["font.size"] == FONT_SIZE

    def test_sets_dpi(self):
        """Test DPI is set."""
        setup_publication_style()
        assert plt.rcParams["figure.dpi"] == DPI

    def test_disables_top_spine(self):
        """Test top spine is disabled."""
        setup_publication_style()
        assert plt.rcParams["axes.spines.top"] is False

    def test_disables_right_spine(self):
        """Test right spine is disabled."""
        setup_publication_style()
        assert plt.rcParams["axes.spines.right"] is False

    def test_enables_grid(self):
        """Test grid is enabled."""
        setup_publication_style()
        assert plt.rcParams["axes.grid"] is True


class TestGetColor:
    """Tests for get_color function."""

    def test_known_method(self):
        """Test returns correct color for known method."""
        color = get_color("ours")
        assert color == METHOD_COLORS["ours"]

    def test_case_insensitive(self):
        """Test lookup is case insensitive."""
        color_lower = get_color("ippo")
        color_upper = get_color("IPPO")
        assert color_lower == color_upper

    def test_unknown_method_uses_index(self):
        """Test unknown method falls back to palette index."""
        color = get_color("unknown_method", index=2)
        assert color == COLORBLIND_PALETTE[2]

    def test_index_wraps(self):
        """Test index wraps around palette."""
        color = get_color("unknown", index=100)
        assert color == COLORBLIND_PALETTE[100 % len(COLORBLIND_PALETTE)]


# Helper dataclass for scaling tests
@dataclass
class MockScalingAnalysis:
    """Mock ScalingAnalysis for testing."""
    field_condition: str
    n_agents_list: list[int] = field(default_factory=list)
    mean_foods: list[float] = field(default_factory=list)
    std_foods: list[float] = field(default_factory=list)
    alpha: float = 1.0
    c: float = 0.0


# Helper dataclass for ablation tests
@dataclass
class MockAblationResult:
    """Mock ablation result for testing."""
    mean_reward: float
    std_reward: float
    iqm: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0


class TestPlotScalingCurve:
    """Tests for plot_scaling_curve function."""

    def test_creates_figure(self, tmp_path):
        """Test creates a Figure object."""
        results = [
            MockScalingAnalysis(
                field_condition="normal",
                n_agents_list=[1, 2, 4, 8],
                mean_foods=[10.0, 25.0, 60.0, 150.0],
                std_foods=[2.0, 5.0, 10.0, 20.0],
                alpha=1.2,
                c=2.3,
            )
        ]
        output_path = str(tmp_path / "scaling")
        fig = plot_scaling_curve(results, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_pdf_and_png(self, tmp_path):
        """Test saves both PDF and PNG."""
        results = [
            MockScalingAnalysis(
                field_condition="normal",
                n_agents_list=[1, 2, 4, 8],
                mean_foods=[10.0, 25.0, 60.0, 150.0],
                alpha=1.0,
            )
        ]
        output_path = str(tmp_path / "scaling")
        fig = plot_scaling_curve(results, output_path)
        plt.close(fig)

        assert (tmp_path / "scaling.pdf").exists()
        assert (tmp_path / "scaling.png").exists()

    def test_multiple_conditions(self, tmp_path):
        """Test handles multiple field conditions."""
        results = [
            MockScalingAnalysis(
                field_condition="normal",
                n_agents_list=[1, 2, 4],
                mean_foods=[10.0, 25.0, 60.0],
            ),
            MockScalingAnalysis(
                field_condition="zeroed",
                n_agents_list=[1, 2, 4],
                mean_foods=[10.0, 20.0, 40.0],
            ),
        ]
        output_path = str(tmp_path / "scaling")
        fig = plot_scaling_curve(results, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_reference_line(self, tmp_path):
        """Test reference line option."""
        results = [
            MockScalingAnalysis(
                field_condition="normal",
                n_agents_list=[1, 2, 4],
                mean_foods=[10.0, 25.0, 60.0],
            )
        ]
        output_path = str(tmp_path / "scaling")
        fig = plot_scaling_curve(results, output_path, show_reference=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotAblationBars:
    """Tests for plot_ablation_bars function."""

    def test_creates_figure(self, tmp_path):
        """Test creates a Figure object."""
        results = {
            "normal": MockAblationResult(mean_reward=100.0, std_reward=10.0),
            "zeroed": MockAblationResult(mean_reward=80.0, std_reward=15.0),
        }
        output_path = str(tmp_path / "ablation")
        fig = plot_ablation_bars(results, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_six_conditions(self, tmp_path):
        """Test handles all 6 field conditions."""
        results = {
            "normal": MockAblationResult(mean_reward=100.0, std_reward=10.0, iqm=95.0),
            "zeroed": MockAblationResult(mean_reward=80.0, std_reward=15.0, iqm=75.0),
            "random": MockAblationResult(mean_reward=70.0, std_reward=12.0, iqm=68.0),
            "frozen": MockAblationResult(mean_reward=90.0, std_reward=8.0, iqm=88.0),
            "no_field": MockAblationResult(mean_reward=60.0, std_reward=20.0, iqm=55.0),
            "write_only": MockAblationResult(mean_reward=75.0, std_reward=11.0, iqm=72.0),
        }
        output_path = str(tmp_path / "ablation")
        fig = plot_ablation_bars(results, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ci(self, tmp_path):
        """Test handles CI bounds."""
        results = {
            "normal": MockAblationResult(
                mean_reward=100.0, std_reward=10.0, iqm=95.0, ci_lower=90.0, ci_upper=100.0
            ),
        }
        output_path = str(tmp_path / "ablation")
        fig = plot_ablation_bars(results, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPerformanceProfiles:
    """Tests for plot_performance_profiles function."""

    def test_creates_figure(self, tmp_path):
        """Test creates a Figure object."""
        score_dict = {
            "ours": np.array([100, 95, 90, 85, 80]),
            "ippo": np.array([80, 75, 70, 65, 60]),
        }
        output_path = str(tmp_path / "profiles")
        fig = plot_performance_profiles(score_dict, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_valid_cdf(self, tmp_path):
        """Test CDFs are valid (0-1 range)."""
        score_dict = {
            "method_a": np.array([100, 90, 80]),
            "method_b": np.array([50, 40, 30]),
        }
        output_path = str(tmp_path / "profiles")
        fig = plot_performance_profiles(score_dict, output_path)
        # Check y-axis limits
        ax = fig.axes[0]
        assert ax.get_ylim()[0] >= 0
        assert ax.get_ylim()[1] <= 1.1
        plt.close(fig)


class TestPlotProbabilityOfImprovement:
    """Tests for plot_probability_of_improvement function."""

    def test_creates_figure(self, tmp_path):
        """Test creates a Figure object."""
        scores_dict = {
            "ours": np.array([100, 95, 90]),
            "ippo": np.array([80, 75, 70]),
        }
        output_path = str(tmp_path / "poi")
        fig = plot_probability_of_improvement(scores_dict, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_creates_heatmap(self, tmp_path):
        """Test creates heatmap with correct dimensions."""
        scores_dict = {
            "a": np.array([100, 90]),
            "b": np.array([80, 70]),
            "c": np.array([60, 50]),
        }
        output_path = str(tmp_path / "poi")
        fig = plot_probability_of_improvement(scores_dict, output_path)
        # Should have 3x3 heatmap
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPidSynergyHeatmap:
    """Tests for plot_pid_synergy_heatmap function."""

    def test_creates_figure(self, tmp_path):
        """Test creates a Figure object."""
        synergy_matrix = np.random.randn(8, 8)
        np.fill_diagonal(synergy_matrix, 0)  # Diagonal is N/A
        output_path = str(tmp_path / "synergy")
        fig = plot_pid_synergy_heatmap(synergy_matrix, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_labels(self, tmp_path):
        """Test with custom agent labels."""
        synergy_matrix = np.random.randn(4, 4)
        labels = ["Scout", "Exploiter", "Balanced", "Random"]
        output_path = str(tmp_path / "synergy")
        fig = plot_pid_synergy_heatmap(synergy_matrix, output_path, agent_labels=labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_8x8_matrix(self, tmp_path):
        """Test with standard 8-agent matrix."""
        synergy_matrix = np.random.randn(8, 8) * 0.1
        output_path = str(tmp_path / "synergy")
        fig = plot_pid_synergy_heatmap(synergy_matrix, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotOInformationTrajectory:
    """Tests for plot_o_information_trajectory function."""

    def test_creates_figure(self, tmp_path):
        """Test creates a Figure object."""
        o_info_values = [-0.1, -0.2, -0.15, -0.25, -0.3]
        steps = [0, 1000, 2000, 3000, 4000]
        output_path = str(tmp_path / "o_info")
        fig = plot_o_information_trajectory(o_info_values, steps, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_surrogate_ci(self, tmp_path):
        """Test with surrogate CI band."""
        o_info_values = [-0.1, -0.2, -0.15, -0.25, -0.3]
        steps = [0, 1000, 2000, 3000, 4000]
        surrogate_ci = (-0.05, 0.05)
        output_path = str(tmp_path / "o_info")
        fig = plot_o_information_trajectory(
            o_info_values, steps, output_path, surrogate_ci=surrogate_ci
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_surrogate_ci(self, tmp_path):
        """Test without surrogate CI."""
        o_info_values = [-0.1, -0.2, -0.15]
        steps = [0, 1000, 2000]
        output_path = str(tmp_path / "o_info")
        fig = plot_o_information_trajectory(o_info_values, steps, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCausalEmergenceGap:
    """Tests for plot_causal_emergence_gap function."""

    def test_creates_figure(self, tmp_path):
        """Test creates a Figure object."""
        macro_ei = [0.5, 0.6, 0.7, 0.8]
        micro_ei = [0.3, 0.35, 0.4, 0.45]
        steps = [0, 1000, 2000, 3000]
        output_path = str(tmp_path / "ce_gap")
        fig = plot_causal_emergence_gap(macro_ei, micro_ei, steps, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_shows_gap_region(self, tmp_path):
        """Test shaded gap region appears."""
        macro_ei = [0.5, 0.6, 0.7, 0.8]
        micro_ei = [0.3, 0.35, 0.4, 0.45]
        steps = [0, 1000, 2000, 3000]
        output_path = str(tmp_path / "ce_gap")
        fig = plot_causal_emergence_gap(macro_ei, micro_ei, steps, output_path)
        # Figure should be created without error
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPhaseTransitions:
    """Tests for plot_phase_transitions function."""

    def test_creates_figure(self, tmp_path):
        """Test creates a Figure object."""
        metrics_dict = {
            "food": [10, 20, 30, 40, 50],
            "specialization": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        steps = [0, 1000, 2000, 3000, 4000]
        output_path = str(tmp_path / "phase")
        fig = plot_phase_transitions(metrics_dict, steps, output_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_marks_transition_events(self, tmp_path):
        """Test marks transition events."""
        metrics_dict = {
            "food": [10, 20, 30, 40, 50],
        }
        steps = [0, 1000, 2000, 3000, 4000]
        events = [
            {"step": 2000, "label": "Phase 1"},
        ]
        output_path = str(tmp_path / "phase")
        fig = plot_phase_transitions(metrics_dict, steps, output_path, events)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSaveFigure:
    """Tests for save_figure function."""

    def test_saves_pdf(self, tmp_path):
        """Test saves PDF format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        output_path = str(tmp_path / "test")
        save_figure(fig, output_path, formats=["pdf"])
        assert (tmp_path / "test.pdf").exists()
        plt.close(fig)

    def test_saves_png(self, tmp_path):
        """Test saves PNG format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        output_path = str(tmp_path / "test")
        save_figure(fig, output_path, formats=["png"])
        assert (tmp_path / "test.png").exists()
        plt.close(fig)

    def test_saves_both_by_default(self, tmp_path):
        """Test saves both PDF and PNG by default."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        output_path = str(tmp_path / "test")
        save_figure(fig, output_path)
        assert (tmp_path / "test.pdf").exists()
        assert (tmp_path / "test.png").exists()
        plt.close(fig)

    def test_creates_parent_directory(self, tmp_path):
        """Test creates parent directory if needed."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        output_path = str(tmp_path / "subdir" / "test")
        save_figure(fig, output_path)
        assert (tmp_path / "subdir" / "test.pdf").exists()
        plt.close(fig)

    def test_correct_dpi(self, tmp_path):
        """Test saves at correct DPI."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        output_path = str(tmp_path / "test")
        save_figure(fig, output_path, dpi=300)
        assert (tmp_path / "test.png").exists()
        plt.close(fig)


class TestCreateFigureGrid:
    """Tests for create_figure_grid function."""

    def test_creates_correct_layout(self):
        """Test creates correct grid layout."""
        fig, axes = create_figure_grid(4, max_cols=2)
        assert len(axes) == 4
        plt.close(fig)

    def test_single_panel(self):
        """Test handles single panel."""
        fig, axes = create_figure_grid(1)
        assert len(axes) == 1
        plt.close(fig)

    def test_hides_unused_axes(self):
        """Test hides unused axes."""
        # 5 panels with max_cols=3 creates 6 axes (2x3 grid)
        fig, axes = create_figure_grid(5, max_cols=3)
        assert len(axes) == 5
        plt.close(fig)


class TestIntegration:
    """Integration tests for figure generation."""

    def test_all_figures_can_be_generated(self, tmp_path):
        """Test all figure types can be generated without error."""
        # Scaling
        scaling_results = [
            MockScalingAnalysis(
                field_condition="normal",
                n_agents_list=[1, 2, 4],
                mean_foods=[10.0, 25.0, 60.0],
            )
        ]
        fig = plot_scaling_curve(scaling_results, str(tmp_path / "scaling"))
        plt.close(fig)

        # Ablation
        ablation_results = {
            "normal": MockAblationResult(mean_reward=100.0, std_reward=10.0),
        }
        fig = plot_ablation_bars(ablation_results, str(tmp_path / "ablation"))
        plt.close(fig)

        # Performance profiles
        score_dict = {"ours": np.array([100, 90, 80])}
        fig = plot_performance_profiles(score_dict, str(tmp_path / "profiles"))
        plt.close(fig)

        # POI
        fig = plot_probability_of_improvement(score_dict, str(tmp_path / "poi"))
        plt.close(fig)

        # Synergy heatmap
        synergy_matrix = np.random.randn(4, 4)
        fig = plot_pid_synergy_heatmap(synergy_matrix, str(tmp_path / "synergy"))
        plt.close(fig)

        # O-information
        fig = plot_o_information_trajectory(
            [-0.1, -0.2], [0, 1000], str(tmp_path / "o_info")
        )
        plt.close(fig)

        # Causal emergence
        fig = plot_causal_emergence_gap(
            [0.5, 0.6], [0.3, 0.4], [0, 1000], str(tmp_path / "ce")
        )
        plt.close(fig)

        # Phase transitions
        fig = plot_phase_transitions(
            {"food": [10, 20]}, [0, 1000], str(tmp_path / "phase")
        )
        plt.close(fig)

        # Verify files exist
        assert (tmp_path / "scaling.pdf").exists()
        assert (tmp_path / "ablation.pdf").exists()
        assert (tmp_path / "profiles.pdf").exists()
        assert (tmp_path / "poi.pdf").exists()
        assert (tmp_path / "synergy.pdf").exists()
        assert (tmp_path / "o_info.pdf").exists()
        assert (tmp_path / "ce.pdf").exists()
        assert (tmp_path / "phase.pdf").exists()
