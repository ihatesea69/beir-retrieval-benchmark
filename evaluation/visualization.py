"""
Visualization for IR Evaluation
Generates precision-recall curves, metric distributions, and system comparisons
"""

from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

STANDARD_RECALL_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class EvaluationVisualizer:
    """
    Visualization utilities for IR evaluation results
    """

    @staticmethod
    def plot_pr_curves(
        system_curves: Dict[str, Dict[float, float]],
        title: str = "Precision-Recall Curves",
        save_path: Optional[str] = None
    ):
        """
        Plot interpolated PR curves for multiple systems

        Args:
            system_curves: Dict mapping system name to 11-point interpolated PR dict
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        for system_name, curve in system_curves.items():
            recalls = sorted(curve.keys())
            precisions = [curve[r] for r in recalls]
            ax.plot(recalls, precisions, marker='o', label=system_name)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved PR curve plot to {save_path}")

        return fig

    @staticmethod
    def plot_metric_distribution(
        system_scores: Dict[str, List[float]],
        metric_name: str = "NDCG@10",
        save_path: Optional[str] = None
    ):
        """
        Plot metric score distributions across queries for multiple systems

        Args:
            system_scores: Dict mapping system name to per-query scores
            metric_name: Name of the metric being plotted
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        data = [scores for scores in system_scores.values()]
        labels = list(system_scores.keys())

        ax.boxplot(data, labels=labels)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Distribution by System")
        ax.grid(True, alpha=0.3, axis='y')

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved distribution plot to {save_path}")

        return fig

    @staticmethod
    def plot_system_comparison(
        comparison_data: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot side-by-side bar chart comparing systems across metrics

        Args:
            comparison_data: Dict mapping system name to dict of metric -> score
            metrics: List of metric names to include
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import numpy as np

        systems = list(comparison_data.keys())
        n_systems = len(systems)
        n_metrics = len(metrics)

        x = np.arange(n_metrics)
        width = 0.8 / n_systems

        fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 5))

        for i, system in enumerate(systems):
            scores = [comparison_data[system].get(m, 0.0) for m in metrics]
            offset = (i - n_systems / 2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=system)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15, ha='right')
        ax.set_ylabel("Score")
        ax.set_title("System Comparison")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved comparison plot to {save_path}")

        return fig
