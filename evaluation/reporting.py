"""
Comprehensive Reporting for IR Evaluation
Generates summary tables, per-query analysis, and multi-format exports
"""

import json
import csv
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """
    Generates comprehensive evaluation reports in multiple formats
    """

    @staticmethod
    def generate_summary_table(
        results_df: pd.DataFrame,
        system_name: str = "System"
    ) -> pd.DataFrame:
        """
        Generate summary table with mean metrics across all queries

        Args:
            results_df: DataFrame with per-query metrics
            system_name: Name of the system being evaluated

        Returns:
            Summary DataFrame with mean values
        """
        metric_cols = [c for c in results_df.columns if c != 'query_id']
        summary = results_df[metric_cols].mean().to_frame(name='mean')
        summary['std'] = results_df[metric_cols].std()
        summary['min'] = results_df[metric_cols].min()
        summary['max'] = results_df[metric_cols].max()
        summary.index.name = 'metric'
        summary['system'] = system_name
        return summary

    @staticmethod
    def identify_best_worst_queries(
        results_df: pd.DataFrame,
        metric: str,
        n: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Identify queries with highest and lowest performance

        Args:
            results_df: DataFrame with per-query metrics
            metric: Metric column to rank by
            n: Number of top/bottom queries to return

        Returns:
            Dict with 'best' and 'worst' DataFrames
        """
        if metric not in results_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        sorted_df = results_df.sort_values(metric, ascending=False)

        return {
            'best': sorted_df.head(n).reset_index(drop=True),
            'worst': sorted_df.tail(n).reset_index(drop=True)
        }

    @staticmethod
    def generate_comparison_table(
        system_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate side-by-side comparison table for multiple systems

        Args:
            system_results: Dict mapping system name to results DataFrame

        Returns:
            Comparison DataFrame with systems as columns
        """
        summaries = {}
        for system_name, df in system_results.items():
            metric_cols = [c for c in df.columns if c != 'query_id']
            summaries[system_name] = df[metric_cols].mean()

        return pd.DataFrame(summaries)

    @staticmethod
    def export_csv(df: pd.DataFrame, filepath: str) -> None:
        """Export DataFrame to CSV"""
        df.to_csv(filepath, index=True)
        logger.info(f"Exported CSV to {filepath}")

    @staticmethod
    def export_json(data: dict, filepath: str) -> None:
        """Export data to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported JSON to {filepath}")

    @staticmethod
    def export_latex(df: pd.DataFrame, filepath: str, caption: str = "Evaluation Results") -> None:
        """
        Export DataFrame as LaTeX table

        Args:
            df: DataFrame to export
            filepath: Output .tex file path
            caption: Table caption
        """
        latex = df.to_latex(
            float_format="%.4f",
            caption=caption,
            label="tab:results",
            escape=True
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex)
        logger.info(f"Exported LaTeX table to {filepath}")

    @staticmethod
    def export_all_formats(
        df: pd.DataFrame,
        base_path: str,
        caption: str = "Evaluation Results"
    ) -> None:
        """
        Export results in CSV, JSON, and LaTeX formats

        Args:
            df: DataFrame to export
            base_path: Base file path without extension
            caption: Caption for LaTeX table
        """
        EvaluationReporter.export_csv(df, f"{base_path}.csv")
        EvaluationReporter.export_json(df.to_dict(), f"{base_path}.json")
        EvaluationReporter.export_latex(df, f"{base_path}.tex", caption)
