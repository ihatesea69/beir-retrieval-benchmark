"""
Integration Tests for IR Evaluation Pipeline
Tests complete evaluation workflows end-to-end
"""

import pytest
import os
import json
import tempfile
import pandas as pd

from evaluation.pipeline import EvaluationPipeline
from evaluation.benchmarks import BenchmarkLoader
from evaluation.reporting import EvaluationReporter
from evaluation.visualization import EvaluationVisualizer


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_qrels():
    return {
        'q1': {'d1': 2, 'd2': 1, 'd3': 0, 'd4': 2},
        'q2': {'d5': 1, 'd6': 0, 'd7': 2},
        'q3': {'d8': 1, 'd9': 1, 'd10': 0},
    }


@pytest.fixture
def system_a_results():
    return {
        'q1': [{'id': 'd1', 'score': 0.9}, {'id': 'd2', 'score': 0.8},
               {'id': 'd3', 'score': 0.7}, {'id': 'd4', 'score': 0.6}],
        'q2': [{'id': 'd5', 'score': 0.9}, {'id': 'd6', 'score': 0.7},
               {'id': 'd7', 'score': 0.5}],
        'q3': [{'id': 'd8', 'score': 0.8}, {'id': 'd9', 'score': 0.6},
               {'id': 'd10', 'score': 0.4}],
    }


@pytest.fixture
def system_b_results():
    return {
        'q1': [{'id': 'd3', 'score': 0.9}, {'id': 'd1', 'score': 0.7},
               {'id': 'd4', 'score': 0.5}, {'id': 'd2', 'score': 0.3}],
        'q2': [{'id': 'd6', 'score': 0.9}, {'id': 'd5', 'score': 0.6},
               {'id': 'd7', 'score': 0.4}],
        'q3': [{'id': 'd10', 'score': 0.8}, {'id': 'd8', 'score': 0.5},
               {'id': 'd9', 'score': 0.3}],
    }


# ─── Pipeline Integration ─────────────────────────────────────────────────────

class TestEvaluationPipeline:

    def test_evaluate_single_system_returns_dataframe(self, system_a_results, sample_qrels):
        pipeline = EvaluationPipeline(k_values=[3, 10])
        df = pipeline.evaluate_system(system_a_results, sample_qrels, "SystemA")

        assert isinstance(df, pd.DataFrame)
        assert 'query_id' in df.columns
        assert 'NDCG@10' in df.columns
        assert len(df) == 3  # 3 queries

    def test_evaluate_with_graded_relevance(self, system_a_results, sample_qrels):
        pipeline = EvaluationPipeline(k_values=[10], graded=True)
        df = pipeline.evaluate_system(system_a_results, sample_qrels, "SystemA")

        assert 'GradedNDCG@10' in df.columns
        assert 'NDCG@10' in df.columns  # Binary NDCG still present

    def test_compare_two_systems(self, system_a_results, system_b_results, sample_qrels):
        pipeline = EvaluationPipeline(k_values=[10])
        comparison = pipeline.compare_systems(
            {'A': system_a_results, 'B': system_b_results},
            sample_qrels,
            metric='NDCG@10'
        )

        assert 'system_results' in comparison
        assert 'comparison_table' in comparison
        assert 'A' in comparison['system_results']
        assert 'B' in comparison['system_results']

    def test_pr_analysis_returns_curves(self, system_a_results, sample_qrels):
        pipeline = EvaluationPipeline()
        pr_result = pipeline.pr_analysis(system_a_results, sample_qrels)

        assert 'per_query' in pr_result
        assert 'averaged_11_point' in pr_result
        assert len(pr_result['averaged_11_point']) == 11


# ─── Benchmark Loader Integration ─────────────────────────────────────────────

class TestBenchmarkLoader:

    def test_load_beir_corpus(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"_id": "d1", "title": "Test", "text": "Content"}\n')
            f.write('{"_id": "d2", "title": "Test2", "text": "Content2"}\n')
            path = f.name
        try:
            corpus = BenchmarkLoader.load_beir_corpus(path)
            assert 'd1' in corpus
            assert corpus['d1']['title'] == 'Test'
        finally:
            os.unlink(path)

    def test_load_beir_queries(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"_id": "q1", "text": "What is IR?"}\n')
            path = f.name
        try:
            queries = BenchmarkLoader.load_beir_queries(path)
            assert 'q1' in queries
            assert queries['q1'] == 'What is IR?'
        finally:
            os.unlink(path)

    def test_load_beir_qrels(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write('query-id\tcorpus-id\tscore\n')
            f.write('q1\td1\t2\n')
            f.write('q1\td2\t0\n')
            path = f.name
        try:
            qrels = BenchmarkLoader.load_beir_qrels(path)
            assert qrels['q1']['d1'] == 2
            assert qrels['q1']['d2'] == 0
        finally:
            os.unlink(path)

    def test_validate_dataset_detects_missing_docs(self):
        corpus = {'d1': {'text': 'doc1'}}
        queries = {'q1': 'query 1'}
        qrels = {'q1': {'d1': 1, 'd_missing': 2}}

        issues = BenchmarkLoader.validate_dataset(corpus, queries, qrels)
        assert any('not found in corpus' in issue for issue in issues)

    def test_validate_dataset_clean(self):
        corpus = {'d1': {'text': 'doc1'}, 'd2': {'text': 'doc2'}}
        queries = {'q1': 'query 1'}
        qrels = {'q1': {'d1': 1, 'd2': 0}}

        issues = BenchmarkLoader.validate_dataset(corpus, queries, qrels)
        assert len(issues) == 0


# ─── Full Workflow Integration ─────────────────────────────────────────────────

class TestFullWorkflow:

    def test_evaluate_export_workflow(self, system_a_results, sample_qrels, tmp_path):
        """Full workflow: evaluate → summarize → export"""
        pipeline = EvaluationPipeline(k_values=[3, 10])
        df = pipeline.evaluate_system(system_a_results, sample_qrels)

        summary = EvaluationReporter.generate_summary_table(df, "SystemA")
        assert 'NDCG@10' in summary.index

        csv_path = str(tmp_path / "results.csv")
        EvaluationReporter.export_csv(summary, csv_path)
        assert os.path.exists(csv_path)

    def test_comparison_with_pr_analysis(self, system_a_results, system_b_results, sample_qrels):
        """Compare systems and compute PR curves"""
        pipeline = EvaluationPipeline(k_values=[10])

        comparison = pipeline.compare_systems(
            {'A': system_a_results, 'B': system_b_results},
            sample_qrels
        )
        pr_a = pipeline.pr_analysis(system_a_results, sample_qrels)
        pr_b = pipeline.pr_analysis(system_b_results, sample_qrels)

        # Both systems should have valid averaged PR curves
        assert len(pr_a['averaged_11_point']) == 11
        assert len(pr_b['averaged_11_point']) == 11

        # Comparison table should have both systems
        assert 'A' in comparison['comparison_table'].columns
        assert 'B' in comparison['comparison_table'].columns
