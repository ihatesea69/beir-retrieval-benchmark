"""
Benchmark Dataset Support
Utilities for loading standard IR test collections (TREC, BEIR)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from evaluation.trec_format import TRECFormatHandler

logger = logging.getLogger(__name__)


class BenchmarkLoader:
    """
    Loader for standard IR benchmark datasets
    Supports TREC format and BEIR benchmark collections
    """

    @staticmethod
    def load_beir_corpus(corpus_path: str) -> Dict[str, Dict]:
        """
        Load BEIR corpus.jsonl file

        Format: {"_id": "...", "title": "...", "text": "..."}

        Args:
            corpus_path: Path to corpus.jsonl

        Returns:
            Dict mapping doc_id to document dict
        """
        corpus = {}
        path = Path(corpus_path)

        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = doc.get('_id') or doc.get('id')
                if doc_id:
                    corpus[str(doc_id)] = doc

        logger.info(f"Loaded {len(corpus)} documents from {corpus_path}")
        return corpus

    @staticmethod
    def load_beir_queries(queries_path: str) -> Dict[str, str]:
        """
        Load BEIR queries.jsonl file

        Format: {"_id": "...", "text": "..."}

        Args:
            queries_path: Path to queries.jsonl

        Returns:
            Dict mapping query_id to query text
        """
        queries = {}
        path = Path(queries_path)

        if not path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                query = json.loads(line)
                query_id = query.get('_id') or query.get('id')
                text = query.get('text') or query.get('query', '')
                if query_id:
                    queries[str(query_id)] = text

        logger.info(f"Loaded {len(queries)} queries from {queries_path}")
        return queries

    @staticmethod
    def load_beir_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
        """
        Load BEIR qrels TSV file

        Format: query-id\tcorpus-id\tscore

        Args:
            qrels_path: Path to qrels TSV file

        Returns:
            Dict mapping query_id to {doc_id: relevance}
        """
        qrels = {}
        path = Path(qrels_path)

        if not path.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header line
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 3:
                    continue
                query_id, doc_id, score = parts
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(score)

        logger.info(f"Loaded qrels for {len(qrels)} queries from {qrels_path}")
        return qrels

    @staticmethod
    def load_beir_dataset(dataset_dir: str) -> Tuple[Dict, Dict, Dict]:
        """
        Load a complete BEIR dataset from directory

        Expected structure:
            dataset_dir/
                corpus.jsonl
                queries.jsonl
                qrels/test.tsv

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            Tuple of (corpus, queries, qrels)
        """
        base = Path(dataset_dir)

        corpus = BenchmarkLoader.load_beir_corpus(str(base / 'corpus.jsonl'))
        queries = BenchmarkLoader.load_beir_queries(str(base / 'queries.jsonl'))

        # Try common qrels locations
        qrels_candidates = [
            base / 'qrels' / 'test.tsv',
            base / 'qrels' / 'dev.tsv',
            base / 'qrels.tsv',
        ]
        qrels = {}
        for candidate in qrels_candidates:
            if candidate.exists():
                qrels = BenchmarkLoader.load_beir_qrels(str(candidate))
                break

        if not qrels:
            logger.warning(f"No qrels file found in {dataset_dir}")

        return corpus, queries, qrels

    @staticmethod
    def load_trec_dataset(
        qrels_path: str,
        topics_path: Optional[str] = None,
        results_path: Optional[str] = None
    ) -> Dict:
        """
        Load a TREC-format dataset

        Args:
            qrels_path: Path to TREC qrels file
            topics_path: Optional path to topics file
            results_path: Optional path to results file

        Returns:
            Dict with loaded data components
        """
        data = {}

        data['qrels'] = TRECFormatHandler.load_qrels(qrels_path)

        if topics_path and Path(topics_path).exists():
            data['topics'] = TRECFormatHandler.load_topics(topics_path)

        if results_path and Path(results_path).exists():
            data['results'] = TRECFormatHandler.load_results(results_path)

        return data

    @staticmethod
    def validate_dataset(
        corpus: Dict,
        queries: Dict,
        qrels: Dict
    ) -> List[str]:
        """
        Validate dataset consistency

        Args:
            corpus: Document corpus
            queries: Query set
            qrels: Relevance judgments

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Check queries in qrels exist in query set
        for qid in qrels:
            if qid not in queries:
                issues.append(f"Query {qid} in qrels but not in queries")

        # Check docs in qrels exist in corpus
        missing_docs = 0
        for qid, judgments in qrels.items():
            for doc_id in judgments:
                if doc_id not in corpus:
                    missing_docs += 1

        if missing_docs > 0:
            issues.append(f"{missing_docs} judged documents not found in corpus")

        if not issues:
            logger.info("Dataset validation passed")
        else:
            for issue in issues:
                logger.warning(f"Dataset issue: {issue}")

        return issues
