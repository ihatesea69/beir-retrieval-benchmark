"""
Advanced IR Evaluation Metrics
Implements graded relevance evaluation and robust metrics for incomplete judgments
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradedQrels:
    """Data structure for graded relevance judgments"""
    query_id: str
    judgments: Dict[str, int]  # doc_id -> relevance_score (0-4)
    total_relevant: int
    relevance_levels: Dict[int, int]  # level -> count


class GradedRelevanceEvaluator:
    """
    Evaluator for graded relevance judgments (0-4 scale)
    Extends basic binary evaluation with multi-level relevance support
    """
    
    @staticmethod
    def calculate_graded_ndcg(
        retrieved: List[str], 
        qrels: Dict[str, int], 
        k: int = 10
    ) -> float:
        """
        Calculate NDCG@k using graded relevance scores as gain values
        
        Args:
            retrieved: List of retrieved document IDs in rank order
            qrels: Dict mapping doc_id to relevance score (0-4)
            k: Cutoff for evaluation
            
        Returns:
            NDCG@k score using graded relevance
        """
        if not retrieved or not qrels:
            return 0.0
            
        # Calculate DCG using actual relevance scores as gains
        dcg = GradedRelevanceEvaluator.calculate_graded_dcg(retrieved, qrels, k)
        
        # Calculate IDCG by sorting relevance scores in descending order
        relevance_scores = list(qrels.values())
        relevance_scores.sort(reverse=True)
        
        ideal_retrieved = [f"ideal_{i}" for i in range(len(relevance_scores))]
        ideal_qrels = {f"ideal_{i}": score for i, score in enumerate(relevance_scores)}
        
        idcg = GradedRelevanceEvaluator.calculate_graded_dcg(ideal_retrieved, ideal_qrels, k)
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    @staticmethod
    def calculate_graded_dcg(
        retrieved: List[str], 
        qrels: Dict[str, int], 
        k: int = 10
    ) -> float:
        """
        Calculate DCG@k using graded relevance scores
        
        Args:
            retrieved: List of retrieved document IDs in rank order
            qrels: Dict mapping doc_id to relevance score (0-4)
            k: Cutoff for evaluation
            
        Returns:
            DCG@k score
        """
        dcg = 0.0
        retrieved_k = retrieved[:k]
        
        for i, doc_id in enumerate(retrieved_k, 1):
            relevance = qrels.get(doc_id, 0)
            # Use actual relevance score as gain (not binary 0/1)
            dcg += relevance / np.log2(i + 1)
            
        return dcg
    
    @staticmethod
    def convert_to_binary(qrels: Dict[str, int], threshold: int = 1) -> Dict[str, int]:
        """
        Convert graded relevance to binary relevance
        
        Args:
            qrels: Graded relevance judgments
            threshold: Minimum score to consider relevant
            
        Returns:
            Binary relevance judgments (0 or 1)
        """
        return {doc_id: 1 if score >= threshold else 0 for doc_id, score in qrels.items()}
    
    @staticmethod
    def validate_relevance_scale(qrels: Dict[str, int], min_score: int = 0, max_score: int = 4) -> bool:
        """
        Validate that relevance scores are within expected range
        
        Args:
            qrels: Relevance judgments to validate
            min_score: Minimum allowed relevance score
            max_score: Maximum allowed relevance score
            
        Returns:
            True if all scores are valid, False otherwise
        """
        for doc_id, score in qrels.items():
            if not isinstance(score, int) or score < min_score or score > max_score:
                logger.warning(f"Invalid relevance score {score} for document {doc_id}")
                return False
        return True


class RobustMetricsCalculator:
    """
    Calculator for robust metrics designed for incomplete relevance judgments
    Implements bpref and infAP for pooled evaluation scenarios
    """
    
    @staticmethod
    def calculate_bpref(
        retrieved: List[str], 
        relevant: List[str], 
        judged: Set[str]
    ) -> float:
        """
        Calculate bpref (binary preference) for incomplete judgments
        
        bpref measures the fraction of judged non-relevant documents 
        retrieved after each relevant document
        
        Args:
            retrieved: List of retrieved document IDs in rank order
            relevant: List of relevant document IDs
            judged: Set of all judged document IDs
            
        Returns:
            bpref score (0-1)
        """
        if not relevant:
            return 0.0
            
        relevant_set = set(relevant)
        judged_nonrel = judged - relevant_set
        
        if not judged_nonrel:
            # If no judged non-relevant docs, bpref = 1.0
            return 1.0
            
        bpref_sum = 0.0
        
        for rel_doc in relevant:
            if rel_doc not in retrieved:
                continue
                
            rel_rank = retrieved.index(rel_doc)
            
            # Count judged non-relevant docs retrieved before this relevant doc
            nonrel_before = 0
            for i in range(rel_rank):
                if retrieved[i] in judged_nonrel:
                    nonrel_before += 1
            
            # bpref contribution for this relevant doc
            bpref_sum += 1.0 - (nonrel_before / len(judged_nonrel))
        
        return bpref_sum / len(relevant)
    
    @staticmethod
    def calculate_inf_ap(
        retrieved: List[str], 
        qrels: Dict[str, int], 
        judged: Set[str]
    ) -> float:
        """
        Calculate inferred Average Precision for pooled evaluations
        
        Estimates AP by making assumptions about unjudged documents
        
        Args:
            retrieved: List of retrieved document IDs in rank order
            qrels: Relevance judgments (doc_id -> relevance)
            judged: Set of judged document IDs
            
        Returns:
            Inferred AP score
        """
        if not retrieved:
            return 0.0
            
        relevant_judged = [doc for doc, rel in qrels.items() if rel > 0]
        if not relevant_judged:
            return 0.0
            
        # Calculate AP on judged documents only
        precision_sum = 0.0
        relevant_found = 0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in judged:
                if qrels.get(doc_id, 0) > 0:
                    relevant_found += 1
                    precision_sum += relevant_found / i
        
        if relevant_found == 0:
            return 0.0
            
        return precision_sum / len(relevant_judged)
    
    @staticmethod
    def estimate_recall_base(
        relevant_found: int, 
        pool_depth: int, 
        total_retrieved: int
    ) -> int:
        """
        Estimate total number of relevant documents for recall calculation
        
        Args:
            relevant_found: Number of relevant documents found in pool
            pool_depth: Depth of evaluation pool
            total_retrieved: Total documents retrieved by system
            
        Returns:
            Estimated total relevant documents
        """
        if pool_depth == 0 or total_retrieved == 0:
            return relevant_found
            
        # Simple estimation: scale by retrieval coverage
        coverage_ratio = min(pool_depth / total_retrieved, 1.0)
        if coverage_ratio == 0:
            return relevant_found
            
        estimated_total = int(relevant_found / coverage_ratio)
        return max(estimated_total, relevant_found)


class PooledEvaluator:
    """
    Evaluation support for pooled relevance judgments
    Handles scenarios where not all documents are judged
    """

    @staticmethod
    def create_pool(
        system_results: Dict[str, List[str]],
        pool_depth: int = 100
    ) -> Set[str]:
        """
        Create evaluation pool from multiple systems' top-k results

        Args:
            system_results: Dict mapping system name to ranked doc IDs
            pool_depth: Number of top results to take from each system

        Returns:
            Set of unique document IDs in the pool
        """
        pool: Set[str] = set()
        for docs in system_results.values():
            pool.update(docs[:pool_depth])
        return pool

    @staticmethod
    def pool_statistics(
        pool: Set[str],
        qrels: Dict[str, int],
        system_results: Dict[str, List[str]],
        pool_depth: int
    ) -> Dict[str, any]:
        """
        Compute pool depth and coverage statistics

        Args:
            pool: Set of pooled document IDs
            qrels: Relevance judgments for the query
            system_results: Dict mapping system name to ranked doc IDs
            pool_depth: Pool depth used

        Returns:
            Dict with pool statistics
        """
        relevant_in_pool = sum(1 for d in pool if qrels.get(d, 0) > 0)
        total_relevant = sum(1 for s in qrels.values() if s > 0)

        return {
            'pool_size': len(pool),
            'pool_depth': pool_depth,
            'n_systems': len(system_results),
            'relevant_in_pool': relevant_in_pool,
            'total_relevant_known': total_relevant,
            'coverage': relevant_in_pool / total_relevant if total_relevant > 0 else 0.0
        }

    @staticmethod
    def compute_pooled_metrics(
        retrieved: List[str],
        qrels: Dict[str, int],
        pool: Set[str],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Compute metrics treating unjudged documents as non-relevant

        Args:
            retrieved: Ranked list of retrieved document IDs
            qrels: Relevance judgments (only pooled docs are judged)
            pool: Set of judged document IDs
            k: Cutoff for evaluation

        Returns:
            Dict with precision, recall, and bpref metrics
        """
        relevant = [d for d, s in qrels.items() if s > 0]
        judged = pool

        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)

        hits = sum(1 for d in retrieved_k if d in relevant_set)
        precision = hits / k if k > 0 else 0.0
        recall = hits / len(relevant_set) if relevant_set else 0.0

        bpref = RobustMetricsCalculator.calculate_bpref(retrieved, relevant, judged)

        return {
            f'Precision@{k}': precision,
            f'Recall@{k}': recall,
            'bpref': bpref
        }
