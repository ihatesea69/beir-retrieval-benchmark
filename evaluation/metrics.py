"""
Module: Evaluation Metrics
Chức năng: Đánh giá hiệu năng của 2 pipelines (BM25 vs RAG)
Metrics: 
- Retrieval: NDCG@10, Recall@100, MAP, MRR
- Generation: Faithfulness, Answer Relevance (using Ragas)
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Đánh giá chất lượng Retrieval (áp dụng cho cả BM25 và Dense Retrieval)
    """
    
    @staticmethod
    def calculate_ndcg_at_k(
        retrieved: List[str], 
        relevant: List[str], 
        k: int = 10
    ) -> float:
        """
        Tính NDCG@k (Normalized Discounted Cumulative Gain)
        
        NDCG đo lường chất lượng ranking của kết quả tìm kiếm.
        Score càng cao = Tài liệu đúng càng nằm ở vị trí cao.
        
        Args:
            retrieved: List doc IDs đã retrieve (theo thứ tự rank)
            relevant: List doc IDs đúng (ground truth)
            k: Đánh giá top-k kết quả
            
        Returns:
            NDCG score (0-1, càng cao càng tốt)
        """
        # Lấy top-k retrieved docs
        retrieved_k = retrieved[:k]
        
        # Tính DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant:
                # rel = 1 nếu document đúng, 0 nếu sai
                rel = 1
                # Discount factor: log2(i+1)
                dcg += rel / np.log2(i + 1)
        
        # Tính IDCG (Ideal DCG) - DCG tốt nhất có thể
        ideal_k = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
        
        # NDCG = DCG / IDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def calculate_recall_at_k(
        retrieved: List[str], 
        relevant: List[str], 
        k: int = 100
    ) -> float:
        """
        Tính Recall@k
        
        Recall đo lường tỷ lệ tài liệu đúng được tìm thấy.
        
        Args:
            retrieved: List doc IDs đã retrieve
            relevant: List doc IDs đúng
            k: Đánh giá top-k kết quả
            
        Returns:
            Recall score (0-1)
        """
        if not relevant:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        # Số lượng docs đúng được tìm thấy / Tổng số docs đúng
        hits = len(retrieved_k & relevant_set)
        return hits / len(relevant_set)
    
    @staticmethod
    def calculate_precision_at_k(
        retrieved: List[str], 
        relevant: List[str], 
        k: int = 10
    ) -> float:
        """
        Tính Precision@k
        
        Precision đo lường tỷ lệ tài liệu đúng trong kết quả trả về.
        
        Args:
            retrieved: List doc IDs đã retrieve
            relevant: List doc IDs đúng
            k: Đánh giá top-k kết quả
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        
        # Số lượng docs đúng trong top-k / k
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        return hits / len(retrieved_k)
    
    @staticmethod
    def calculate_map(
        retrieved: List[str], 
        relevant: List[str]
    ) -> float:
        """
        Tính MAP (Mean Average Precision)
        
        MAP tính trung bình precision tại mỗi vị trí có document đúng.
        
        Args:
            retrieved: List doc IDs đã retrieve
            relevant: List doc IDs đúng
            
        Returns:
            MAP score (0-1)
        """
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        precisions = []
        num_hits = 0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                num_hits += 1
                precision = num_hits / i
                precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant)
    
    @staticmethod
    def calculate_mrr(
        retrieved: List[str], 
        relevant: List[str]
    ) -> float:
        """
        Tính MRR (Mean Reciprocal Rank)
        
        MRR = 1 / rank của document đúng đầu tiên
        
        Args:
            retrieved: List doc IDs đã retrieve
            relevant: List doc IDs đúng
            
        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant)
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def evaluate_retrieval(
        results: Dict[str, List[Dict]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int] = [3, 5, 10, 100]
    ) -> pd.DataFrame:
        """
        Đánh giá kết quả retrieval cho nhiều queries
        
        Args:
            results: Dict {query_id: [retrieved_docs]}
            qrels: Ground truth {query_id: {doc_id: relevance}}
            k_values: List các k để đánh giá
            
        Returns:
            DataFrame chứa metrics cho từng query
        """
        logger.info(f"Đang đánh giá {len(results)} queries...")
        
        evaluation_data = []
        
        for query_id, retrieved_docs in tqdm(results.items(), desc="Evaluating"):
            # Lấy doc IDs đã retrieve
            retrieved_ids = [doc['id'] for doc in retrieved_docs]
            
            # Lấy ground truth
            if query_id not in qrels:
                continue
            
            relevant_ids = [
                doc_id for doc_id, score in qrels[query_id].items() 
                if score > 0
            ]
            
            # Tính metrics
            row = {'query_id': query_id}
            
            # NDCG và Recall cho các k khác nhau
            for k in k_values:
                ndcg = RetrievalEvaluator.calculate_ndcg_at_k(
                    retrieved_ids, relevant_ids, k
                )
                recall = RetrievalEvaluator.calculate_recall_at_k(
                    retrieved_ids, relevant_ids, k
                )
                precision = RetrievalEvaluator.calculate_precision_at_k(
                    retrieved_ids, relevant_ids, k
                )
                
                row[f'NDCG@{k}'] = ndcg
                row[f'Recall@{k}'] = recall
                row[f'Precision@{k}'] = precision
            
            # MAP và MRR
            row['MAP'] = RetrievalEvaluator.calculate_map(retrieved_ids, relevant_ids)
            row['MRR'] = RetrievalEvaluator.calculate_mrr(retrieved_ids, relevant_ids)
            
            evaluation_data.append(row)
        
        df = pd.DataFrame(evaluation_data)
        
        # Tính trung bình
        logger.info("\n=== Kết quả trung bình ===")
        for col in df.columns:
            if col != 'query_id':
                logger.info(f"{col}: {df[col].mean():.4f}")
        
        return df


class GenerationEvaluator:
    """
    Đánh giá chất lượng Generation (chỉ cho RAG pipeline)
    Sử dụng Ragas framework
    """
    
    def __init__(self):
        """Initialize Ragas evaluator"""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            
            self.evaluate = evaluate
            self.metrics = {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall
            }
            self.available = True
            logger.info("✓ Ragas evaluator đã sẵn sàng")
            
        except ImportError:
            logger.warning("⚠ Ragas chưa được cài đặt. Generation evaluation sẽ bị giới hạn.")
            self.available = False
    
    def evaluate_generation(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> pd.DataFrame:
        """
        Đánh giá chất lượng generation sử dụng Ragas
        
        Args:
            questions: List các câu hỏi
            answers: List các câu trả lời từ RAG
            contexts: List các contexts (retrieved docs) cho mỗi câu hỏi
            ground_truths: List câu trả lời đúng (optional)
            
        Returns:
            DataFrame chứa các metrics
        """
        if not self.available:
            logger.error("Ragas không khả dụng. Không thể đánh giá generation.")
            return pd.DataFrame()
        
        from datasets import Dataset
        
        # Chuẩn bị data cho Ragas
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts
        }
        
        if ground_truths:
            data['ground_truth'] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Chọn metrics (không dùng context_recall nếu không có ground_truth)
        metrics_to_use = [
            self.metrics['faithfulness'],
            self.metrics['answer_relevancy']
        ]
        
        if ground_truths:
            metrics_to_use.append(self.metrics['context_recall'])
        
        logger.info("Đang chạy Ragas evaluation...")
        
        try:
            # Run evaluation
            result = self.evaluate(
                dataset,
                metrics=metrics_to_use
            )
            
            # Convert to DataFrame
            df = result.to_pandas()
            
            logger.info("\n=== Generation Metrics ===")
            for col in df.columns:
                if col not in ['question', 'answer', 'contexts', 'ground_truth']:
                    logger.info(f"{col}: {df[col].mean():.4f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Lỗi khi chạy Ragas: {e}")
            return pd.DataFrame()
    
    def simple_faithfulness_check(
        self,
        answer: str,
        context_docs: List[str]
    ) -> Dict[str, any]:
        """
        Kiểm tra faithfulness đơn giản (không dùng Ragas)
        
        Kiểm tra xem answer có chứa thông tin không có trong context không
        
        Args:
            answer: Câu trả lời
            context_docs: Các documents làm context
            
        Returns:
            Dict chứa kết quả phân tích
        """
        # Combine all context
        full_context = " ".join(context_docs).lower()
        answer_lower = answer.lower()
        
        # Simple word overlap check
        answer_words = set(answer_lower.split())
        context_words = set(full_context.split())
        
        overlap = answer_words & context_words
        overlap_ratio = len(overlap) / len(answer_words) if answer_words else 0
        
        return {
            'overlap_ratio': overlap_ratio,
            'is_faithful': overlap_ratio > 0.5,  # Threshold
            'answer_length': len(answer.split()),
            'context_length': len(full_context.split())
        }


# Example usage
if __name__ == "__main__":
    # Demo NDCG calculation
    print("=== DEMO: NDCG Calculation ===\n")
    
    retrieved = ['doc1', 'doc5', 'doc2', 'doc8', 'doc10']
    relevant = ['doc2', 'doc5', 'doc7']
    
    ndcg = RetrievalEvaluator.calculate_ndcg_at_k(retrieved, relevant, k=5)
    recall = RetrievalEvaluator.calculate_recall_at_k(retrieved, relevant, k=5)
    precision = RetrievalEvaluator.calculate_precision_at_k(retrieved, relevant, k=5)
    
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    print(f"\nNDCG@5: {ndcg:.4f}")
    print(f"Recall@5: {recall:.4f}")
    print(f"Precision@5: {precision:.4f}")
