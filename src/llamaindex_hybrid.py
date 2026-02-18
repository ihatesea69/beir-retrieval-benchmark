"""
Module: LlamaIndex Hybrid Retriever (BM25 + Dense with RRF)
Chức năng: Hybrid search kết hợp BM25 và semantic search với RRF fusion
"""

import logging
from typing import TYPE_CHECKING, Dict, List
from tqdm import tqdm

if TYPE_CHECKING:
    from src.kb.metadata import MetadataFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaIndexHybrid:
    """
    Hybrid Retriever: BM25 + Dense Vector với Reciprocal Rank Fusion
    
    Công thức RRF:
    score(d) = α/(k+rank_bm25(d)) + (1-α)/(k+rank_dense(d))
    
    Ưu điểm:
    - Kết hợp lexical matching (BM25) và semantic matching (Dense)
    - RRF algorithm chuẩn từ literature
    - Phân tích nguồn gốc results (from_bm25, from_dense, from_both)
    """
    
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        alpha: float = 0.5,
        k: int = 60
    ):
        """
        Args:
            bm25_retriever: LlamaIndexBM25 instance
            dense_retriever: LlamaIndexRAG instance
            alpha: Trọng số BM25 (0.0 = chỉ dense, 1.0 = chỉ BM25)
            k: RRF constant (thường là 60)
        """
        logger.info(f"🔗 Initializing Hybrid Retriever (α={alpha}, k={k})")
        
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.k = k
        
        logger.info("✓ Hybrid Retriever initialized")
    
    def search(self, query: str, top_k: int = 10,
               metadata_filter: "MetadataFilter" = None) -> List[Dict]:
        """Hybrid search with RRF fusion and optional metadata filtering. Requirements: 4.1"""
        # Retrieve từ cả 2 systems với top_k=100 để có đủ candidates
        bm25_results = self.bm25.search(query, top_k=100)
        dense_results = self.dense.search(query, top_k=100)
        
        # RRF Fusion
        fused_scores = {}
        doc_info = {}  # Lưu title, text
        doc_sources = {}  # Track nguồn gốc
        
        # Process BM25 results
        for result in bm25_results:
            doc_id = result['id']
            rank = result['rank']
            
            # RRF score contribution từ BM25
            fused_scores[doc_id] = self.alpha / (self.k + rank)
            doc_info[doc_id] = {
                'title': result['title'],
                'text': result['text']
            }
            doc_sources[doc_id] = 'bm25'
        
        # Process Dense results
        for result in dense_results:
            doc_id = result['id']
            rank = result['rank']
            
            # RRF score contribution từ Dense
            dense_contribution = (1 - self.alpha) / (self.k + rank)
            
            if doc_id in fused_scores:
                fused_scores[doc_id] += dense_contribution
                doc_sources[doc_id] = 'both'  # Xuất hiện ở cả 2
            else:
                fused_scores[doc_id] = dense_contribution
                doc_info[doc_id] = {
                    'title': result['title'],
                    'text': result['text']
                }
                doc_sources[doc_id] = 'dense'
        
        # Sort by fused score
        sorted_docs = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            results.append({
                'id': doc_id,
                'score': score,
                'title': doc_info[doc_id]['title'],
                'text': doc_info[doc_id]['text'],
                'rank': rank,
                'source': doc_sources[doc_id]
            })

        # Apply metadata filter post-fusion
        if metadata_filter is not None:
            try:
                from src.kb.metadata import MetadataFilterEngine, DocumentMetadata
                from src.kb.knowledge_base import KnowledgeBase
                kb = KnowledgeBase()
                engine = MetadataFilterEngine()
                allowed_ids = set(engine.apply(metadata_filter, [r['id'] for r in results], kb))
                results = [r for r in results if r['id'] in allowed_ids]
                for i, r in enumerate(results, 1):
                    r['rank'] = i
            except ImportError:
                pass

        return results

    def batch_search(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        metadata_filter: "MetadataFilter" = None,
    ) -> Dict[str, List[Dict]]:
        """Batch hybrid search with optional metadata filtering. Requirements: 4.1"""
            Dict {query_id: results}
        """
        logger.info(f"Batch Hybrid Search: {len(queries)} queries...")
        results_dict = {}
        for qid, query_text in tqdm(queries.items(), desc="Hybrid Search"):
            results_dict[qid] = self.search(query_text, top_k=top_k,
                                            metadata_filter=metadata_filter)
        return results_dict
    
    def analyze_sources(self, results: List[Dict], top_k: int = 10) -> Dict:
        """
        Phân tích nguồn gốc của results
        
        Args:
            results: List results từ search()
            top_k: Phân tích top k results
            
        Returns:
            Dict thống kê nguồn
        """
        top_results = results[:top_k]
        
        stats = {
            'from_bm25': 0,
            'from_dense': 0,
            'from_both': 0
        }
        
        for result in top_results:
            source = result.get('source', 'unknown')
            if source == 'bm25':
                stats['from_bm25'] += 1
            elif source == 'dense':
                stats['from_dense'] += 1
            elif source == 'both':
                stats['from_both'] += 1
        
        return stats


# Demo
if __name__ == "__main__":
    from data_loader import BeirDataLoader
    from llamaindex_bm25 import LlamaIndexBM25
    from llamaindex_rag import LlamaIndexRAG
    
    print("\n=== DEMO: LlamaIndex Hybrid Retriever ===\n")
    
    # Load dataset
    loader = BeirDataLoader()
    corpus, queries, qrels = loader.load_dataset('nfcorpus')
    documents = loader.prepare_corpus_for_indexing(corpus)
    
    # Initialize retrievers
    print("Building BM25 index...")
    bm25 = LlamaIndexBM25()
    bm25.build_index(documents[:100])
    
    print("\nBuilding Dense index...")
    rag = LlamaIndexRAG()
    rag.build_index(documents[:100])
    
    # Create hybrid
    hybrid = LlamaIndexHybrid(
        bm25_retriever=bm25,
        dense_retriever=rag,
        alpha=0.5,
        k=60
    )
    
    # Test query
    test_query = "What causes diabetes?"
    print(f"\nQuery: {test_query}\n")
    
    results = hybrid.search(test_query, top_k=10)
    
    print("=== Top-10 Hybrid Results ===")
    for doc in results:
        print(f"[Rank {doc['rank']}] Score: {doc['score']:.4f} | Source: {doc['source']}")
        print(f"Title: {doc['title']}")
        print(f"Text: {doc['text'][:150]}...\n")
    
    # Analyze sources
    stats = hybrid.analyze_sources(results, top_k=10)
    print("\n=== Source Analysis (Top-10) ===")
    print(f"From BM25 only: {stats['from_bm25']}")
    print(f"From Dense only: {stats['from_dense']}")
    print(f"From Both: {stats['from_both']}")
