"""
LlamaIndex Experiment Runner: So sánh BM25 vs Dense vs Hybrid
Framework: LlamaIndex + PostgreSQL+pgvector
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import BeirDataLoader
from llamaindex_bm25 import LlamaIndexBM25
from llamaindex_rag import LlamaIndexRAG
from llamaindex_hybrid import LlamaIndexHybrid
from metrics import RetrievalEvaluator
import warnings
warnings.filterwarnings('ignore')

# Setup plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 10


class LlamaIndexExperiment:
    """
    Experiment Runner với LlamaIndex Framework
    
    So sánh:
    - BM25 (Sparse/Lexical)
    - Dense (Semantic/Vector)
    - Hybrid (BM25 + Dense với RRF)
    """
    
    def __init__(self, dataset_name: str = 'nfcorpus', n_queries: int = 50):
        """
        Args:
            dataset_name: Tên dataset BeIR
            n_queries: Số queries để test
        """
        self.dataset_name = dataset_name
        self.n_queries = n_queries
        
        print(f"\n{'='*80}")
        print(f"🚀 LLAMAINDEX EXPERIMENT: {dataset_name.upper()}")
        print(f"Framework: LlamaIndex + PostgreSQL+pgvector")
        print(f"Comparing: BM25 | Dense | Hybrid (RRF)")
        print(f"{'='*80}\n")
        
        # Load data
        self.loader = BeirDataLoader()
        self.corpus, self.queries, self.qrels = self.loader.load_dataset(dataset_name)
        self.test_queries = self.loader.get_sample_queries(self.queries, n=n_queries)
        self.documents = self.loader.prepare_corpus_for_indexing(self.corpus)
        
        print(f"✓ Dataset: {len(self.corpus)} docs, {len(self.test_queries)} test queries")
        
        # Evaluator
        self.evaluator = RetrievalEvaluator(self.qrels)
    
    def run_bm25(self) -> pd.DataFrame:
        """Run BM25 retrieval"""
        print(f"\n{'='*80}")
        print("📊 PIPELINE A: BM25 (LlamaIndex + Docstore Persistence)")
        print('='*80)
        
        persist_dir = f"./storage/bm25_{self.dataset_name}"
        
        # Check if already persisted
        if os.path.exists(os.path.join(persist_dir, "docstore.json")):
            print(f"\n✅ Found existing BM25 index at {persist_dir}")
            print("Loading from disk...\n")
            bm25 = LlamaIndexBM25.from_persist_dir(persist_dir)
        else:
            print(f"\n🔨 Building new BM25 index...")
            # Build index
            bm25 = LlamaIndexBM25(persist_dir=persist_dir)
            bm25.build_index(self.documents)
            
            # Save to disk
            print("\n💾 Saving BM25 index...")
            bm25.persist()
            print(f"✓ Saved to {persist_dir}\n")
        
        # Batch search
        results = bm25.batch_search(self.test_queries, top_k=100)
        
        # Evaluate
        metrics = self.evaluator.evaluate_batch(results)
        
        return metrics
    
    def run_dense(self) -> pd.DataFrame:
        """Run Dense retrieval"""
        print(f"\n{'='*80}")
        print("🤖 PIPELINE B: DENSE (LlamaIndex + PostgreSQL)")
        print('='*80)
        
        # Build index
        rag = LlamaIndexRAG(table_name="llamaindex_nfcorpus")
        rag.build_index(self.documents)
        
        # Batch search
        results = rag.batch_search(self.test_queries, top_k=100)
        
        # Evaluate
        metrics = self.evaluator.evaluate_batch(results)
        
        return rag, metrics
    
    def run_hybrid(self, bm25, dense) -> tuple:
        """Run Hybrid retrieval"""
        print(f"\n{'='*80}")
        print("🔗 PIPELINE C: HYBRID (BM25 + Dense + RRF)")
        print('='*80)
        
        # Create hybrid
        hybrid = LlamaIndexHybrid(
            bm25_retriever=bm25,
            dense_retriever=dense,
            alpha=0.5,
            k=60
        )
        
        # Batch search
        results = hybrid.batch_search(self.test_queries, top_k=100)
        
        # Evaluate
        metrics = self.evaluator.evaluate_batch(results)
        
        # Source analysis
        all_sources = {'from_bm25': 0, 'from_dense': 0, 'from_both': 0}
        for qid, result_list in results.items():
            stats = hybrid.analyze_sources(result_list, top_k=10)
            for key in all_sources:
                all_sources[key] += stats[key]
        
        # Average
        n_queries = len(results)
        avg_sources = {k: v/n_queries for k, v in all_sources.items()}
        
        print("\n=== Hybrid Source Analysis (Top-10) ===")
        print(f"Avg docs from both: {avg_sources['from_both']:.1f}/10")
        print(f"Avg docs from BM25 only: {avg_sources['from_bm25']:.1f}/10")
        print(f"Avg docs from Dense only: {avg_sources['from_dense']:.1f}/10")
        
        return metrics, avg_sources
    
    def run_all(self):
        """Run tất cả pipelines"""
        # BM25
        print("\n🔨 Building BM25...")
        bm25 = LlamaIndexBM25()
        bm25.build_index(self.documents)
        df_bm25 = self.run_bm25()
        
        # Dense
        print("\n🔨 Building Dense (PostgreSQL)...")
        rag, df_dense = self.run_dense()
        
        # Hybrid
        df_hybrid, hybrid_stats = self.run_hybrid(bm25, rag)
        
        # Compare
        self.compare(df_bm25, df_dense, df_hybrid)
        
        # Save
        self.save_results(df_bm25, df_dense, df_hybrid)
        
        return df_bm25, df_dense, df_hybrid, hybrid_stats
    
    def compare(self, df_bm25, df_dense, df_hybrid):
        """So sánh 3 pipelines"""
        print(f"\n{'='*80}")
        print("📈 COMPARISON: BM25 vs Dense vs Hybrid")
        print('='*80)
        
        # Key metrics
        metrics_to_compare = ['NDCG@10', 'Recall@100', 'MAP', 'MRR', 'Precision@10']
        
        comparison_data = []
        for metric in metrics_to_compare:
            bm25_val = df_bm25[metric].mean()
            dense_val = df_dense[metric].mean()
            hybrid_val = df_hybrid[metric].mean()
            
            # Tìm winner
            best_val = max(bm25_val, dense_val, hybrid_val)
            if best_val == bm25_val:
                winner = 'BM25'
            elif best_val == dense_val:
                winner = 'Dense'
            else:
                winner = 'Hybrid'
            
            comparison_data.append({
                'Metric': metric,
                'BM25': bm25_val,
                'Dense': dense_val,
                'Hybrid': hybrid_val,
                'Winner': winner
            })
        
        df_comp = pd.DataFrame(comparison_data)
        print("\n", df_comp.to_string(index=False))
        
        # Win count
        print("\n=== Win Count ===")
        win_counts = df_comp['Winner'].value_counts()
        for system, count in win_counts.items():
            print(f"{system}: {count} metrics")
        
        return df_comp
    
    def save_results(self, df_bm25, df_dense, df_hybrid):
        """Lưu kết quả"""
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        df_bm25.to_csv(f"{results_dir}/llamaindex_bm25_{self.dataset_name}.csv", index=False)
        df_dense.to_csv(f"{results_dir}/llamaindex_dense_{self.dataset_name}.csv", index=False)
        df_hybrid.to_csv(f"{results_dir}/llamaindex_hybrid_{self.dataset_name}.csv", index=False)
        
        print(f"\n✓ Saved results to {results_dir}/")


# Main
if __name__ == "__main__":
    # Run experiment
    runner = LlamaIndexExperiment(dataset_name='nfcorpus', n_queries=50)
    
    df_bm25, df_dense, df_hybrid, hybrid_stats = runner.run_all()
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETED")
    print("="*80)
