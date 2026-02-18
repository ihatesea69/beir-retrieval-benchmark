"""
Module: LlamaIndex BM25 Retriever
Chức năng: BM25 sparse retrieval sử dụng LlamaIndex framework
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from tqdm import tqdm

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever

if TYPE_CHECKING:
    from src.kb.metadata import MetadataFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaIndexBM25:
    """
    BM25 Retriever sử dụng LlamaIndex
    
    Ưu điểm:
    - Unified interface với LlamaIndex ecosystem
    - Tự động tokenization và preprocessing
    - Compatible với hybrid search
    """
    
    def __init__(self, language: str = "english", persist_dir: Optional[str] = None):
        """
        Args:
            language: Ngôn ngữ cho stemmer
            persist_dir: Thư mục để lưu/load docstore (default: ./storage/bm25)
        """
        logger.info("🔍 Initializing LlamaIndex BM25 Retriever")
        
        self.corpus = None
        self.retriever = None
        self.nodes = None
        self.docstore = None
        
        # Persistence directory
        self.persist_dir = persist_dir or "./storage/bm25"
        
        # Node parser for chunking
        self.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        logger.info(f"✓ BM25 Retriever initialized (persist_dir: {self.persist_dir})")
    
    def build_index(self, documents: List[Dict]):
        """
        Xây dựng BM25 index
        
        Args:
            documents: List các dict {'id', 'title', 'text', 'full_text'}
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        self.corpus = documents
        
        # Convert to LlamaIndex Documents
        llama_docs = []
        for doc in documents:
            llama_doc = Document(
                text=doc['full_text'],
                metadata={
                    'doc_id': doc['id'],
                    'title': doc['title'],
                    'text_snippet': doc['text'][:200]
                },
                id_=doc['id']
            )
            llama_docs.append(llama_doc)
        
        # Parse into nodes
        logger.info("Parsing documents into nodes...")
        self.nodes = self.node_parser.get_nodes_from_documents(llama_docs)
        
        # Create docstore and add nodes
        logger.info("Creating docstore...")
        self.docstore = SimpleDocumentStore()
        self.docstore.add_documents(self.nodes)
        
        # Create BM25 retriever from docstore
        logger.info("Building BM25 index from docstore...")
        self.retriever = BM25Retriever.from_defaults(
            docstore=self.docstore,
            similarity_top_k=100  # Default, can override in search
        )
        
        logger.info(f"✓ Indexed {len(documents)} documents ({len(self.nodes)} nodes)")
    
    def search(self, query: str, top_k: int = 10,
               metadata_filter: Optional["MetadataFilter"] = None) -> List[Dict]:
        """
        BM25 search with optional metadata filtering. Requirements: 4.1, 4.2, 4.3
        """
        if self.retriever is None:
            raise ValueError("Chưa build index! Gọi build_index() trước.")

        # Update similarity_top_k
        self.retriever.similarity_top_k = top_k

        # Retrieve
        retrieved_nodes = self.retriever.retrieve(query)

        # Format results
        results = []
        for rank, node in enumerate(retrieved_nodes, 1):
            doc_id = node.node.metadata.get('doc_id', node.node.id_)
            title = node.node.metadata.get('title', 'No Title')
            text_snippet = node.node.metadata.get('text_snippet', '')

            if self.corpus:
                original_doc = next(
                    (d for d in self.corpus if d['id'] == doc_id),
                    None
                )
                if original_doc:
                    title = original_doc['title']
                    text_snippet = original_doc['text']

            results.append({
                'id': doc_id,
                'score': node.score if hasattr(node, 'score') else 0.0,
                'title': title,
                'text': text_snippet,
                'rank': rank
            })

        # Apply metadata filter post-retrieval
        if metadata_filter is not None:
            results = self._apply_filter(results, metadata_filter)
            # Re-rank after filtering
            for i, r in enumerate(results, 1):
                r['rank'] = i

        return results

    def _apply_filter(self, results: List[Dict],
                      metadata_filter: "MetadataFilter") -> List[Dict]:
        """Post-filter results by metadata. Requirements: 4.1"""
        try:
            from src.kb.metadata import MetadataFilterEngine
            from src.kb.knowledge_base import KnowledgeBase
        except ImportError:
            return results
        # Build a minimal KB from corpus metadata for filtering
        kb = KnowledgeBase()
        if self.corpus:
            from src.kb.metadata import DocumentMetadata
            for doc in self.corpus:
                meta = DocumentMetadata(doc_id=doc['id'])
                kb._metadata[doc['id']] = meta
        engine = MetadataFilterEngine()
        allowed_ids = set(engine.apply(metadata_filter, [r['id'] for r in results], kb))
        return [r for r in results if r['id'] in allowed_ids]
    
    def batch_search(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        metadata_filter: Optional["MetadataFilter"] = None,
    ) -> Dict[str, List[Dict]]:
        """Batch search with optional metadata filtering. Requirements: 4.1"""
        logger.info(f"Processing {len(queries)} queries with BM25...")
        results_dict = {}
        for qid, query_text in tqdm(queries.items(), desc="BM25 Retrieval"):
            results_dict[qid] = self.search(query_text, top_k=top_k,
                                            metadata_filter=metadata_filter)
        return results_dict
    
    def persist(self, persist_dir: Optional[str] = None):
        """
        Lưu docstore và nodes xuống disk
        
        Args:
            persist_dir: Thư mục lưu (mặc định dùng self.persist_dir)
        """
        if self.docstore is None:
            raise ValueError("Chưa build index! Gọi build_index() trước.")
        
        save_dir = persist_dir or self.persist_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save docstore
        docstore_path = os.path.join(save_dir, "docstore.json")
        self.docstore.persist(docstore_path)
        
        logger.info(f"✓ Saved BM25 docstore to {docstore_path}")
    
    def load(self, persist_dir: Optional[str] = None):
        """
        Load docstore từ disk và rebuild BM25 retriever
        
        Args:
            persist_dir: Thư mục chứa docstore (mặc định dùng self.persist_dir)
        """
        load_dir = persist_dir or self.persist_dir
        docstore_path = os.path.join(load_dir, "docstore.json")
        
        if not os.path.exists(docstore_path):
            raise FileNotFoundError(f"Docstore không tồn tại: {docstore_path}")
        
        logger.info(f"Loading BM25 docstore from {docstore_path}...")
        
        # Load docstore
        self.docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        
        # Rebuild BM25 retriever from docstore
        logger.info("Rebuilding BM25 index from loaded docstore...")
        self.retriever = BM25Retriever.from_defaults(
            docstore=self.docstore,
            similarity_top_k=100
        )
        
        # Reconstruct nodes list for search method
        self.nodes = list(self.docstore.docs.values())
        
        logger.info(f"✓ Loaded BM25 with {len(self.nodes)} nodes")
    
    @classmethod
    def from_persist_dir(cls, persist_dir: str, language: str = "english"):
        """
        Class method để load từ persist directory
        
        Args:
            persist_dir: Thư mục chứa docstore
            language: Ngôn ngữ cho stemmer
            
        Returns:
            LlamaIndexBM25 instance đã load
        """
        instance = cls(language=language, persist_dir=persist_dir)
        instance.load(persist_dir)
        return instance


# Demo
if __name__ == "__main__":
    from data_loader import BeirDataLoader
    
    print("\n=== DEMO: LlamaIndex BM25 Retriever with Persistence ===\n")
    
    # Load dataset
    loader = BeirDataLoader()
    corpus, queries, qrels = loader.load_dataset('nfcorpus')
    documents = loader.prepare_corpus_for_indexing(corpus)
    
    persist_dir = "./storage/bm25_demo"
    
    # Check if already persisted
    if os.path.exists(os.path.join(persist_dir, "docstore.json")):
        print(f"\n✅ Found existing BM25 index at {persist_dir}")
        print("Loading from disk...\n")
        bm25 = LlamaIndexBM25.from_persist_dir(persist_dir)
    else:
        print(f"\n🔨 Building new BM25 index for {len(documents[:100])} documents...")
        # Initialize BM25
        bm25 = LlamaIndexBM25(persist_dir=persist_dir)
        
        # Build index
        bm25.build_index(documents[:100])
        
        # Save to disk
        print("\n💾 Saving BM25 index to disk...")
        bm25.persist()
        print(f"✓ Saved to {persist_dir}\n")
    
    # Test query
    test_query = "What causes diabetes?"
    print(f"Query: {test_query}\n")
    
    results = bm25.search(test_query, top_k=5)
    
    print("=== Top-5 BM25 Results ===")
    for doc in results:
        print(f"[Rank {doc['rank']}] Score: {doc['score']:.4f}")
        print(f"Title: {doc['title']}")
        print(f"Text: {doc['text'][:200]}...\n")
    
    print(f"\n💡 Next run will load from {persist_dir} (faster!)")
