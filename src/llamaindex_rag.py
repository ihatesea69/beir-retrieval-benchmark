"""
Module: LlamaIndex RAG System with PostgreSQL+pgvector
Chức năng: RAG system sử dụng LlamaIndex framework với PostgreSQL vector store
Thay thế: FAISS → PostgreSQL+pgvector
"""

import os
from typing import TYPE_CHECKING, Dict, List, Optional
import logging
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.kb.metadata import MetadataFilter

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaIndexRAG:
    """
    RAG System với LlamaIndex + PostgreSQL+pgvector
    
    Ưu điểm:
    - Persistent storage với PostgreSQL
    - Unified interface qua LlamaIndex
    - Semantic search với pgvector
    - Generation với OpenAI LLM
    - Tự động chunking và embedding management
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        table_name: str = "llamaindex_documents",
        openai_api_key: Optional[str] = None
    ):
        """
        Args:
            embedding_model: HuggingFace model cho embeddings
            llm_model: OpenAI model cho generation
            table_name: PostgreSQL table name
            openai_api_key: OpenAI API key
        """
        logger.info(f"🚀 Initializing LlamaIndex RAG with PostgreSQL")
        
        # Setup embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            cache_folder="./models"
        )
        
        # Setup LLM
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠ Không tìm thấy OPENAI_API_KEY")
            llm = None
        else:
            llm = OpenAI(model=llm_model, api_key=api_key, temperature=0.1)
        
        # Configure global settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Setup PostgreSQL connection
        connection_string = self._get_connection_string()
        url = make_url(connection_string)
        
        # Create vector store
        self.vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=table_name,
            embed_dim=384,  # all-MiniLM-L6-v2 dimension
        )
        
        # Storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = None
        self.corpus = None
        
        logger.info("✓ LlamaIndex RAG initialized with PostgreSQL")
    
    def _get_connection_string(self) -> str:
        """Tạo PostgreSQL connection string từ .env"""
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5433")
        db = os.getenv("POSTGRES_DB", "beir_benchmark")
        user = os.getenv("POSTGRES_USER", "beir_user")
        password = os.getenv("POSTGRES_PASSWORD", "beir_password")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    def build_index(self, documents: List[Dict]):
        """
        Xây dựng vector index từ corpus
        
        Args:
            documents: List các dict {'id', 'title', 'text', 'full_text'}
        """
        logger.info(f"Building LlamaIndex with {len(documents)} documents...")
        
        self.corpus = documents
        
        # Convert to LlamaIndex Document objects
        llama_docs = []
        for doc in documents:
            # Create Document with metadata
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
        
        # Build index (automatically embeds and stores in PostgreSQL)
        logger.info("Creating embeddings and storing in PostgreSQL...")
        self.index = VectorStoreIndex.from_documents(
            llama_docs,
            storage_context=self.storage_context,
            show_progress=True
        )
        
        logger.info(f"✓ Indexed {len(documents)} documents in PostgreSQL")
    
    def search(self, query: str, top_k: int = 10,
               metadata_filter: Optional["MetadataFilter"] = None) -> List[Dict]:
        """Semantic search with optional metadata filtering. Requirements: 4.1"""
        if self.index is None:
            raise ValueError("Chưa build index! Gọi build_index() trước.")

        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []
        for rank, node in enumerate(nodes, 1):
            doc_id = node.node.metadata.get('doc_id', node.node.id_)
            original_doc = next(
                (d for d in self.corpus if d['id'] == doc_id), None
            )
            if original_doc:
                results.append({
                    'id': doc_id,
                    'score': node.score,
                    'title': original_doc['title'],
                    'text': original_doc['text'],
                    'rank': rank
                })

        if metadata_filter is not None:
            results = self._apply_filter(results, metadata_filter)
            for i, r in enumerate(results, 1):
                r['rank'] = i

        return results

    def _apply_filter(self, results: List[Dict],
                      metadata_filter: "MetadataFilter") -> List[Dict]:
        """Post-filter results by metadata. Requirements: 4.1"""
        try:
            from src.kb.metadata import MetadataFilterEngine, DocumentMetadata
            from src.kb.knowledge_base import KnowledgeBase
        except ImportError:
            return results
        kb = KnowledgeBase()
        if self.corpus:
            for doc in self.corpus:
                kb._metadata[doc['id']] = DocumentMetadata(doc_id=doc['id'])
        engine = MetadataFilterEngine()
        allowed_ids = set(engine.apply(metadata_filter, [r['id'] for r in results], kb))
        return [r for r in results if r['id'] in allowed_ids]
    
    def generate_answer(
        self,
        query: str,
        contexts: List[Dict],
        max_contexts: int = 3
    ) -> Dict:
        """
        Sinh câu trả lời dựa trên contexts
        
        Args:
            query: Câu hỏi
            contexts: Danh sách contexts từ search()
            max_contexts: Số lượng contexts sử dụng
            
        Returns:
            Dict {'answer': str, 'contexts_used': List}
        """
        if Settings.llm is None:
            return {
                'answer': "⚠ Không có OpenAI API key để generate answer",
                'contexts_used': []
            }
        
        # Take top contexts
        top_contexts = contexts[:max_contexts]
        
        # Build prompt
        context_text = "\n\n".join([
            f"Document {i+1}: {ctx['title']}\n{ctx['text']}"
            for i, ctx in enumerate(top_contexts)
        ])
        
        prompt = f"""Based on the following context documents, answer the question.

Context:
{context_text}

Question: {query}

Answer (be concise and factual):"""
        
        # Generate
        response = Settings.llm.complete(prompt)
        
        return {
            'answer': response.text.strip(),
            'contexts_used': top_contexts
        }
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        generate: bool = False
    ) -> Dict:
        """
        End-to-end RAG query
        
        Args:
            query_text: Câu hỏi
            top_k: Số documents retrieve
            generate: Có sinh answer không
            
        Returns:
            Dict {'results': List, 'answer': str (nếu generate=True)}
        """
        # Retrieve
        results = self.search(query_text, top_k=top_k)
        
        output = {'results': results}
        
        # Generate nếu cần
        if generate and results:
            gen_result = self.generate_answer(query_text, results)
            output['answer'] = gen_result['answer']
            output['contexts_used'] = gen_result['contexts_used']
        
        return output
    
    def batch_search(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        metadata_filter: Optional["MetadataFilter"] = None,
    ) -> Dict[str, List[Dict]]:
        """Batch search with optional metadata filtering. Requirements: 4.1"""
        logger.info(f"Đang xử lý {len(queries)} queries...")
        results_dict = {}
        from tqdm import tqdm
        for qid, query_text in tqdm(queries.items(), desc="Dense Retrieval"):
            results_dict[qid] = self.search(query_text, top_k=top_k,
                                            metadata_filter=metadata_filter)
        return results_dict


# Demo
if __name__ == "__main__":
    from data_loader import BeirDataLoader
    
    print("\n=== DEMO: LlamaIndex RAG with PostgreSQL ===\n")
    
    # Load dataset
    loader = BeirDataLoader()
    corpus, queries, qrels = loader.load_dataset('nfcorpus')
    documents = loader.prepare_corpus_for_indexing(corpus)
    
    # Initialize RAG
    rag = LlamaIndexRAG()
    
    # Build index (embed vào PostgreSQL)
    print(f"\nBuilding index for {len(documents[:100])} documents...")
    rag.build_index(documents[:100])
    
    # Test query
    test_query = "What causes diabetes?"
    print(f"\nQuery: {test_query}\n")
    
    # Search only
    result = rag.query(test_query, top_k=5, generate=False)
    
    print("=== Top-5 Results ===")
    for doc in result['results']:
        print(f"[Rank {doc['rank']}] Score: {doc['score']:.4f}")
        print(f"Title: {doc['title']}")
        print(f"Text: {doc['text'][:200]}...\n")
    
    # With generation
    if os.getenv("OPENAI_API_KEY"):
        print("\n=== Generated Answer ===")
        result_gen = rag.query(test_query, top_k=5, generate=True)
        print(result_gen['answer'])
