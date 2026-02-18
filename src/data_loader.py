"""
Module: BeIR Data Loader
Chức năng: Tải và xử lý dữ liệu từ BeIR benchmark
Datasets: NFCorpus (Y tế), MS MARCO (General Domain)
"""

import os
from typing import Dict, List, Tuple
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeirDataLoader:
    """
    Class quản lý việc tải và chuẩn bị dữ liệu từ BeIR benchmark
    """
    
    SUPPORTED_DATASETS = {
        'nfcorpus': 'NFCorpus - Y tế/Dinh dưỡng (3,633 docs)',
        'msmarco': 'MS MARCO - Kiến thức chung (8.8M docs - subset)',
        'fiqa': 'FiQA - Tài chính (57,638 docs)',
        'scifact': 'SciFact - Khoa học (5,183 docs)'
    }
    
    def __init__(self, data_path: str = "./data/beir_datasets"):
        """
        Args:
            data_path: Đường dẫn lưu trữ dữ liệu BeIR
        """
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        
    def download_dataset(self, dataset_name: str) -> str:
        """
        Tải dataset từ BeIR
        
        Args:
            dataset_name: Tên dataset ('nfcorpus', 'msmarco', etc.)
            
        Returns:
            Đường dẫn đến thư mục dataset
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' không được hỗ trợ. "
                f"Chọn từ: {list(self.SUPPORTED_DATASETS.keys())}"
            )
        
        logger.info(f"Đang tải dataset: {self.SUPPORTED_DATASETS[dataset_name]}")
        
        dataset_path = os.path.join(self.data_path, dataset_name)
        
        # Tải dataset nếu chưa có
        if not os.path.exists(dataset_path):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, self.data_path)
            logger.info(f"✓ Tải thành công: {dataset_name}")
        else:
            logger.info(f"✓ Dataset đã tồn tại: {dataset_name}")
            
        return dataset_path
    
    def load_dataset(self, dataset_name: str) -> Tuple[Dict, Dict, Dict]:
        """
        Load corpus, queries và qrels từ dataset
        
        Args:
            dataset_name: Tên dataset
            
        Returns:
            Tuple gồm (corpus, queries, qrels)
            - corpus: Dict[doc_id, {"title": str, "text": str}]
            - queries: Dict[query_id, str]
            - qrels: Dict[query_id, Dict[doc_id, relevance_score]]
        """
        dataset_path = self.download_dataset(dataset_name)
        
        logger.info(f"Đang load dữ liệu từ: {dataset_path}")
        
        # Load dữ liệu bằng GenericDataLoader của BeIR
        corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")
        
        logger.info(f"✓ Corpus: {len(corpus)} documents")
        logger.info(f"✓ Queries: {len(queries)} questions")
        logger.info(f"✓ Qrels: {len(qrels)} query-document pairs")
        
        return corpus, queries, qrels
    
    def get_sample_queries(self, queries: Dict, n: int = 100) -> Dict:
        """
        Lấy n câu hỏi mẫu để test
        
        Args:
            queries: Dict chứa tất cả queries
            n: Số lượng queries muốn lấy
            
        Returns:
            Dict chứa n queries đầu tiên
        """
        query_ids = list(queries.keys())[:n]
        return {qid: queries[qid] for qid in query_ids}
    
    def prepare_corpus_for_indexing(self, corpus: Dict) -> List[Dict]:
        """
        Chuẩn bị corpus cho việc indexing (BM25 hoặc Vector)
        
        Args:
            corpus: Dict corpus từ BeIR
            
        Returns:
            List các documents dạng dict với format chuẩn
        """
        documents = []
        for doc_id, doc_data in corpus.items():
            # Kết hợp title và text thành một document hoàn chỉnh
            full_text = f"{doc_data.get('title', '')} {doc_data.get('text', '')}".strip()
            
            documents.append({
                'id': doc_id,
                'title': doc_data.get('title', ''),
                'text': doc_data.get('text', ''),
                'full_text': full_text
            })
            
        logger.info(f"✓ Đã chuẩn bị {len(documents)} documents để indexing")
        return documents
    
    def get_ground_truth(self, qrels: Dict, query_id: str) -> List[str]:
        """
        Lấy danh sách document IDs đúng cho một query
        
        Args:
            qrels: Ground truth data
            query_id: ID của query
            
        Returns:
            List các document IDs có relevance > 0
        """
        if query_id not in qrels:
            return []
        
        # Lấy tất cả docs có relevance score > 0
        relevant_docs = [
            doc_id for doc_id, score in qrels[query_id].items() 
            if score > 0
        ]
        
        return relevant_docs
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict:
        """
        Thống kê thông tin về dataset
        
        Args:
            dataset_name: Tên dataset
            
        Returns:
            Dict chứa các thông tin thống kê
        """
        corpus, queries, qrels = self.load_dataset(dataset_name)
        
        # Tính toán thống kê
        avg_doc_length = sum(
            len(doc['title'].split()) + len(doc['text'].split()) 
            for doc in corpus.values()
        ) / len(corpus)
        
        avg_query_length = sum(len(q.split()) for q in queries.values()) / len(queries)
        
        avg_relevant_docs = sum(
            len([d for d, s in docs.items() if s > 0]) 
            for docs in qrels.values()
        ) / len(qrels)
        
        return {
            'dataset_name': dataset_name,
            'num_documents': len(corpus),
            'num_queries': len(queries),
            'num_qrels': len(qrels),
            'avg_doc_length': avg_doc_length,
            'avg_query_length': avg_query_length,
            'avg_relevant_docs_per_query': avg_relevant_docs
        }


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = BeirDataLoader()
    
    # Test với NFCorpus (dataset nhỏ, dễ demo)
    print("\n=== TEST: Load NFCorpus ===")
    corpus, queries, qrels = loader.load_dataset('nfcorpus')
    
    # Hiển thị thống kê
    stats = loader.get_dataset_statistics('nfcorpus')
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Lấy 5 query mẫu
    sample_queries = loader.get_sample_queries(queries, n=5)
    print("\n=== Sample Queries ===")
    for qid, query_text in list(sample_queries.items())[:3]:
        print(f"Query ID: {qid}")
        print(f"Text: {query_text}")
        print(f"Ground Truth Docs: {loader.get_ground_truth(qrels, qid)[:3]}")
        print()
