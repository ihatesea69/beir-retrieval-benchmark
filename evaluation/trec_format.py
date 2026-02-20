"""
TREC Format Handler
Handles loading and saving of TREC evaluation data formats
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TRECFormatHandler:
    """
    Handler for TREC evaluation data formats
    Supports qrels, topics, and results files
    """
    
    @staticmethod
    def load_qrels(filepath: str) -> Dict[str, Dict[str, int]]:
        """
        Load TREC qrels file
        
        Format: query_id 0 doc_id relevance
        
        Args:
            filepath: Path to qrels file
            
        Returns:
            Dict mapping query_id to dict of doc_id -> relevance
        """
        qrels = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) != 4:
                        logger.warning(f"Invalid qrels format at line {line_num}: {line}")
                        continue
                    
                    query_id, iteration, doc_id, relevance = parts
                    
                    # Validate relevance score
                    try:
                        relevance = int(relevance)
                    except ValueError:
                        logger.warning(f"Invalid relevance score at line {line_num}: {relevance}")
                        continue
                    
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    
                    qrels[query_id][doc_id] = relevance
                    
        except FileNotFoundError:
            logger.error(f"Qrels file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading qrels file {filepath}: {e}")
            raise
        
        logger.info(f"Loaded qrels for {len(qrels)} queries from {filepath}")
        return qrels
    
    @staticmethod
    def save_qrels(qrels: Dict[str, Dict[str, int]], filepath: str) -> None:
        """
        Save qrels in TREC format
        
        Args:
            qrels: Dict mapping query_id to dict of doc_id -> relevance
            filepath: Output file path
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for query_id in sorted(qrels.keys()):
                    for doc_id in sorted(qrels[query_id].keys()):
                        relevance = qrels[query_id][doc_id]
                        f.write(f"{query_id} 0 {doc_id} {relevance}\n")
                        
            logger.info(f"Saved qrels to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving qrels to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_topics(filepath: str) -> Dict[str, str]:
        """
        Load TREC topics file (XML/SGML format)
        
        Args:
            filepath: Path to topics file
            
        Returns:
            Dict mapping topic_id to query text
        """
        topics = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try XML parsing first
            try:
                root = ET.fromstring(content)
                for topic in root.findall('.//topic'):
                    topic_id = topic.get('number') or topic.find('num').text.strip()
                    title = topic.find('title')
                    if title is not None:
                        topics[topic_id] = title.text.strip()
                        
            except ET.ParseError:
                # Fall back to SGML-style parsing
                topics = TRECFormatHandler._parse_sgml_topics(content)
                
        except FileNotFoundError:
            logger.error(f"Topics file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading topics file {filepath}: {e}")
            raise
        
        logger.info(f"Loaded {len(topics)} topics from {filepath}")
        return topics
    
    @staticmethod
    def _parse_sgml_topics(content: str) -> Dict[str, str]:
        """
        Parse SGML-style TREC topics
        
        Args:
            content: File content as string
            
        Returns:
            Dict mapping topic_id to query text
        """
        topics = {}
        
        # Pattern to match TREC topic structure
        topic_pattern = r'<top>(.*?)</top>'
        num_pattern = r'<num>\s*Number:\s*(\d+)'
        title_pattern = r'<title>\s*(.*?)(?=<|$)'
        
        topic_matches = re.findall(topic_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for topic_content in topic_matches:
            num_match = re.search(num_pattern, topic_content, re.IGNORECASE)
            title_match = re.search(title_pattern, topic_content, re.IGNORECASE)
            
            if num_match and title_match:
                topic_id = num_match.group(1)
                title = title_match.group(1).strip()
                topics[topic_id] = title
        
        return topics
    
    @staticmethod
    def load_results(filepath: str) -> Dict[str, List[Dict]]:
        """
        Load TREC results file
        
        Format: query_id Q0 doc_id rank score run_id
        
        Args:
            filepath: Path to results file
            
        Returns:
            Dict mapping query_id to list of retrieved documents
        """
        results = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) != 6:
                        logger.warning(f"Invalid results format at line {line_num}: {line}")
                        continue
                    
                    query_id, q0, doc_id, rank, score, run_id = parts
                    
                    # Validate numeric fields
                    try:
                        rank = int(rank)
                        score = float(score)
                    except ValueError:
                        logger.warning(f"Invalid numeric values at line {line_num}: {line}")
                        continue
                    
                    if query_id not in results:
                        results[query_id] = []
                    
                    results[query_id].append({
                        'id': doc_id,
                        'rank': rank,
                        'score': score,
                        'run_id': run_id
                    })
                    
        except FileNotFoundError:
            logger.error(f"Results file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading results file {filepath}: {e}")
            raise
        
        # Sort results by rank for each query
        for query_id in results:
            results[query_id].sort(key=lambda x: x['rank'])
        
        logger.info(f"Loaded results for {len(results)} queries from {filepath}")
        return results
    
    @staticmethod
    def save_results_trec_format(
        results: Dict[str, List[Dict]], 
        filepath: str, 
        run_id: str
    ) -> None:
        """
        Save results in TREC format
        
        Args:
            results: Dict mapping query_id to list of retrieved documents
            filepath: Output file path
            run_id: Run identifier
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for query_id in sorted(results.keys()):
                    docs = results[query_id]
                    for doc in docs:
                        rank = doc.get('rank', 1)
                        score = doc.get('score', 0.0)
                        doc_id = doc['id']
                        
                        f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_id}\n")
                        
            logger.info(f"Saved results to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results to {filepath}: {e}")
            raise
    
    @staticmethod
    def validate_qrels_format(filepath: str) -> List[str]:
        """
        Validate TREC qrels file format
        
        Args:
            filepath: Path to qrels file
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) != 4:
                        errors.append(f"Line {line_num}: Expected 4 fields, got {len(parts)}")
                        continue
                    
                    query_id, iteration, doc_id, relevance = parts
                    
                    # Check iteration field (should be 0)
                    if iteration != '0':
                        errors.append(f"Line {line_num}: Iteration field should be '0', got '{iteration}'")
                    
                    # Check relevance score
                    try:
                        rel_score = int(relevance)
                        if rel_score < 0 or rel_score > 4:
                            errors.append(f"Line {line_num}: Relevance score {rel_score} outside valid range (0-4)")
                    except ValueError:
                        errors.append(f"Line {line_num}: Invalid relevance score '{relevance}'")
                        
        except FileNotFoundError:
            errors.append(f"File not found: {filepath}")
        except Exception as e:
            errors.append(f"Error reading file: {e}")
        
        return errors
    
    @staticmethod
    def convert_to_internal_format(
        qrels: Dict[str, Dict[str, int]], 
        results: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        Convert TREC format data to internal evaluation format
        
        Args:
            qrels: TREC qrels data
            results: TREC results data
            
        Returns:
            Dict with converted data for internal evaluation
        """
        converted = {
            'qrels': qrels,
            'results': {},
            'queries': set(qrels.keys()) & set(results.keys())
        }
        
        # Convert results format
        for query_id in converted['queries']:
            converted['results'][query_id] = [
                {
                    'id': doc['id'],
                    'score': doc['score'],
                    'rank': doc['rank']
                }
                for doc in results[query_id]
            ]
        
        return converted