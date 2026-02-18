"""
Knowledge Base package for structured document management.
"""
from .metadata import DocumentMetadata, MetadataFilter, MetadataFilterEngine, ALLOWED_DOC_TYPES
from .chunker import DocumentChunk, StructureAwareChunker
from .knowledge_base import KnowledgeBase
from .ingestion import DocumentIngestionPipeline

__all__ = [
    "DocumentMetadata",
    "MetadataFilter",
    "MetadataFilterEngine",
    "ALLOWED_DOC_TYPES",
    "DocumentChunk",
    "StructureAwareChunker",
    "KnowledgeBase",
    "DocumentIngestionPipeline",
]
