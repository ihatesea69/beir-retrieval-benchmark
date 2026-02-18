"""
KnowledgeBase: central store for documents, metadata, and chunks.
Requirements: 3.3, 3.5, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from .chunker import DocumentChunk, StructureAwareChunker
from .metadata import DocumentMetadata, MetadataFilter, MetadataFilterEngine

logger = logging.getLogger(__name__)

KB_VERSION = "1.0"


class KnowledgeBase:
    """
    In-memory KB with Document Store, Metadata Store, and Chunk Store.
    Requirements: 3.3, 3.5, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3
    """

    def __init__(self, max_chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self._documents: Dict[str, str] = {}           # doc_id -> raw text
        self._metadata: Dict[str, DocumentMetadata] = {}  # doc_id -> metadata
        self._chunks: Dict[str, List[DocumentChunk]] = {}  # doc_id -> chunks
        self._chunker = StructureAwareChunker(max_chunk_size, chunk_overlap)
        self._filter_engine = MetadataFilterEngine()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        """
        Add or update a document (upsert semantics). Requirements: 3.5
        """
        self._documents[doc_id] = text
        self._metadata[doc_id] = metadata
        chunks = self._chunker.chunk(doc_id, text, metadata)
        self._chunks[doc_id] = chunks
        return chunks

    def update_document(
        self,
        doc_id: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        """Alias for add_document (explicit update). Requirements: 3.5"""
        return self.add_document(doc_id, text, metadata)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_document(self, doc_id: str) -> Optional[str]:
        return self._documents.get(doc_id)

    def get_chunks(self, doc_id: str) -> List[DocumentChunk]:
        return self._chunks.get(doc_id, [])

    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        return self._metadata.get(doc_id)

    def all_doc_ids(self) -> List[str]:
        return list(self._documents.keys())

    def filter_documents(self, metadata_filter: MetadataFilter) -> List[str]:
        """Return doc_ids matching the filter. Requirements: 4.1, 4.2, 4.3"""
        return self._filter_engine.apply(metadata_filter, self.all_doc_ids(), self)

    # ------------------------------------------------------------------
    # Stats & Health
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """
        Return KB statistics without modifying any data. Requirements: 5.1, 5.3
        """
        total_docs = len(self._documents)
        total_chunks = sum(len(c) for c in self._chunks.values())
        per_type: Dict[str, int] = defaultdict(int)
        for meta in self._metadata.values():
            per_type[meta.doc_type] += 1
        avg_chunks = total_chunks / total_docs if total_docs > 0 else 0.0

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "per_doc_type": dict(per_type),
            "avg_chunks_per_document": round(avg_chunks, 4),
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Identify documents with missing metadata fields. Requirements: 5.2
        """
        required_fields = ["doc_type", "source_url", "published_date", "effective_date"]
        issues: Dict[str, List[str]] = {}

        for doc_id, meta in self._metadata.items():
            missing = []
            for f in required_fields:
                val = getattr(meta, f, None)
                if val is None or val == "":
                    missing.append(f)
            if missing:
                issues[doc_id] = missing

        return {
            "total_documents": len(self._documents),
            "documents_with_issues": len(issues),
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Export entire KB to a JSON file. Requirements: 6.1"""
        data = {
            "version": KB_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "documents": [
                {"doc_id": doc_id, "text": text}
                for doc_id, text in self._documents.items()
            ],
            "metadata": [
                meta.to_dict() for meta in self._metadata.values()
            ],
            "chunks": [
                chunk.to_dict()
                for chunks in self._chunks.values()
                for chunk in chunks
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Exported KB (%d docs, %d chunks) to %s",
                    len(self._documents), sum(len(c) for c in self._chunks.values()), path)

    def import_json(self, path: str) -> None:
        """Import KB from a JSON file, replacing current state. Requirements: 6.2"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "")
        if version != KB_VERSION:
            raise ValueError(f"Incompatible KB version: '{version}' (expected '{KB_VERSION}')")

        self._documents.clear()
        self._metadata.clear()
        self._chunks.clear()

        # Restore documents
        for doc_entry in data.get("documents", []):
            self._documents[doc_entry["doc_id"]] = doc_entry["text"]

        # Restore metadata
        for meta_entry in data.get("metadata", []):
            meta = DocumentMetadata.from_dict(meta_entry)
            self._metadata[meta.doc_id] = meta

        # Restore chunks (grouped by doc_id)
        chunk_map: Dict[str, List[DocumentChunk]] = defaultdict(list)
        for chunk_entry in data.get("chunks", []):
            chunk = DocumentChunk.from_dict(chunk_entry)
            chunk_map[chunk.doc_id].append(chunk)

        # Sort chunks by chunk_index
        for doc_id, chunks in chunk_map.items():
            self._chunks[doc_id] = sorted(chunks, key=lambda c: c.chunk_index)

        logger.info("Imported KB (%d docs) from %s", len(self._documents), path)
