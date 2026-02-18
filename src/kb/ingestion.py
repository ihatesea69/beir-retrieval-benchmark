"""
DocumentIngestionPipeline: ingest documents from JSONL, directory, or BeIR corpus.
Requirements: 3.1, 3.2, 3.3, 3.4
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from .knowledge_base import KnowledgeBase
from .metadata import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Orchestrates: extract -> clean -> metadata -> chunk -> add to KB.
    Requirements: 3.1, 3.2, 3.3, 3.4
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    # ------------------------------------------------------------------
    # Public ingestion methods
    # ------------------------------------------------------------------

    def ingest_jsonl(
        self,
        path: str,
        metadata_map: Optional[Dict[str, DocumentMetadata]] = None,
    ) -> Dict:
        """
        Ingest a JSONL file. Each line must have at least '_id' and 'text'.
        Requirements: 3.1, 3.4
        """
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping malformed JSONL line: %s", e)

        return self._ingest_records(records, metadata_map, source_label=path)

    def ingest_directory(
        self,
        dir_path: str,
        metadata_map: Optional[Dict[str, DocumentMetadata]] = None,
    ) -> Dict:
        """
        Ingest all .txt and .md files in a directory.
        Filename (without extension) becomes the doc_id. Requirements: 3.2, 3.4
        """
        records = []
        for fname in os.listdir(dir_path):
            if fname.endswith((".txt", ".md")):
                doc_id = os.path.splitext(fname)[0]
                fpath = os.path.join(dir_path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                    records.append({"_id": doc_id, "text": text})
                except Exception as e:
                    logger.warning("Failed to read %s: %s", fpath, e)

        return self._ingest_records(records, metadata_map, source_label=dir_path)

    def ingest_beir_corpus(
        self,
        corpus: Dict,
        dataset_name: str,
    ) -> Dict:
        """
        Ingest a BeIR corpus dict {doc_id: {"title": str, "text": str, "metadata": {...}}}.
        Assigns doc_type="research_paper" and source_url from metadata. Requirements: 3.1, 3.4
        """
        records = []
        for doc_id, doc_data in corpus.items():
            title = doc_data.get("title", "")
            text = doc_data.get("text", "")
            full_text = f"{title} {text}".strip() if title else text
            source_url = doc_data.get("metadata", {}).get("url", "")
            records.append({
                "_id": doc_id,
                "text": full_text,
                "_source_url": source_url,
            })

        # Build metadata map with research_paper type
        metadata_map: Dict[str, DocumentMetadata] = {}
        for rec in records:
            doc_id = rec["_id"]
            metadata_map[doc_id] = DocumentMetadata(
                doc_id=doc_id,
                doc_type="research_paper",
                source_url=rec.get("_source_url", ""),
            )

        return self._ingest_records(records, metadata_map, source_label=dataset_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ingest_records(
        self,
        records: List[Dict],
        metadata_map: Optional[Dict[str, DocumentMetadata]],
        source_label: str,
    ) -> Dict:
        processed = 0
        chunks_produced = 0
        errors: List[Dict] = []

        for rec in records:
            try:
                doc_id, text = self._extract(rec)
                text = self._clean(text)
                meta = (metadata_map or {}).get(doc_id) or DocumentMetadata(doc_id=doc_id)
                chunks = self.kb.add_document(doc_id, text, meta)
                processed += 1
                chunks_produced += len(chunks)
            except Exception as e:
                errors.append({"doc_id": rec.get("_id", "unknown"), "error": str(e)})
                logger.warning("Failed to ingest doc '%s': %s", rec.get("_id"), e)

        report = self._report(processed, chunks_produced, errors)
        logger.info(
            "Ingestion from '%s': %d docs, %d chunks, %d errors",
            source_label, processed, chunks_produced, len(errors),
        )
        return report

    def _extract(self, raw: Dict) -> Tuple[str, str]:
        """Extract (doc_id, text) from a raw record. Requirements: 3.1"""
        doc_id = raw.get("_id") or raw.get("id")
        if not doc_id:
            raise ValueError("Record missing '_id' or 'id' field")
        text = raw.get("text") or raw.get("content") or ""
        if not text:
            raise ValueError(f"Record '{doc_id}' has no text content")
        return str(doc_id), text

    def _clean(self, text: str) -> str:
        """Strip extra whitespace and normalize unicode. Requirements: 3.3"""
        # Normalize unicode (NFC)
        text = unicodedata.normalize("NFC", text)
        # Collapse multiple spaces/newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _report(self, processed: int, chunks: int, errors: List) -> Dict:
        return {
            "processed": processed,
            "chunks_produced": chunks,
            "errors": errors,
            "error_count": len(errors),
        }
