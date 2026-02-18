"""
DocumentChunk dataclass and StructureAwareChunker.
Requirements: 2.1, 2.2, 2.3, 2.4
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .metadata import DocumentMetadata


@dataclass
class DocumentChunk:
    """A single chunk produced from a document. Requirements: 2.3, 2.4"""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    heading: str = ""
    metadata: Optional[DocumentMetadata] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "heading": self.heading,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        meta = None
        if data.get("metadata"):
            meta = DocumentMetadata.from_dict(data["metadata"])
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            heading=data.get("heading", ""),
            metadata=meta,
        )


# Heading patterns: markdown (#, ##, ###) and numbered (1., 2.1, Điều X, Chapter X)
_HEADING_RE = re.compile(
    r"^(?:#{1,4}\s+.+|(?:\d+\.)+\d*\s+.+|\d+\.\s+.+|(?:Điều|Chương|Mục|Article|Chapter|Section)\s+\S+.*)",
    re.MULTILINE | re.IGNORECASE,
)


class StructureAwareChunker:
    """
    Splits documents by heading boundaries first; falls back to sentence splitting.
    Requirements: 2.1, 2.2, 2.3
    """

    def __init__(self, max_chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        doc_id: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        """
        Produce a list of DocumentChunk from a document.
        Each chunk inherits doc_id and doc_type from metadata. Requirements: 2.3
        """
        text = text.strip()
        if not text:
            return []

        sections = self._split_by_headings(text)

        chunks: List[DocumentChunk] = []
        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            words = body.split()
            if len(words) <= self.max_chunk_size:
                chunks.append(self._make_chunk(doc_id, len(chunks), body, heading, metadata))
            else:
                # Fall back to sentence splitting for oversized sections
                sub_texts = self._split_by_sentences(body)
                for sub in sub_texts:
                    if sub.strip():
                        chunks.append(self._make_chunk(doc_id, len(chunks), sub.strip(), heading, metadata))

        if not chunks and text.strip():
            # No headings found at all — split entire text by sentences
            for sub in self._split_by_sentences(text):
                if sub.strip():
                    chunks.append(self._make_chunk(doc_id, len(chunks), sub.strip(), "", metadata))

        return chunks

    def _make_chunk(
        self,
        doc_id: str,
        index: int,
        text: str,
        heading: str,
        metadata: DocumentMetadata,
    ) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{index}",
            doc_id=doc_id,
            chunk_index=index,
            text=text,
            heading=heading,
            metadata=metadata,
        )

    def _split_by_headings(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text at heading boundaries.
        Returns list of (heading, body) tuples. Requirements: 2.1
        """
        lines = text.split("\n")
        sections: List[Tuple[str, str]] = []
        current_heading = ""
        current_body: List[str] = []

        for line in lines:
            if _HEADING_RE.match(line.strip()):
                if current_body:
                    sections.append((current_heading, "\n".join(current_body)))
                    current_body = []
                current_heading = line.strip()
            else:
                current_body.append(line)

        if current_body:
            sections.append((current_heading, "\n".join(current_body)))

        # If no headings were found, return the whole text as one section
        if not sections or (len(sections) == 1 and sections[0][0] == ""):
            return [("", text)]

        return sections

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks of at most max_chunk_size words. Requirements: 2.2
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.max_chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start = end - self.chunk_overlap if end < len(words) else end

        return chunks
