"""
Metadata schema, filter, and filter engine for the Knowledge Base.
Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.2, 4.3, 4.4
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

ALLOWED_DOC_TYPES = {
    "regulation",
    "guide",
    "announcement",
    "form",
    "schedule",
    "research_paper",
    "other",
}


def validate_doc_type(doc_type: str) -> str:
    """Coerce unknown doc_type values to 'other' with a warning."""
    if doc_type in ALLOWED_DOC_TYPES:
        return doc_type
    logger.warning("Unknown doc_type '%s', coercing to 'other'", doc_type)
    return "other"


@dataclass
class DocumentMetadata:
    """Structured metadata for a single document. Requirements: 1.1, 1.2, 1.3"""

    doc_id: str
    doc_type: str = "other"
    source_url: str = ""
    published_date: Optional[str] = None   # ISO 8601 e.g. "2024-01-15"
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    issuer_unit: str = ""
    audience: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.doc_type = validate_doc_type(self.doc_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "source_url": self.source_url,
            "published_date": self.published_date,
            "effective_date": self.effective_date,
            "expiry_date": self.expiry_date,
            "issuer_unit": self.issuer_unit,
            "audience": list(self.audience),
            "tags": list(self.tags),
            "attachments": list(self.attachments),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        return cls(
            doc_id=data["doc_id"],
            doc_type=data.get("doc_type", "other"),
            source_url=data.get("source_url", ""),
            published_date=data.get("published_date"),
            effective_date=data.get("effective_date"),
            expiry_date=data.get("expiry_date"),
            issuer_unit=data.get("issuer_unit", ""),
            audience=list(data.get("audience") or []),
            tags=list(data.get("tags") or []),
            attachments=list(data.get("attachments") or []),
        )


@dataclass
class MetadataFilter:
    """Filter criteria for metadata-filtered retrieval. Requirements: 4.1, 4.2, 4.3"""

    doc_type: Optional[str] = None
    effective_after: Optional[str] = None    # ISO 8601
    effective_before: Optional[str] = None   # ISO 8601
    audience: Optional[str] = None
    tags: Optional[List[str]] = None


class MetadataFilterEngine:
    """Applies a MetadataFilter to a list of doc_ids using a KnowledgeBase."""

    def apply(
        self,
        metadata_filter: MetadataFilter,
        doc_ids: List[str],
        kb: Any,  # KnowledgeBase — avoid circular import
    ) -> List[str]:
        """
        Return only doc_ids whose metadata satisfies all non-null filter criteria.
        Requirements: 4.1, 4.2, 4.3
        """
        result = []
        for doc_id in doc_ids:
            meta = kb.get_metadata(doc_id)
            if meta is None:
                continue
            if not self._matches(metadata_filter, meta):
                continue
            result.append(doc_id)
        return result

    def _matches(self, f: MetadataFilter, meta: DocumentMetadata) -> bool:
        if f.doc_type is not None and meta.doc_type != f.doc_type:
            return False

        if f.effective_after is not None:
            if meta.effective_date is None:
                return False
            if meta.effective_date < f.effective_after:
                return False

        if f.effective_before is not None:
            if meta.effective_date is None:
                return False
            if meta.effective_date > f.effective_before:
                return False

        if f.audience is not None:
            if f.audience not in meta.audience:
                return False

        if f.tags is not None:
            for tag in f.tags:
                if tag not in meta.tags:
                    return False

        return True
