"""
Property-based tests for the Knowledge Base enhancement.
Feature: knowledge-base-enhancement
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings
from hypothesis import strategies as st

from src.kb.metadata import (
    ALLOWED_DOC_TYPES,
    DocumentMetadata,
    MetadataFilter,
    MetadataFilterEngine,
)
from src.kb.chunker import DocumentChunk, StructureAwareChunker
from src.kb.knowledge_base import KnowledgeBase

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_safe_str = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "),
    min_size=0,
    max_size=40,
)
_safe_id = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)
_iso_date = st.one_of(
    st.none(),
    st.dates(
        min_value=__import__("datetime").date(2000, 1, 1),
        max_value=__import__("datetime").date(2030, 12, 31),
    ).map(lambda d: d.isoformat()),
)
_doc_type = st.sampled_from(sorted(ALLOWED_DOC_TYPES))
_str_list = st.lists(_safe_str, min_size=0, max_size=3)


def metadata_strategy():
    return st.builds(
        DocumentMetadata,
        doc_id=_safe_id,
        doc_type=_doc_type,
        source_url=_safe_str,
        published_date=_iso_date,
        effective_date=_iso_date,
        expiry_date=_iso_date,
        issuer_unit=_safe_str,
        audience=_str_list,
        tags=_str_list,
        attachments=_str_list,
    )


def chunk_strategy():
    return st.builds(
        DocumentChunk,
        chunk_id=_safe_id,
        doc_id=_safe_id,
        chunk_index=st.integers(min_value=0, max_value=100),
        text=st.text(min_size=1, max_size=200),
        heading=_safe_str,
        metadata=metadata_strategy(),
    )


# ---------------------------------------------------------------------------
# Property 1: Metadata serialization round-trip
# **Feature: knowledge-base-enhancement, Property 1: Metadata serialization round-trip**
# **Validates: Requirements 1.4**
# ---------------------------------------------------------------------------

@given(meta=metadata_strategy())
@settings(max_examples=100)
def test_metadata_round_trip(meta: DocumentMetadata):
    """
    **Feature: knowledge-base-enhancement, Property 1: Metadata serialization round-trip**
    **Validates: Requirements 1.4**
    For any DocumentMetadata, to_dict() then from_dict() produces an equivalent object.
    """
    restored = DocumentMetadata.from_dict(meta.to_dict())
    assert restored.doc_id == meta.doc_id
    assert restored.doc_type == meta.doc_type
    assert restored.source_url == meta.source_url
    assert restored.published_date == meta.published_date
    assert restored.effective_date == meta.effective_date
    assert restored.expiry_date == meta.expiry_date
    assert restored.issuer_unit == meta.issuer_unit
    assert restored.audience == meta.audience
    assert restored.tags == meta.tags
    assert restored.attachments == meta.attachments


# ---------------------------------------------------------------------------
# Property 2: Chunk serialization round-trip
# **Feature: knowledge-base-enhancement, Property 2: Chunk serialization round-trip**
# **Validates: Requirements 2.4**
# ---------------------------------------------------------------------------

@given(chunk=chunk_strategy())
@settings(max_examples=100)
def test_chunk_round_trip(chunk: DocumentChunk):
    """
    **Feature: knowledge-base-enhancement, Property 2: Chunk serialization round-trip**
    **Validates: Requirements 2.4**
    For any DocumentChunk, to_dict() then from_dict() produces an equivalent object.
    """
    restored = DocumentChunk.from_dict(chunk.to_dict())
    assert restored.chunk_id == chunk.chunk_id
    assert restored.doc_id == chunk.doc_id
    assert restored.chunk_index == chunk.chunk_index
    assert restored.text == chunk.text
    assert restored.heading == chunk.heading
    assert restored.metadata.doc_id == chunk.metadata.doc_id
    assert restored.metadata.doc_type == chunk.metadata.doc_type


# ---------------------------------------------------------------------------
# Property 3: Chunking preserves all text tokens
# **Feature: knowledge-base-enhancement, Property 3: Chunking preserves all text tokens**
# **Validates: Requirements 2.1, 2.2**
# ---------------------------------------------------------------------------

_plain_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" \n"),
    min_size=1,
    max_size=500,
)

@given(text=_plain_text, doc_type=_doc_type)
@settings(max_examples=100)
def test_chunking_preserves_tokens(text: str, doc_type: str):
    """
    **Feature: knowledge-base-enhancement, Property 3: Chunking preserves all text tokens**
    **Validates: Requirements 2.1, 2.2**
    For any document text, all non-whitespace tokens appear in the combined chunk texts.
    """
    chunker = StructureAwareChunker(max_chunk_size=256)
    meta = DocumentMetadata(doc_id="test-doc", doc_type=doc_type)
    chunks = chunker.chunk("test-doc", text, meta)

    original_tokens = set(text.split())
    if not original_tokens:
        return  # empty text produces no chunks — valid

    combined = " ".join(c.text for c in chunks)
    combined_tokens = set(combined.split())

    assert original_tokens == combined_tokens, (
        f"Tokens lost during chunking.\n"
        f"Missing: {original_tokens - combined_tokens}"
    )


# ---------------------------------------------------------------------------
# Property 4: Chunk metadata inheritance
# **Feature: knowledge-base-enhancement, Property 4: Chunk metadata inheritance**
# **Validates: Requirements 2.3**
# ---------------------------------------------------------------------------

@given(meta=metadata_strategy(), text=_plain_text)
@settings(max_examples=100)
def test_chunk_metadata_inheritance(meta: DocumentMetadata, text: str):
    """
    **Feature: knowledge-base-enhancement, Property 4: Chunk metadata inheritance**
    **Validates: Requirements 2.3**
    Every chunk produced from a document carries the same doc_id and doc_type as the parent.
    """
    chunker = StructureAwareChunker(max_chunk_size=256)
    chunks = chunker.chunk(meta.doc_id, text, meta)
    for chunk in chunks:
        assert chunk.doc_id == meta.doc_id
        assert chunk.metadata.doc_id == meta.doc_id
        assert chunk.metadata.doc_type == meta.doc_type


# ---------------------------------------------------------------------------
# Property 5: Ingestion idempotence
# **Feature: knowledge-base-enhancement, Property 5: Ingestion idempotence**
# **Validates: Requirements 3.5**
# ---------------------------------------------------------------------------

@given(meta=metadata_strategy(), text=_plain_text)
@settings(max_examples=100)
def test_ingestion_idempotence(meta: DocumentMetadata, text: str):
    """
    **Feature: knowledge-base-enhancement, Property 5: Ingestion idempotence**
    **Validates: Requirements 3.5**
    Adding the same document twice produces the same chunk count as adding it once.
    """
    kb = KnowledgeBase()
    kb.add_document(meta.doc_id, text, meta)
    count_after_first = len(kb.get_chunks(meta.doc_id))

    kb.add_document(meta.doc_id, text, meta)
    count_after_second = len(kb.get_chunks(meta.doc_id))

    assert count_after_first == count_after_second


# ---------------------------------------------------------------------------
# Property 6: Metadata filter correctness
# **Feature: knowledge-base-enhancement, Property 6: Metadata filter correctness**
# **Validates: Requirements 4.1, 4.2, 4.3**
# ---------------------------------------------------------------------------

def _build_kb_with_metas(metas):
    kb = KnowledgeBase()
    for meta in metas:
        kb.add_document(meta.doc_id, "sample text for " + meta.doc_id, meta)
    return kb


@given(
    metas=st.lists(metadata_strategy(), min_size=1, max_size=10),
    filter_doc_type=st.one_of(st.none(), _doc_type),
    filter_eff_after=_iso_date,
    filter_eff_before=_iso_date,
)
@settings(max_examples=100)
def test_metadata_filter_correctness(metas, filter_doc_type, filter_eff_after, filter_eff_before):
    """
    **Feature: knowledge-base-enhancement, Property 6: Metadata filter correctness**
    **Validates: Requirements 4.1, 4.2, 4.3**
    Every doc_id returned by filter_documents() satisfies all non-null filter criteria.
    """
    # Deduplicate by doc_id (keep last)
    seen = {}
    for m in metas:
        seen[m.doc_id] = m
    unique_metas = list(seen.values())

    kb = _build_kb_with_metas(unique_metas)
    engine = MetadataFilterEngine()

    f = MetadataFilter(
        doc_type=filter_doc_type,
        effective_after=filter_eff_after,
        effective_before=filter_eff_before,
    )

    all_ids = [m.doc_id for m in unique_metas]
    result_ids = engine.apply(f, all_ids, kb)

    for doc_id in result_ids:
        meta = kb.get_metadata(doc_id)
        assert meta is not None

        if f.doc_type is not None:
            assert meta.doc_type == f.doc_type

        if f.effective_after is not None:
            assert meta.effective_date is not None
            assert meta.effective_date >= f.effective_after

        if f.effective_before is not None:
            assert meta.effective_date is not None
            assert meta.effective_date <= f.effective_before


# ---------------------------------------------------------------------------
# Property 7: KB export/import round-trip
# **Feature: knowledge-base-enhancement, Property 7: KB export/import round-trip**
# **Validates: Requirements 6.3**
# ---------------------------------------------------------------------------

@given(
    metas=st.lists(metadata_strategy(), min_size=1, max_size=5),
)
@settings(max_examples=50)
def test_kb_export_import_round_trip(metas):
    """
    **Feature: knowledge-base-enhancement, Property 7: KB export/import round-trip**
    **Validates: Requirements 6.3**
    Exporting a KB to JSON and importing into a new KB produces an equivalent KB.
    """
    seen = {}
    for m in metas:
        seen[m.doc_id] = m
    unique_metas = list(seen.values())

    kb1 = KnowledgeBase()
    for meta in unique_metas:
        kb1.add_document(meta.doc_id, "text for " + meta.doc_id, meta)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        kb1.export_json(tmp_path)

        kb2 = KnowledgeBase()
        kb2.import_json(tmp_path)

        s1 = kb1.stats()
        s2 = kb2.stats()
        assert s1["total_documents"] == s2["total_documents"]
        assert s1["total_chunks"] == s2["total_chunks"]

        for meta in unique_metas:
            m1 = kb1.get_metadata(meta.doc_id)
            m2 = kb2.get_metadata(meta.doc_id)
            assert m1 is not None and m2 is not None
            assert m1.doc_type == m2.doc_type
            assert m1.effective_date == m2.effective_date
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property 8: Stats consistency + read-only invariant
# **Feature: knowledge-base-enhancement, Property 8: Stats consistency**
# **Validates: Requirements 5.1, 5.3**
# ---------------------------------------------------------------------------

@given(
    metas=st.lists(metadata_strategy(), min_size=1, max_size=8),
)
@settings(max_examples=100)
def test_stats_consistency(metas):
    """
    **Feature: knowledge-base-enhancement, Property 8: Stats consistency + read-only invariant**
    **Validates: Requirements 5.1, 5.3**
    stats()["total_chunks"] equals sum of chunk counts per document,
    and calling stats() does not modify the KB.
    """
    seen = {}
    for m in metas:
        seen[m.doc_id] = m
    unique_metas = list(seen.values())

    kb = KnowledgeBase()
    for meta in unique_metas:
        kb.add_document(meta.doc_id, "some text content " + meta.doc_id, meta)

    stats = kb.stats()

    # Consistency: total_chunks == sum of per-doc chunk counts
    expected_total = sum(len(kb.get_chunks(m.doc_id)) for m in unique_metas)
    assert stats["total_chunks"] == expected_total

    # Read-only: calling stats() again gives same result
    stats2 = kb.stats()
    assert stats2["total_documents"] == stats["total_documents"]
    assert stats2["total_chunks"] == stats["total_chunks"]
