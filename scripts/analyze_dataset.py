"""
Analyze BeIR NFCorpus Dataset Statistics
"""
from beir.datasets.data_loader import GenericDataLoader
import numpy as np

# Load dataset
corpus, queries, qrels = GenericDataLoader(data_folder='./data/beir_datasets/nfcorpus').load(split="test")

print("="*80)
print("BEIR NFCORPUS DATASET ANALYSIS")
print("="*80)

# Basic statistics
print(f"\n📊 BASIC STATISTICS:")
print(f"  Total documents: {len(corpus):,}")
print(f"  Total queries: {len(queries):,}")
print(f"  Total relevance judgments: {sum(len(v) for v in qrels.values()):,}")

# Document analysis
doc_lengths = [len(doc['text']) for doc in corpus.values()]
title_lengths = [len(doc['title']) for doc in corpus.values()]

print(f"\n📄 DOCUMENT STATISTICS:")
print(f"  Average document length: {np.mean(doc_lengths):.0f} chars")
print(f"  Median document length: {np.median(doc_lengths):.0f} chars")
print(f"  Min document length: {np.min(doc_lengths):,} chars")
print(f"  Max document length: {np.max(doc_lengths):,} chars")
print(f"  Std document length: {np.std(doc_lengths):.0f} chars")

print(f"\n📝 TITLE STATISTICS:")
print(f"  Average title length: {np.mean(title_lengths):.0f} chars")
print(f"  Median title length: {np.median(title_lengths):.0f} chars")
print(f"  Max title length: {np.max(title_lengths):,} chars")

# Query analysis
query_lengths = [len(q) for q in queries.values()]

print(f"\n❓ QUERY STATISTICS:")
print(f"  Average query length: {np.mean(query_lengths):.0f} chars")
print(f"  Median query length: {np.median(query_lengths):.0f} chars")
print(f"  Min query length: {np.min(query_lengths):,} chars")
print(f"  Max query length: {np.max(query_lengths):,} chars")

# Relevance analysis
relevant_per_query = [len(docs) for docs in qrels.values()]
relevance_scores = []
for query_rels in qrels.values():
    for score in query_rels.values():
        relevance_scores.append(score)

print(f"\n⭐ RELEVANCE STATISTICS:")
print(f"  Avg relevant docs per query: {np.mean(relevant_per_query):.2f}")
print(f"  Median relevant docs per query: {np.median(relevant_per_query):.0f}")
print(f"  Min relevant docs per query: {np.min(relevant_per_query)}")
print(f"  Max relevant docs per query: {np.max(relevant_per_query)}")
print(f"  Unique relevance scores: {sorted(set(relevance_scores))}")

# Sample data
print(f"\n📋 SAMPLE DOCUMENT:")
sample_doc_id = list(corpus.keys())[0]
sample_doc = corpus[sample_doc_id]
print(f"  ID: {sample_doc_id}")
print(f"  Title: {sample_doc['title'][:100]}...")
print(f"  Text: {sample_doc['text'][:200]}...")

print(f"\n📋 SAMPLE QUERY:")
sample_query_id = list(queries.keys())[0]
sample_query = queries[sample_query_id]
print(f"  ID: {sample_query_id}")
print(f"  Query: {sample_query}")
print(f"  Relevant docs: {len(qrels.get(sample_query_id, {}))}")

# Domain analysis
print(f"\n🏥 DOMAIN CHARACTERISTICS:")
medical_keywords = ['patient', 'treatment', 'disease', 'health', 'medical', 
                   'cancer', 'diabetes', 'study', 'clinical', 'risk']
keyword_counts = {kw: 0 for kw in medical_keywords}

for doc in corpus.values():
    text_lower = doc['text'].lower()
    for kw in medical_keywords:
        if kw in text_lower:
            keyword_counts[kw] += 1

print("  Top medical keywords:")
for kw, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"    '{kw}': {count} documents ({count/len(corpus)*100:.1f}%)")

print("\n" + "="*80)
