-- Initialize database schema for BeIR retrieval system
-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents table (corpus)
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    title TEXT,
    text TEXT,
    full_text TEXT,
    embedding vector(384),  -- For sentence-transformers/all-MiniLM-L6-v2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_name VARCHAR(50)
);

-- Create indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- BM25 full-text search index
CREATE INDEX IF NOT EXISTS idx_documents_fulltext ON documents 
USING gin (to_tsvector('english', full_text));

-- Regular text search index for BM25
CREATE INDEX IF NOT EXISTS idx_documents_text_gin ON documents 
USING gin (full_text gin_trgm_ops);

-- Queries table (for logging)
CREATE TABLE IF NOT EXISTS queries (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(255),
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    dataset_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Results table (for caching)
CREATE TABLE IF NOT EXISTS retrieval_results (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(255),
    doc_id VARCHAR(255),
    method VARCHAR(20),  -- 'bm25', 'dense', 'hybrid'
    score FLOAT,
    rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_query ON retrieval_results(query_id, method);

-- Ground truth table (qrels)
CREATE TABLE IF NOT EXISTS qrels (
    query_id VARCHAR(255),
    doc_id VARCHAR(255),
    relevance INTEGER,
    dataset_name VARCHAR(50),
    PRIMARY KEY (query_id, doc_id)
);

-- Evaluation metrics table
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100),
    method VARCHAR(20),
    dataset_name VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    num_queries INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create function for BM25-like scoring using ts_rank_cd
CREATE OR REPLACE FUNCTION bm25_search(
    search_query TEXT,
    k1 FLOAT DEFAULT 1.2,
    b FLOAT DEFAULT 0.75
)
RETURNS TABLE (
    doc_id VARCHAR(255),
    score FLOAT,
    title TEXT,
    text TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        ts_rank_cd(to_tsvector('english', d.full_text), 
                   plainto_tsquery('english', search_query)) AS score,
        d.title,
        d.text
    FROM documents d
    WHERE to_tsvector('english', d.full_text) @@ plainto_tsquery('english', search_query)
    ORDER BY score DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function for vector similarity search
CREATE OR REPLACE FUNCTION vector_search(
    query_embedding vector(384),
    top_k INTEGER DEFAULT 10
)
RETURNS TABLE (
    doc_id VARCHAR(255),
    score FLOAT,
    title TEXT,
    text TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        1 - (d.embedding <=> query_embedding) AS score,  -- Cosine similarity
        d.title,
        d.text
    FROM documents d
    WHERE d.embedding IS NOT NULL
    ORDER BY d.embedding <=> query_embedding
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

-- Create function for hybrid search with RRF
CREATE OR REPLACE FUNCTION hybrid_search(
    search_query TEXT,
    query_embedding vector(384),
    alpha FLOAT DEFAULT 0.5,
    k_param INTEGER DEFAULT 60,
    top_k INTEGER DEFAULT 10
)
RETURNS TABLE (
    doc_id VARCHAR(255),
    hybrid_score FLOAT,
    bm25_rank INTEGER,
    vector_rank INTEGER,
    title TEXT,
    text TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH bm25_results AS (
        SELECT 
            d.id,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', d.full_text), 
                                                    plainto_tsquery('english', search_query)) DESC) as rank
        FROM documents d
        WHERE to_tsvector('english', d.full_text) @@ plainto_tsquery('english', search_query)
        LIMIT 100
    ),
    vector_results AS (
        SELECT 
            d.id,
            ROW_NUMBER() OVER (ORDER BY d.embedding <=> query_embedding) as rank
        FROM documents d
        WHERE d.embedding IS NOT NULL
        LIMIT 100
    )
    SELECT 
        COALESCE(b.id, v.id) as doc_id,
        (COALESCE(alpha / (k_param + b.rank), 0) + 
         COALESCE((1 - alpha) / (k_param + v.rank), 0)) as hybrid_score,
        b.rank::INTEGER as bm25_rank,
        v.rank::INTEGER as vector_rank,
        d.title,
        d.text
    FROM bm25_results b
    FULL OUTER JOIN vector_results v ON b.id = v.id
    JOIN documents d ON d.id = COALESCE(b.id, v.id)
    ORDER BY hybrid_score DESC
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO beir_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO beir_user;

-- Insert sample statistics view
CREATE OR REPLACE VIEW corpus_stats AS
SELECT 
    dataset_name,
    COUNT(*) as num_documents,
    AVG(LENGTH(full_text)) as avg_doc_length,
    MAX(LENGTH(full_text)) as max_doc_length,
    MIN(LENGTH(full_text)) as min_doc_length
FROM documents
GROUP BY dataset_name;

COMMENT ON TABLE documents IS 'BeIR corpus documents with embeddings';
COMMENT ON TABLE queries IS 'Search queries with embeddings';
COMMENT ON TABLE qrels IS 'Ground truth relevance judgments';
COMMENT ON TABLE evaluation_metrics IS 'Experiment results and metrics';
