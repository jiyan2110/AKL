-- Embedding backlog for one embedding version.
WITH emb AS (
    SELECT e.chunk_id, e.embedding_version, e.embedded_text_sha256, e.embedded_at
    FROM {{ chunk_embeddings }} AS e
    WHERE e.embedding_version = '{{ embedding_version }}'
    QUALIFY row_number() OVER (PARTITION BY e.chunk_id ORDER BY e.embedded_at DESC) = 1
)
SELECT
    u.chunk_id, u.document_id, u.source_type, u.security_level,
    u.embedded_text_sha256, u.gold_snapshot_id,
    e.embedding_version, e.embedded_at,
    (e.chunk_id IS NOT NULL) AS has_embedding,
    (e.chunk_id IS NOT NULL AND e.embedded_text_sha256 IS DISTINCT FROM u.embedded_text_sha256) AS stale_embedding
FROM v_gold_active_units AS u
LEFT JOIN emb AS e ON e.chunk_id = u.chunk_id
