-- v_current_chunks
WITH latest AS (
    SELECT c.*
    FROM {{ chunks }} AS c
    QUALIFY row_number() OVER (
        PARTITION BY c.chunk_id
        ORDER BY c.created_at DESC
    ) = 1
)
SELECT l.*
FROM latest AS l
JOIN v_current_documents AS d
  ON d.document_version_id = l.document_version_id
WHERE l.is_current
  AND NOT l.is_deleted
