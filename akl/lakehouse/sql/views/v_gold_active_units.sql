-- Gold rows still backed by current Silver chunks.
WITH latest AS (
    SELECT u.*
    FROM {{ retrieval_units }} AS u
    QUALIFY row_number() OVER (PARTITION BY u.chunk_id ORDER BY u.created_at DESC) = 1
)
SELECT l.*
FROM latest AS l
JOIN v_current_chunks AS c ON c.chunk_id = l.chunk_id
