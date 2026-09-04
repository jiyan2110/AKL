-- v_current_documents
WITH ranked AS (
    SELECT
        d.*,
        row_number() OVER (
            PARTITION BY d.document_id
            ORDER BY d.parsed_at DESC, d.document_version_id DESC
        ) AS _rn
    FROM {{ documents }} AS d
)
SELECT * EXCLUDE (_rn)
FROM ranked
WHERE _rn = 1
  AND NOT is_deleted
