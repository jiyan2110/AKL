-- Silver to Gold retrieval-unit projection.
SELECT
    c.chunk_id, c.chunk_key, c.lineage_id, c.chunk_checksum, c.embedded_text_sha256,
    c.document_id, c.document_version_id, c.chunk_index, c.source_type,
    d.canonical_source_uri, d.source_uri, d.title, c.heading_path,
    array_to_string(c.heading_path, ' › ') AS heading_breadcrumb,
    c.chunk_type, c.code_language, c.text, c.context_prefix, c.token_count,
    c.page_start, c.page_end, c.line_start, c.line_end, c.security_level,
    c.allowed_groups,
    list_extract(map_extract(d.metadata, 'git.repo'), 1) AS repo,
    list_extract(map_extract(d.metadata, 'git.branch'), 1) AS branch,
    list_extract(map_extract(d.metadata, 'git.path'), 1) AS path,
    coalesce(try_cast(list_extract(map_extract(d.metadata, 'source.updated_at'), 1) AS TIMESTAMPTZ), d.parsed_at) AS document_updated_at,
    c.quality_score, c.quality_flags, c.language,
    '{{ gold_snapshot_id }}' AS gold_snapshot_id,
    now() AT TIME ZONE 'UTC' AS created_at
FROM v_current_chunks AS c
JOIN v_current_documents AS d
  ON d.document_id = c.document_id
 AND d.document_version_id = c.document_version_id
LEFT JOIN {{ retrieval_units }} AS u ON u.chunk_id = c.chunk_id
WHERE u.chunk_id IS NULL
  AND c.quality_score >= {{ chunk_quality_min }}
  AND d.quality_score >= {{ doc_quality_min }}
  AND d.is_duplicate_of IS NULL
  AND NOT coalesce(list_contains(c.quality_flags, 'low_quality'), false)
