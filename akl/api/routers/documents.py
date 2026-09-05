"""Documents API (PRD §10.3): upload, list, detail, chunks, soft/hard delete."""

from __future__ import annotations

import mimetypes
import uuid
from typing import Annotated, Any, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    Query,
    Request,
    UploadFile,
)
from sqlalchemy import func, select

from akl.api import metrics
from akl.api.deps import AppState, get_request_id, get_state, rate_limited, scoped
from akl.api.schemas.common import Page
from akl.api.schemas.documents import (
    ChunkOut,
    DeleteResponse,
    DocumentDetail,
    DocumentSummary,
    DocumentVersionOut,
    UploadItem,
    UploadResponse,
)
from akl.db.models import Chunk, Document
from akl.db.repositories.chunks import ChunkRepository
from akl.db.repositories.documents import DocumentRepository
from akl.errors import AKLError
from akl.ids import canonicalize_uri
from akl.lakehouse.bronze import BronzeStore
from akl.pipelines.local import run_post_ingest
from akl.security.principal import Principal

router = APIRouter(prefix="/v1", tags=["documents"])

_EXT_SOURCE: dict[str, tuple[str, str]] = {
    "pdf": ("pdf", "application/pdf"),
    "md": ("markdown", "text/markdown"),
    "mdx": ("markdown", "text/markdown"),
    "markdown": ("markdown", "text/markdown"),
    "html": ("html", "text/html"),
    "htm": ("html", "text/html"),
}


class UnsupportedUploadError(AKLError):
    code = "AKL-E3003"
    http_status = 415
    retryable = False


class UploadTooLargeError(AKLError):
    code = "AKL-E3001"
    http_status = 413
    retryable = False


class DocumentNotFoundError(AKLError):
    code = "AKL-E6030"
    http_status = 404
    retryable = False


def _visible(doc: Document, principal: Principal) -> bool:
    return principal.can_read(doc.security_level, list(doc.allowed_groups or []))


def _summary(doc: Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=str(doc.document_id),
        canonical_source_uri=doc.canonical_source_uri,
        source_type=doc.source_type,
        connector_id=doc.connector_id,
        title=doc.title,
        status=doc.status,
        security_level=doc.security_level,
        allowed_groups=list(doc.allowed_groups or []),
        is_duplicate_of=str(doc.is_duplicate_of) if doc.is_duplicate_of else None,
        latest_content_sha256=doc.latest_content_sha256,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def _background_pipeline(state: AppState, run_id: str, document_ids: list[uuid.UUID]) -> None:
    state.jobs.update(run_id, state="running")
    if state.db is None:
        state.jobs.update(
            run_id, state="failed", errors=[{"code": "AKL-E3023", "error": "no database"}]
        )
        return

    def on_stage(stage: str, report: Any) -> None:
        state.jobs.update(run_id, stages=list(report.stages))

    report = run_post_ingest(
        state.settings, state.db, document_ids=document_ids, run_id=run_id, on_stage=on_stage
    )
    state.jobs.update(
        run_id,
        state="failed" if report.errors else "succeeded",
        stages=report.stages,
        report=report.as_dict(),
        errors=report.errors,
    )
    reload = getattr(state.rag, "reload_indexes", None)
    if callable(reload) and not report.errors:
        with state.lock:
            reload()


@router.post(
    "/documents",
    response_model=UploadResponse,
    status_code=202,
    dependencies=[Depends(rate_limited("default"))],
)
def upload(
    request: Request,
    background: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="1–20 PDF/Markdown/HTML files")],
    security_level: Annotated[Literal["public", "internal", "restricted"], Form()] = "internal",
    allowed_groups: Annotated[str, Form(description="comma-separated")] = "",
    process: Annotated[Literal["async", "sync"], Form()] = "async",
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("documents:write")),
    request_id: str = Depends(get_request_id),
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> UploadResponse:
    if not 1 <= len(files) <= 20:
        raise AKLError("upload 1–20 files", details={"count": len(files)})
    if state.db is None or state.rag is None:
        raise AKLError("service not ready", details={"reason": state.ready_error})
    max_bytes = state.settings.api.max_upload_mb * 1024 * 1024
    run_id = f"api-{request_id[:12]}" if idempotency_key is None else f"api-{idempotency_key[:32]}"
    store = BronzeStore(state.rag.io)
    items: list[UploadItem] = []
    doc_ids: list[uuid.UUID] = []
    groups = [g.strip() for g in allowed_groups.split(",") if g.strip()]
    with state.lock, state.db.session() as session:
        repo = DocumentRepository(session)
        rows = []
        for f in files:
            name = f.filename or "upload.bin"
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in _EXT_SOURCE:
                raise UnsupportedUploadError(
                    f"unsupported file type: {name}", details={"allowed": sorted(_EXT_SOURCE)}
                )
            source_type, default_mime = _EXT_SOURCE[ext]
            data = f.file.read()
            if len(data) > max_bytes:
                raise UploadTooLargeError(
                    f"{name} exceeds {state.settings.api.max_upload_mb} MiB",
                    details={"size": len(data)},
                )
            mime = f.content_type or mimetypes.guess_type(name)[0] or default_mime
            put = store.put_raw(data, source_type=source_type, mime_type=mime, filename=name)
            uri = f"upload://{principal.subject}/{name}"
            recorded = repo.record_bronze(
                canonical_source_uri=canonicalize_uri(uri),
                source_type=source_type,
                connector_id="api-upload",
                content_sha256=put.content_sha256,
                bronze_object_key=put.object_key,
                run_id=run_id,
                security_level=security_level,
                allowed_groups=groups,
                title=name.rsplit(".", 1)[0],
            )
            rows.append(
                BronzeStore.build_manifest_row(
                    source_uri=uri,
                    source_type=source_type,
                    put=put,
                    connector_name="api-upload",
                    connector_version="1.0.0",
                    run_id=run_id,
                    mime_type=mime,
                    source_metadata={"filename": name, "uploader": principal.subject},
                )
            )
            metrics.UPLOADS.labels(
                source_type=source_type, outcome="dedup" if put.deduplicated else "stored"
            ).inc()
            metrics.UPLOAD_BYTES.labels(source_type=source_type).inc(len(data))
            items.append(
                UploadItem(
                    filename=name,
                    document_id=str(recorded.document_id),
                    content_sha256=put.content_sha256,
                    status="bronze",
                    deduplicated=put.deduplicated,
                    size_bytes=len(data),
                )
            )
            doc_ids.append(recorded.document_id)
        store.write_manifest(rows, run_id=run_id)
    state.jobs.create(run_id)
    if process == "sync":
        _background_pipeline(state, run_id, doc_ids)
        job = state.jobs.get(run_id)
        for item in items:
            item.status = "silver" if job and job.state == "succeeded" else "bronze"
    else:
        background.add_task(_background_pipeline, state, run_id, doc_ids)
    return UploadResponse(
        items=items,
        run_id=run_id,
        processing=process,
        status_url=str(request.url_for("job_status", run_id=run_id)),
    )


@router.get("/documents", response_model=Page[DocumentSummary])
def list_documents(
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("search:read")),
    source_type: str | None = None,
    status: str | None = None,
    q: str | None = Query(default=None, description="substring of title or URI"),
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> Page[DocumentSummary]:
    if state.db is None:
        return Page(items=[], next_cursor=None, total_estimate=0)
    with state.db.session() as s:
        stmt = select(Document).order_by(Document.updated_at.desc(), Document.document_id)
        if source_type:
            stmt = stmt.where(Document.source_type == source_type)
        if status:
            stmt = stmt.where(Document.status == status)
        if q:
            like = f"%{q}%"
            stmt = stmt.where(
                Document.title.ilike(like) | Document.canonical_source_uri.ilike(like)
            )
        docs = [d for d in s.scalars(stmt) if _visible(d, principal)]
        start = int(cursor) if cursor and cursor.isdigit() else 0
        page = docs[start : start + limit]
        nxt = str(start + limit) if start + limit < len(docs) else None
        return Page(items=[_summary(d) for d in page], next_cursor=nxt, total_estimate=len(docs))


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def document_detail(
    document_id: str,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("search:read")),
) -> DocumentDetail:
    did = _uuid(document_id)
    if state.db is None:
        raise DocumentNotFoundError("document not found")
    with state.db.session() as s:
        repo = DocumentRepository(s)
        doc = repo.get(did)
        if doc is None or not _visible(doc, principal):
            raise DocumentNotFoundError("document not found", details={"document_id": document_id})
        versions = repo.versions(did)
        chunk_count = int(
            s.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_id == did, Chunk.is_current.is_(True))
            )
            or 0
        )
        return DocumentDetail(
            **_summary(doc).model_dump(),
            current_version_id=str(doc.current_version_id) if doc.current_version_id else None,
            versions=[
                DocumentVersionOut(
                    document_version_id=str(v.document_version_id),
                    content_sha256=v.content_sha256,
                    parser_name=v.parser_name,
                    parser_version=v.parser_version,
                    quality_score=v.quality_score,
                    language=v.language,
                    word_count=v.word_count,
                    parsed_at=v.parsed_at,
                    fetched_at=v.fetched_at,
                )
                for v in versions
            ],
            chunk_count=chunk_count,
            metadata=dict(doc.metadata_ or {}),
        )


@router.get("/documents/{document_id}/chunks", response_model=Page[ChunkOut])
def document_chunks(
    document_id: str,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("search:read")),
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> Page[ChunkOut]:
    did = _uuid(document_id)
    if state.db is None or state.rag is None:
        raise DocumentNotFoundError("document not found")
    with state.db.session() as s:
        doc = DocumentRepository(s).get(did)
        if doc is None or not _visible(doc, principal):
            raise DocumentNotFoundError("document not found", details={"document_id": document_id})
    with state.lock:
        table = state.rag.gold.active_units(where=f"document_id = '{did}'")
    rows = table.to_pylist()
    start = int(cursor) if cursor and cursor.isdigit() else 0
    page = rows[start : start + limit]
    return Page(
        items=[
            ChunkOut(
                chunk_id=str(r["chunk_id"]),
                chunk_index=int(r["chunk_index"]),
                chunk_type=str(r["chunk_type"]),
                heading_breadcrumb=r.get("heading_breadcrumb"),
                text=str(r["text"]),
                token_count=r.get("token_count"),
                quality_score=r.get("quality_score"),
                page_start=r.get("page_start"),
                line_start=r.get("line_start"),
            )
            for r in page
        ],
        next_cursor=str(start + limit) if start + limit < len(rows) else None,
        total_estimate=len(rows),
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(
    document_id: str,
    mode: Literal["soft", "hard"] = "soft",
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("documents:delete")),
    request_id: str = Depends(get_request_id),
    x_confirm: Annotated[str | None, Header(alias="X-Confirm")] = None,
) -> DeleteResponse:
    did = _uuid(document_id)
    if state.db is None or state.rag is None:
        raise DocumentNotFoundError("document not found")
    if mode == "hard" and x_confirm != "hard-delete":
        raise AKLError("hard delete requires header X-Confirm: hard-delete", details={"mode": mode})
    run_id = f"api-{request_id[:12]}"
    with state.lock, state.db.session() as s:
        repo = DocumentRepository(s)
        doc = repo.get(did)
        if doc is None or not _visible(doc, principal):
            raise DocumentNotFoundError("document not found", details={"document_id": document_id})
        from akl.lakehouse.silver import SilverStore

        silver = SilverStore(state.rag.io, state.rag.engine)
        _docs, chunks = silver.tombstone_documents([str(did)], run_id=run_id)
        chunk_ids = [c.chunk_id for c in ChunkRepository(s).current_for_document(did)]
        ChunkRepository(s).retire(chunk_ids, deleted=True)
        repo.set_status(did, "deleted")
        if state.rag.qdrant is not None:
            state.rag.qdrant.delete_documents([str(did)])
        state.rag.exclude_deleted([str(c) for c in chunk_ids])
    return DeleteResponse(
        document_id=document_id, mode=mode, status="deleted", chunks_tombstoned=chunks
    )


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise DocumentNotFoundError("document not found", details={"document_id": value}) from exc
