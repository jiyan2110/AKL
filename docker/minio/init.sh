#!/bin/sh
set -eu

: "${MINIO_ENDPOINT:?}" "${MINIO_ROOT_USER:?}" "${MINIO_ROOT_PASSWORD:?}"
: "${AKL_S3_BUCKET:?}" "${AKL_S3_ACCESS_KEY:?}" "${AKL_S3_SECRET_KEY:?}"
QUARANTINE_DAYS="${AKL_QUARANTINE_RETENTION_DAYS:-90}"
BACKUP_DAYS="${AKL_BACKUP_RETENTION_DAYS:-14}"
ALIAS="akl"
POLICY_NAME="akl-lakehouse-rw"

echo "[akl-minio-init] waiting for ${MINIO_ENDPOINT}"
i=0
until mc alias set "$ALIAS" "$MINIO_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -ge 30 ]; then
    echo "[akl-minio-init] ERROR: MinIO not reachable after 30 attempts" >&2
    exit 1
  fi
  sleep 2
done

mc mb --ignore-existing "$ALIAS/$AKL_S3_BUCKET"
mc version enable "$ALIAS/$AKL_S3_BUCKET"
rules="$(mc ilm rule ls "$ALIAS/$AKL_S3_BUCKET" 2>/dev/null || true)"
case "$rules" in
  *quarantine/*) ;;
  *)
  mc ilm rule add --prefix "quarantine/" --expire-days "$QUARANTINE_DAYS" "$ALIAS/$AKL_S3_BUCKET"
  ;;
esac
case "$rules" in
  *backups/*) ;;
  *)
  mc ilm rule add --prefix "backups/" --expire-days "$BACKUP_DAYS" "$ALIAS/$AKL_S3_BUCKET"
  ;;
esac

cat > /tmp/policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation", "s3:ListBucketMultipartUploads", "s3:GetBucketVersioning"], "Resource": ["arn:aws:s3:::${AKL_S3_BUCKET}"]},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"], "Resource": ["arn:aws:s3:::${AKL_S3_BUCKET}/*"]}
  ]
}
EOF
mc admin policy create "$ALIAS" "$POLICY_NAME" /tmp/policy.json >/dev/null 2>&1 || true
mc admin user add "$ALIAS" "$AKL_S3_ACCESS_KEY" "$AKL_S3_SECRET_KEY" >/dev/null 2>&1 || true
mc admin policy attach "$ALIAS" "$POLICY_NAME" --user "$AKL_S3_ACCESS_KEY" >/dev/null 2>&1 || true

for prefix in bronze/raw bronze/manifest silver/documents silver/chunks gold/retrieval_units gold/chunk_embeddings gold/indexes gold/eval quarantine inbox/pdf mlflow backups; do
  printf '' | mc pipe "$ALIAS/$AKL_S3_BUCKET/$prefix/.keep" >/dev/null
done

mc alias set aklsvc "$MINIO_ENDPOINT" "$AKL_S3_ACCESS_KEY" "$AKL_S3_SECRET_KEY" >/dev/null
mc ls "aklsvc/$AKL_S3_BUCKET" >/dev/null
echo "[akl-minio-init] done"
