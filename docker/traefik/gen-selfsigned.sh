#!/usr/bin/env bash
# Generates a self-signed cert for local/staging use when AKL_TLS_MODE=selfsigned.
# Real internet-facing deployments should use the ACME (Let's Encrypt) resolver instead.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/dynamic" && pwd)"
DOMAIN="${1:-localhost}"
openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
  -keyout "$DIR/selfsigned.key" -out "$DIR/selfsigned.crt" \
  -subj "/CN=${DOMAIN}" -addext "subjectAltName=DNS:${DOMAIN}"
echo "wrote $DIR/selfsigned.crt and selfsigned.key for ${DOMAIN}"
