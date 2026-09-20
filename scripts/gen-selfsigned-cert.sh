#!/usr/bin/env sh
# Генерирует самоподписанный сертификат для nginx (порт 443) в .docker/nginx/certs/.
# Для прода замените fullchain.pem / privkey.pem на настоящие (например, от Let's Encrypt).
set -eu

DOMAIN="${1:-localhost}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/.docker/nginx/certs"
mkdir -p "$DIR"

openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
  -keyout "$DIR/privkey.pem" -out "$DIR/fullchain.pem" \
  -subj "/CN=$DOMAIN" \
  -addext "subjectAltName=DNS:$DOMAIN,DNS:localhost,IP:127.0.0.1"

echo "Сертификат для $DOMAIN записан в $DIR"
