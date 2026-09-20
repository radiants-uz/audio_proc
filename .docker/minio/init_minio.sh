#!/bin/sh
set -e

# Runs inside quay.io/minio/mc (mc preinstalled). The download branch is kept
# for a bare alpine image, but needs network access to dl.min.io.
if ! command -v mc >/dev/null 2>&1; then
    echo "🛠 Installing dependencies (curl)..."
    apk add --no-cache curl

    echo "⬇️ Downloading MinIO Client (mc)..."
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)  MC_ARCH="amd64" ;;
        aarch64) MC_ARCH="arm64" ;;
        *)       echo "❌ Unknown architecture: $ARCH"; exit 1 ;;
    esac
    curl -f -s -o /usr/bin/mc "https://dl.min.io/client/mc/release/linux-${MC_ARCH}/mc"
    chmod +x /usr/bin/mc
fi


echo "⏳ Waiting for MinIO to be ready..."

until mc alias set minio http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"; do
  echo "MinIO is not ready yet..."
  sleep 5
done

echo "MinIO is ready. Setup MinIO..."


echo "Creating buckets."
mc mb minio/"$S3_STORAGE_BUCKET" --ignore-existing
mc mb minio/"$S3_ASSETS_BUCKET" --ignore-existing

mc ilm add --expire-days 7 minio/"$S3_ASSETS_BUCKET"

echo "Creating users."
mc admin user add minio "$S3_ACCESS_KEY" "$S3_SECRET_KEY"
mc admin user add minio "$S3_USER" "$S3_USER_PASS"


echo "Setup policies."

# Substitute the `bucket_name` placeholder of a policy statement template.
# Plain shell: the mc image ships neither sed nor awk.
statement_for() { # <template> <bucket>
    _tpl=$(cat "$1")
    _out=""
    while [ "${_tpl#*bucket_name}" != "$_tpl" ]; do
        _out="$_out${_tpl%%bucket_name*}$2"
        _tpl="${_tpl#*bucket_name}"
    done
    printf '%s' "$_out$_tpl"
}

if [ -f /policies/users/user-app.json ]; then
    cat > /tmp/user-app.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
  $(statement_for /policies/users/user-app.json "$S3_STORAGE_BUCKET"),
  $(statement_for /policies/users/user-app.json "$S3_ASSETS_BUCKET")
  ]
}
EOF
    mc admin policy remove minio user-app-policy || true
    mc admin policy create minio user-app-policy /tmp/user-app.json
    mc admin policy attach minio user-app-policy --user="$S3_ACCESS_KEY"
else
    echo "Warning: policies/users/user-app.json not found"
fi

if [ -f /policies/users/user-ro.json ]; then
    cat > /tmp/user-client.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
  $(statement_for /policies/users/user-ro.json "$S3_STORAGE_BUCKET"),
  $(statement_for /policies/users/user-ro.json "$S3_ASSETS_BUCKET")
  ]
}
EOF
    mc admin policy remove minio user-client-policy || true
    mc admin policy create minio user-client-policy /tmp/user-client.json
    mc admin policy attach minio user-client-policy --user="$S3_USER"
else
    echo "Warning: policies/users/user-ro.json not found"
fi

mc anonymous set download minio/$S3_ASSETS_BUCKET

echo "MinIO inicialization completed."

exit 0