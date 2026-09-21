#!/usr/bin/env sh
set -eu

COMPOSE="docker compose"
FILES_DEV="-f docker-compose.yml -f docker-compose.dev.yml"
FILES_PROD="-f docker-compose.yml -f docker-compose.prod.yml"
# --with-minio: прод-стек с локальным MinIO вместо внешнего S3
if [ "${1:-}" = "--with-minio" ]; then
  FILES_PROD="$FILES_PROD -f docker-compose.minio.yml"
  shift
fi

usage() {
  cat >&2 <<USAGE
Usage: $0 [--with-minio] [--with-libreoffice|--without-libreoffice] {up|dev|migrate|rebuild|down|restart|logs|test}

  --with-libreoffice     образ с LibreOffice — нужен для PDF-отчётов (по умолчанию, см. WITH_LIBREOFFICE в .env)
  --without-libreoffice  без него: отчёт только в docx

Примеры:
  sh $0 dev                        # dev-стек (api + worker + db + redis + minio)
  sh $0 dev --scale backend-worker=2
  sh $0 test                       # pytest внутри backend-api
USAGE
  exit 1
}

case "${1:-}" in
  --with-libreoffice)
    export WITH_LIBREOFFICE=1
    shift
    ;;
  --without-libreoffice)
    export WITH_LIBREOFFICE=0
    shift
    ;;
esac

cmd="${1:-up}"
[ $# -gt 0 ] && shift

case "$cmd" in
  up)
    $COMPOSE $FILES_PROD up -d --build --remove-orphans "$@"
    ;;
  dev)
    $COMPOSE $FILES_DEV up -d --build --remove-orphans "$@"
    ;;
  migrate)
    $COMPOSE $FILES_DEV run --rm backend-migrate
    ;;
  rebuild)
    $COMPOSE $FILES_DEV build --no-cache
    $COMPOSE $FILES_DEV up -d --force-recreate --remove-orphans
    ;;
  down)
    $COMPOSE $FILES_DEV down --remove-orphans
    ;;
  restart)
    $COMPOSE $FILES_DEV restart
    ;;
  logs)
    $COMPOSE $FILES_DEV logs -f --tail=200 "$@"
    ;;
  test)
    $COMPOSE $FILES_DEV exec backend-api pytest "$@"
    ;;
  *)
    usage
    ;;
esac
