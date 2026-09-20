#!/bin/sh
set -e

echo "Creating database."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v backend_db="$POSTGRES_BACKEND_DB" \
  -v backend_user="$POSTGRES_BACKEND_USER" \
  -v backend_pass="$POSTGRES_BACKEND_PASSWORD" \
  -v backend_migrator_user="$POSTGRES_BACKEND_MIGRATOR_USER" \
  -v backend_migrator_pass="$POSTGRES_BACKEND_MIGRATOR_PASSWORD" \
  <<-'EOSQL'
    CREATE DATABASE :"backend_db";

    CREATE USER :"backend_user" WITH PASSWORD :'backend_pass';
    CREATE USER :"backend_migrator_user" WITH PASSWORD :'backend_migrator_pass';

    -- Рекомендуется: явные права на подключение
    GRANT CONNECT ON DATABASE :"backend_db" TO :"backend_user";
    GRANT CONNECT ON DATABASE :"backend_db" TO :"backend_migrator_user";

    -- Рекомендуется: сделать мигратора владельцем БД (для миграций)
    ALTER DATABASE :"backend_db" OWNER TO :"backend_migrator_user";
EOSQL

echo "Creating complete."
echo "Setup privileges."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_BACKEND_DB" \
  -v backend_user="$POSTGRES_BACKEND_USER" \
  -v migrator_user="$POSTGRES_BACKEND_MIGRATOR_USER" \
  <<-'EOSQL'

    -- Рекомендуется: мигратор владеет схемой
    ALTER SCHEMA public OWNER TO :"migrator_user";

    -- Права мигратора
    GRANT ALL PRIVILEGES ON SCHEMA public TO :"migrator_user";
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO :"migrator_user";
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO :"migrator_user";
    GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO :"migrator_user";

    -- Права приложения (DML)
    GRANT USAGE ON SCHEMA public TO :"backend_user";
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO :"backend_user";
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO :"backend_user";

    -- Автоправа на новые объекты, создаваемые мигратором
    ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_user" IN SCHEMA public
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"backend_user";
    ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_user" IN SCHEMA public
        GRANT USAGE, SELECT ON SEQUENCES TO :"backend_user";

EOSQL

echo "Initialization complete."
