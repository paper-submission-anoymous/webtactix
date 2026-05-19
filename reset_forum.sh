#!/usr/bin/env bash
set -euo pipefail

FORUM_PORT=9999
FORUM_NAME=forum
FORUM_IMAGE=postmill-populated-exposed-withimg
FORUM_URL="http://143.215.184.110:${FORUM_PORT}"

wait_forum_db() {
  echo "    - waiting PostgreSQL in container: $FORUM_NAME"
  for i in {1..120}; do
    if docker exec "$FORUM_NAME" bash -c "pg_isready -q" >/dev/null 2>&1; then
      echo "      DB is ready."
      return 0
    fi
    sleep 1
  done
  echo "      ERROR: DB not ready in $FORUM_NAME after 120s" >&2
  return 1
}

echo "[1/3] Stopping and removing forum container..."
docker rm -f "$FORUM_NAME" 2>/dev/null || true

echo "[2/3] Starting fresh forum container from clean image..."
docker run --name "$FORUM_NAME" -p "${FORUM_PORT}:80" -d "$FORUM_IMAGE"

wait_forum_db

echo "[3/3] Waiting for forum to be ready at ${FORUM_URL} ..."
for i in $(seq 1 60); do
    body=$(curl -sf --max-time 10 "${FORUM_URL}/all" 2>/dev/null || true)
    if echo "$body" | grep -q 'class="submission__header"'; then
        echo "    Forum is ready with content (${i}s)"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Forum did not serve post content after 60s" >&2
exit 1
