#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
SHOPPING_PORT=7770
ADMIN_PORT=7780
GITLAB_PORT=8023
FORUM_PORT=9999

SHOPPING_NAME=shopping
ADMIN_NAME=shopping_admin
GITLAB_NAME=gitlab
FORUM_NAME=forum

SHOPPING_IMAGE=shopping_final_0712
ADMIN_IMAGE=shopping_admin_final_0719
GITLAB_IMAGE=gitlab-populated-final-port8023
FORUM_IMAGE=postmill-populated-exposed-withimg

MAGENTO_DB_USER=magentouser
MAGENTO_DB_PASS=MyPassword
MAGENTO_DB_NAME=magentodb

BASE_HOST="127.0.0.1"  # 也可以改成 localhost
# ------------------------

echo "[1/6] Stop & remove containers (ignore if not exist)"
docker rm -f "$SHOPPING_NAME" "$ADMIN_NAME" "$GITLAB_NAME" "$FORUM_NAME" 2>/dev/null || true

echo "[2/6] Run containers"
docker run --name "$SHOPPING_NAME" -p "${SHOPPING_PORT}:80" -d "$SHOPPING_IMAGE"
docker run --name "$ADMIN_NAME"   -p "${ADMIN_PORT}:80"   -d "$ADMIN_IMAGE"
docker run --name "$GITLAB_NAME"  -d -p "${GITLAB_PORT}:${GITLAB_PORT}" "$GITLAB_IMAGE" /opt/gitlab/embedded/bin/runsvdir-start
docker run --name "$FORUM_NAME"   -p "${FORUM_PORT}:80"   -d "$FORUM_IMAGE"

wait_mysql() {
  local ctn="$1"
  echo "    - waiting MySQL in container: $ctn"
  for i in {1..60}; do
    if docker exec "$ctn" bash -lc "mysqladmin ping -u${MAGENTO_DB_USER} -p${MAGENTO_DB_PASS} --silent" >/dev/null 2>&1; then
      echo "      MySQL is ready."
      return 0
    fi
    sleep 1
  done
  echo "      ERROR: MySQL not ready in $ctn after 60s" >&2
  return 1
}

set_magento_base_urls() {
  local ctn="$1"
  local port="$2"
  local base="http://${BASE_HOST}:${port}"

  echo "[3/6] Configure Magento base urls for $ctn -> $base/"

  wait_mysql "$ctn"

  # 1) set base url (unsecure) via magento CLI (no trailing slash per your comment)
  docker exec "$ctn" bash -lc "/var/www/magento2/bin/magento setup:store-config:set --base-url=\"${base}\""

  # 2) force-update BOTH unsecure & secure base_url in DB (with trailing slash)
  docker exec "$ctn" bash -lc "mysql -u${MAGENTO_DB_USER} -p${MAGENTO_DB_PASS} ${MAGENTO_DB_NAME} -e \"
    UPDATE core_config_data SET value='${base}/' WHERE path IN ('web/unsecure/base_url','web/secure/base_url');
    SELECT path,value FROM core_config_data WHERE path IN ('web/unsecure/base_url','web/secure/base_url');
  \""

  # 3) flush cache
  docker exec "$ctn" bash -lc "/var/www/magento2/bin/magento cache:flush"
}

set_magento_base_urls "$SHOPPING_NAME" "$SHOPPING_PORT"
set_magento_base_urls "$ADMIN_NAME" "$ADMIN_PORT"

echo "[4/6] Disable forced password reset for admin"
docker exec "$ADMIN_NAME" bash -lc "php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0"
docker exec "$ADMIN_NAME" bash -lc "php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0"
docker exec "$ADMIN_NAME" bash -lc "php /var/www/magento2/bin/magento cache:flush"

echo "[5/6] Configure GitLab external_url"
docker exec "$GITLAB_NAME" bash -lc "sed -i \"s|^external_url.*|external_url 'http://${BASE_HOST}:${GITLAB_PORT}'|\" /etc/gitlab/gitlab.rb"
docker exec "$GITLAB_NAME" bash -lc "gitlab-ctl reconfigure"

echo "[6/6] Done."
echo "Shopping:       http://${BASE_HOST}:${SHOPPING_PORT}/"
echo "Shopping Admin: http://${BASE_HOST}:${ADMIN_PORT}/"
echo "GitLab:         http://${BASE_HOST}:${GITLAB_PORT}/"
echo "Forum:          http://${BASE_HOST}:${FORUM_PORT}/"