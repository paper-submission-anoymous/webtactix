#!/usr/bin/env bash
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost,172.17.0.1,172.17.0.2,172.17.0.3,::1"
export no_proxy="$NO_PROXY"

# 1) 启动 openstreetmap-website（Rails + Postgres）
cd "/mnt/d/coding/GUI Agent/new_version/webarena/openstreetmap-website"
docker compose up -d

# 2) 启动 tile server（如果你之前就是 docker run 起的）
# 已存在就不重复创建，直接 start
if docker ps -a --format '{{.Names}}' | grep -qx 'osm-tile'; then
  docker start osm-tile >/dev/null
else
  docker run --name osm-tile \
    --volume=osm-data:/data/database/ \
    --volume=osm-tiles:/data/tiles/ \
    -p 8080:80 --detach=true \
    overv/openstreetmap-tile-server run
fi

# 3) 启动 nominatim（你之前是 docker run 起的）
if docker ps -a --format '{{.Names}}' | grep -qx 'nominatim'; then
  docker start nominatim >/dev/null
else
  # 注意：把 /path/to/osm_dump 改成你真实路径（你之前是 /mnt/e/webarena/map_backend/osm_dump）
  docker run --name nominatim \
    --env=IMPORT_STYLE=extratags \
    --env=PBF_PATH=/nominatim/data/us-northeast-latest.osm.pbf \
    --env=IMPORT_WIKIPEDIA=/nominatim/data/wikimedia-importance.sql.gz \
    --volume=/mnt/e/webarena/map_backend/osm_dump:/nominatim/data \
    --volume=nominatim-data:/var/lib/postgresql/14/main \
    --volume=nominatim-flatnode:/nominatim/flatnode \
    -p 8085:8080 \
    -d mediagis/nominatim:4.2 /app/start.sh
fi

# 4) 启动 OSRM（car/bike/foot）
for svc in car bike foot; do
  name="osrm-$svc"
  port="5000"
  [ "$svc" = "bike" ] && port="5001"
  [ "$svc" = "foot" ] && port="5002"

  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    docker start "$name" >/dev/null
  else
    # 把 /your/routing/path 替换成你的真实 routing 数据目录
    ROUTING_BASE="/mnt/e/webarena/map_backend/osrm_routing"
    echo "ELSE"
    docker run --name "$name" \
      -v "$ROUTING_BASE/$svc:/data:ro" \
      -p "$port:5000" -d \
      ghcr.io/project-osrm/osrm-backend \
      osrm-routed --algorithm mld /data/us-northeast-latest.osrm
  fi
done

echo "OK: OSM website http://127.0.0.1:3000"
echo "OK: tiles      http://127.0.0.1:8080"
echo "OK: nominatim  http://127.0.0.1:8085/status"
echo "OK: osrm       http://127.0.0.1:5000 (car), 5001 (bike), 5002 (foot)"
