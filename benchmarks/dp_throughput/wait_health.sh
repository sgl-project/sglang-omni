#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# wait_health.sh <port> [max_tries]   (each try = 6s)
PORT=$1; N=${2:-45}
for i in $(seq 1 "$N"); do
  c=$(curl -s -o /dev/null -w "%{http_code}" -m 3 "127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$c" = "200" ] && { echo "port $PORT HEALTHY after $((i*6))s"; exit 0; }
  sleep 6
done
echo "port $PORT NOT healthy (last=$c)"; exit 1
