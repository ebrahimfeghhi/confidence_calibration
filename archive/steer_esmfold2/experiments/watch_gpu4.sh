#!/bin/bash
# fires (exits) when a NON-ebrahim process appears on GPU4, OR the swarm dies, OR 40min heartbeat
cd /home/ebrahim/tts_abag
BUS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' '$1==4{print $2}')
for i in $(seq 1 53); do
  # swarm alive?
  if ! pgrep -f "tag 8jyr_gencloud" >/dev/null; then echo "ALERT: swarm process DIED"; exit 0; fi
  # any foreign owner on GPU4?
  foreign=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | while IFS=',' read b p; do
    [ "${b// /}" = "$BUS" ] || continue
    o=$(ps -o user= -p ${p// /} 2>/dev/null); [ "$o" != "ebrahim" ] && [ -n "$o" ] && echo "$o:${p// /}"
  done)
  if [ -n "$foreign" ]; then echo "ALERT: FOREIGN process on GPU4 -> $foreign"; exit 0; fi
  sleep 45
done
echo "heartbeat: 40min clean, swarm alive, GPU4 uncontended"
