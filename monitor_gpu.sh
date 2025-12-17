#!/bin/bash
# GPU Monitoring script for training
# Usage: ./monitor_gpu.sh

echo "=== GPU Training Monitor ==="
echo "Watching for GPU activity during MAPPO training..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=== GPU Status @ $(date '+%H:%M:%S') ==="
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    echo ""
    echo "Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | awk -F', ' '{printf "  %s MB / %s MB (%.1f%%)\n", $1, $2, ($1/$2)*100}'
    echo ""
    echo "GPU Utilization:"
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{printf "  %s%%\n", $1}'
    echo ""
    echo "Process Info:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    echo ""
    echo "Waiting for policy update phase..."
    echo "You should see 2000-2800 MB when training starts"
    sleep 1
done
