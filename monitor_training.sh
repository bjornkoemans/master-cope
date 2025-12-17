#!/bin/bash
# Comprehensive training monitor (CPU + GPU)
# Shows which phase you're in and what to expect

echo "=== MAPPO Training Monitor ==="
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║          MAPPO Training Monitor @ $(date '+%H:%M:%S')                  ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # GPU Status
    echo "┌─ GPU STATUS ────────────────────────────────────────────────────┐"
    if command -v nvidia-smi &> /dev/null; then
        gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)

        echo "  Memory:       ${gpu_mem} MB / 3000 MB"
        echo "  Utilization:  ${gpu_util}%"
        echo "  Temperature:  ${gpu_temp}°C"
        echo ""

        # Determine phase based on GPU usage
        if [ "$gpu_mem" -lt 500 ]; then
            echo "  Phase: 📊 EPISODE COLLECTION (CPU-bound)"
            echo "  Expected: 200-400 MB, 0-10% GPU util"
        elif [ "$gpu_mem" -gt 1500 ]; then
            echo "  Phase: 🚀 POLICY UPDATE (GPU-bound)"
            echo "  Expected: 2000-2800 MB, 70-95% GPU util"
        else
            echo "  Phase: 🔄 TRANSITION"
        fi
    else
        echo "  nvidia-smi not found"
    fi
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo ""

    # CPU Status
    echo "┌─ CPU STATUS ────────────────────────────────────────────────────┐"
    echo "  Per-Core Usage:"
    mpstat -P ALL 1 1 | awk '/Average:/ && $2 ~ /[0-9]+/ {printf "    Core %s: %5.1f%%\n", $2, 100-$NF}'
    echo ""

    # Get total CPU usage
    total_cpu=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    echo "  Total CPU: ${total_cpu}%"
    echo ""

    # Find Python process
    python_pid=$(pgrep -f "python.*train.py" | head -1)
    if [ -n "$python_pid" ]; then
        cpu_percent=$(ps -p $python_pid -o %cpu= | awk '{print $1}')
        threads=$(ps -p $python_pid -o nlwp= | awk '{print $1}')
        echo "  Python Process:"
        echo "    PID:     $python_pid"
        echo "    CPU:     ${cpu_percent}%"
        echo "    Threads: $threads"
        echo ""

        if (( $(echo "$cpu_percent < 150" | bc -l) )); then
            echo "  Phase Indicator: 📊 Single-threaded (Episode Collection)"
            echo "  Expect 100% on 1 core, cores may switch"
        else
            echo "  Phase Indicator: 🚀 Multi-threaded (Policy Update)"
            echo "  PyTorch using multiple cores"
        fi
    fi
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo ""

    # Process Info
    echo "┌─ PROCESS INFO ──────────────────────────────────────────────────┐"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | \
            awk -F', ' '{printf "  PID %s: %s (%s MB)\n", $1, $2, $3}' || echo "  No GPU processes"
    fi
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo ""

    echo "TIP: Watch for GPU memory jump to 2000+ MB during policy updates"

    sleep 2
done
