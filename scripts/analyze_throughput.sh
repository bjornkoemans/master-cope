#!/bin/bash
# Analyze experiment results and extract metrics
# Usage: ./scripts/analyze_throughput.sh <experiment_dir>

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_dir>"
    echo ""
    echo "Example:"
    echo "  $0 experiments/mappo_20251221_143028"
    exit 1
fi

EXPERIMENT_DIR="$1"

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory '$EXPERIMENT_DIR' does not exist."
    exit 1
fi

echo "Analyzing experiment: $EXPERIMENT_DIR"
echo "============================================================"

# 1. Check training summary
SUMMARY_FILE="$EXPERIMENT_DIR/training_summary.txt"
if [ -f "$SUMMARY_FILE" ]; then
    echo ""
    echo "📊 TRAINING SUMMARY:"
    echo "------------------------------------------------------------"
    cat "$SUMMARY_FILE"
fi

# 2. List log files
LOG_DIR="$EXPERIMENT_DIR/logs"
if [ -d "$LOG_DIR" ]; then
    echo ""
    echo "📈 LOG FILES:"
    echo "------------------------------------------------------------"
    ls -lh "$LOG_DIR"

    # Show first few lines of each log
    for log in "$LOG_DIR"/log_*.csv; do
        if [ -f "$log" ]; then
            echo ""
            echo "Log file: $(basename "$log")"
            echo "First 10 lines:"
            head -10 "$log"
            echo ""
            echo "Total lines: $(wc -l < "$log")"
        fi
    done
fi

# 3. Check final evaluation
EVAL_DIR="$EXPERIMENT_DIR/final_evaluation"
if [ -d "$EVAL_DIR" ]; then
    echo ""
    echo "📋 FINAL EVALUATION:"
    echo "------------------------------------------------------------"

    if [ -d "$EVAL_DIR/logs" ]; then
        ls -lh "$EVAL_DIR/logs"

        for log in "$EVAL_DIR/logs"/log_*.csv; do
            if [ -f "$log" ]; then
                echo ""
                echo "Evaluation log: $(basename "$log")"
                echo "First 10 lines:"
                head -10 "$log"
            fi
        done
    fi
fi

echo ""
echo "============================================================"
echo "Analysis complete."
