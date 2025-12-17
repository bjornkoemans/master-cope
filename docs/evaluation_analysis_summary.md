## Evaluation Metrics Analysis Summary

### Overview
This analysis examines the log metrics from the evaluation run `evaluation_20250707_124234` under the MAPPO experiment `mappo_20250705_104325`. The evaluation includes 47 log files total, with the first 10 skipped and the remaining 37 grouped into 4 baseline experiments:

1. **Random** (10 logs): Episodes 11-20
2. **Best Median** (10 logs): Episodes 21-30  
3. **Ground Truth** (10 logs): Episodes 31-40
4. **MAPPO** (7 logs): Episodes 41-47 (fewer logs available)

### Key Findings

#### Throughput Time (minutes)
- **Random**: Mean = 44.25, Median = 29.01 (940 data points)
- **Best Median**: Mean = 49.92, Median = 28.51 (940 data points)
- **Ground Truth**: Mean = 54.91, Median = 28.11 (940 data points)
- **MAPPO**: Mean = 40.72, Median = 24.55 (658 data points)

**MAPPO shows the best performance** with the lowest mean and median throughput times.

#### Waiting Time (minutes)
- **Random**: Mean = 0.19, Median = 0.00 (940 data points)
- **Best Median**: Mean = 0.33, Median = 0.00 (940 data points)
- **Ground Truth**: Mean = 2.26, Median = 0.00 (940 data points)
- **MAPPO**: Mean = 1.72, Median = 0.02 (658 data points)

**Random baseline shows the lowest waiting times**, while Ground Truth has the highest waiting times.

#### Processing Time (minutes)
- **Random**: Mean = 44.06, Median = 28.99 (940 data points)
- **Best Median**: Mean = 49.59, Median = 28.46 (940 data points)
- **Ground Truth**: Mean = 52.64, Median = 27.65 (940 data points)
- **MAPPO**: Mean = 39.00, Median = 23.47 (658 data points)

**MAPPO achieves the best processing performance** with the lowest mean and median processing times.

### Performance Ranking

1. **MAPPO**: Best overall performance with lowest throughput and processing times
2. **Random**: Good performance, especially low waiting times
3. **Best Median**: Moderate performance across all metrics
4. **Ground Truth**: Highest times across most metrics, particularly waiting time

### Generated Plots

Two comprehensive visualization files have been created:

1. **`evaluation_metrics_comparison.png`**: Violin plots showing the distribution of each metric across all baselines, with median (red squares) and mean (blue circles) markers
2. **`evaluation_metrics_detailed.png`**: Detailed analysis with box plots (top row) and mean vs median bar charts with error bars (bottom row)

### Technical Notes

- The analysis processed CSV log files containing case-level task execution data
- Metrics were calculated by grouping tasks by case_id and summing times for each case
- All timestamps were properly parsed and converted to UTC for consistent calculations
- The MAPPO baseline had fewer episodes (7 vs 10) due to available log files
