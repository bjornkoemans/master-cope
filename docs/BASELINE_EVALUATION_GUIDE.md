# Consolidated Baseline Evaluation Guide

This guide explains how to use the new consolidated baseline evaluation system that includes dataset names in filenames and supports multiple datasets and algorithms.

## New Files Created

### 1. `run_multi_dataset_baseline_evaluation.py`
Main consolidated script that replaces all individual baseline files. Features:
- ✅ Dataset names included in output filenames
- ✅ Support for multiple datasets in one run
- ✅ All baseline agents in one file (Random, Best Median, Ground Truth)
- ✅ Support for trained agents (MAPPO, QMIX)
- ✅ Configurable evaluation parameters

### 2. `run_comprehensive_baseline_evaluation.py`
Convenience script that runs the full evaluation with optimal settings.

### 3. `test_baseline_script.py`
Quick test script to verify the system works.

## Dataset Configuration

The script supports these datasets (auto-detected from `data/input/`):
- `BPI12W` - BPI12W.csv
- `BPI12W_1` - BPI12W_1.csv  
- `BPI12W_2` - BPI12W_2.csv
- `LoanApp` - LoanApp.csv
- `CVS_Pharmacy` - cvs_pharmacy.csv
- `Train_Preprocessed` - train_preprocessed.csv

## Usage Examples

### Run All Datasets with Baselines Only
```bash
python run_multi_dataset_baseline_evaluation.py --datasets all --episodes 20
```

### Run Specific Datasets
```bash
python run_multi_dataset_baseline_evaluation.py --datasets BPI12W LoanApp --episodes 10
```

### Include Trained Models
```bash
python run_multi_dataset_baseline_evaluation.py \
  --datasets all \
  --episodes 20 \
  --include-trained \
  --mappo-model-path ./models/mappo_final \
  --qmix-model-path ./models/qmix_final
```

### Quick Test
```bash
python test_baseline_script.py
```

### Full Comprehensive Evaluation
```bash
python run_comprehensive_baseline_evaluation.py
```

## Output Structure

Results are saved with dataset names in the filenames:

```
experiments/
├── baseline_evaluation_BPI12W_20250715_143022/
│   ├── baseline_comparison_results_BPI12W.json
│   └── ... (environment logs)
├── baseline_evaluation_LoanApp_20250715_143055/
│   ├── baseline_comparison_results_LoanApp.json
│   └── ... (environment logs)
└── multi_dataset_summary_20250715_143122.json
```

## Baseline Agents Included

1. **Random Agent**: Selects actions randomly
2. **Best Median Agent**: Only the best performing agent raises hand
3. **Ground Truth Agent**: Follows actual assignments from data

## Trained Agents Supported

1. **MAPPO Agent**: Multi-Agent Proximal Policy Optimization
2. **QMIX Agent**: Q-Mixing Networks

## Command Line Options

```
--datasets LIST         Datasets to evaluate (default: all available)
--episodes N           Number of evaluation episodes per dataset (default: 20)
--seed N              Random seed for reproducibility (default: 42)
--use-test-data       Use test data split instead of training data
--include-trained     Include trained agents (MAPPO, QMIX) in comparison
--mappo-model-path    Path to trained MAPPO model directory
--qmix-model-path     Path to trained QMIX model directory
--output-dir          Base directory for experiment outputs (default: ./experiments)
```

## Migration from Old System

### Old Files (now obsolete):
- `run_baseline_evaluation.py` - replaced by consolidated script
- `src/baselines.py` - baseline classes now included in main script  
- Individual demo/plot scripts - functionality consolidated

### What Changed:
1. ✅ All baseline functionality consolidated into one script
2. ✅ Dataset names automatically included in output filenames
3. ✅ Support for multiple datasets in single run
4. ✅ QMIX agent integration added
5. ✅ Better error handling and progress reporting
6. ✅ Configurable output directories

### Running New Evaluations:
Instead of:
```bash
python run_baseline_evaluation.py --episodes 20 --include-mappo
```

Use:
```bash
python run_multi_dataset_baseline_evaluation.py --datasets all --episodes 20 --include-trained
```

## Benefits of New System

1. **Dataset Names in Files**: Easy identification of which dataset results belong to
2. **Multi-Dataset Support**: Run all datasets at once for comprehensive comparison
3. **QMIX Integration**: Compare against latest QMIX implementations
4. **Consolidated Code**: All baseline logic in one maintainable file
5. **Better Organization**: Clear experiment directory structure
6. **Flexible Configuration**: Easy to add new datasets or algorithms

## Next Steps

1. Run comprehensive evaluation: `python run_comprehensive_baseline_evaluation.py`
2. Compare results across datasets using the generated JSON files
3. Use the multi-dataset summary for high-level analysis
4. Archive old experiment directories if desired
