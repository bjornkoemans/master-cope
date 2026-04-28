#!/usr/bin/env python3
"""
CLI tool to augment event logs with collaboration requirements.

Usage:
    python scripts/augment_collaboration.py --input data/cvs_pharmacy/processed/cvs_pharmacy.csv --output data/cvs_pharmacy/processed/train_collaborative.csv
"""
import argparse
import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from preprocessing.collaboration_augmenter import augment_event_log


def main():
    parser = argparse.ArgumentParser(
        description='Augment event log with collaboration requirements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/augment_collaboration.py \\
      --input data/cvs_pharmacy/processed/cvs_pharmacy.csv \\
      --output data/cvs_pharmacy/processed/train_collaborative.csv

  # With custom rules
  python scripts/augment_collaboration.py \\
      --input data/cvs_pharmacy/processed/cvs_pharmacy.csv \\
      --output data/cvs_pharmacy/processed/train_collaborative.csv \\
      --rules configs/my_custom_rules.yaml

  # With specific seed
  python scripts/augment_collaboration.py \\
      --input data/cvs_pharmacy/processed/cvs_pharmacy.csv \\
      --output data/cvs_pharmacy/processed/train_collaborative.csv \\
      --seed 123
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--rules', '-r',
        default='src/configs/collaboration_rules.yaml',
        help='Collaboration rules YAML file (default: src/configs/collaboration_rules.yaml)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Validate rules file exists
    if not Path(args.rules).exists():
        print(f"Error: Rules file not found: {args.rules}")
        print(f"   Create one using the template at src/configs/collaboration_rules.yaml")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run augmentation
    try:
        augment_event_log(
            input_path=args.input,
            output_path=args.output,
            rules_config_path=args.rules,
            seed=args.seed
        )
        print("\nAugmentation completed successfully.")
    except Exception as e:
        print(f"\nError during augmentation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
