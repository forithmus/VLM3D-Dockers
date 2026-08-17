"""
MR-RATE MRI Preprocessing Pipeline - PACS Metadata Filtering Block

This script processes a raw PACS metadata CSV file by:
1. Loading a CSV file containing DICOM metadata
2. Enforcing that all required_dicom_columns from config are present (error if missing)
3. Keeping as many optional_dicom_columns as possible (skipping those not present)
4. Removing rows with missing critical identifiers
5. Removing duplicate entries based on unique series identifiers

The script ensures clean, deduplicated metadata for downstream processing steps.
The metadata columns config is loaded automatically from:
    configs/config_metadata_columns.json  (relative to mr_rate_preprocessing package root)

Example Usage:
    # Process a metadata CSV and create filtered output
    python src/mr_rate_preprocessing/mri_preprocessing/pacs_metadata_filtering.py \\
        --input-csv data/raw/batch00/batch00_pacs_metadata.csv \\
        --output-csv data/interim/batch00/batch00_raw_metadata.csv \\
        --log-dir logs/batch00
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Set

import pandas as pd

from utils import setup_logging
from mr_rate_preprocessing.configs.config_mri_preprocessing import (
    METADATA_COLUMNS_CONFIG_PATH, 
    NOT_ALLOWED_TO_HAVE_NAN_COLUMNS,
    NOT_ALLOWED_TO_HAVE_DUPLICATES_COLUMNS
)


def load_metadata_columns_config(config_path: Path, logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """
    Load required and optional DICOM column lists from config file.

    Args:
        config_path: Path to metadata columns JSON config
        logger: Logger instance

    Returns:
        Tuple of (required_dicom_columns, optional_dicom_columns)

    Raises:
        ValueError: If required_dicom_columns is missing or empty in the config
    """
    logger.info(f"Loading metadata columns config from: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    required_columns = config.get('required_dicom_columns', [])
    optional_columns = config.get('optional_dicom_columns', [])

    if not required_columns:
        raise ValueError("No 'required_dicom_columns' found in config file")

    logger.info(f"  Required DICOM columns: {len(required_columns)}")
    logger.info(f"  Optional DICOM columns: {len(optional_columns)}")

    return required_columns, optional_columns


def load_csv(
    csv_path: Path,
    required_columns: List[str],
    optional_columns: List[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load a single PACS metadata CSV file.

    All required_columns must be present — a ValueError is raised otherwise.
    Optional columns are kept when available and silently skipped when absent.

    Args:
        csv_path: Path to the CSV file
        required_columns: Columns that must exist in the CSV
        optional_columns: Columns to keep when present
        logger: Logger instance

    Returns:
        DataFrame containing required columns and any optional columns that
        were present in the source file

    Raises:
        ValueError: If any required column is missing from the CSV
    """
    logger.info(f"Loading CSV: {csv_path.name}")

    # Peek at the header without loading the full file
    available_columns: Set[str] = set(pd.read_csv(csv_path, nrows=0).columns)

    # Validate required columns
    missing_required = set(required_columns) - available_columns
    if missing_required:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: "
            f"{sorted(missing_required)}"
        )

    # Determine which optional columns are present
    present_optional = [c for c in optional_columns if c in available_columns]
    missing_optional = [c for c in optional_columns if c not in available_columns]

    if missing_optional:
        logger.info(
            f"  Skipping {len(missing_optional)} optional column(s) not present: "
            f"{missing_optional}"
        )

    columns_to_load = required_columns + present_optional

    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False, usecols=columns_to_load)
    logger.info(f"  Shape: {df.shape}  ({len(present_optional)} optional columns kept)")

    return df[columns_to_load].copy()


def clean_metadata(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Clean metadata by removing rows with missing key identifiers and duplicates.

    Key cleaning steps:
    1. Check for NaN values in critical columns
    2. Drop rows with missing AccessionNumber, SeriesDescription, SeriesNumber,
       SeriesInstanceUID, StudyInstanceUID, or Patient'sAge
    3. Remove duplicate rows based on (AccessionNumber, SeriesDescription, SeriesNumber)

    Args:
        df: Input metadata DataFrame
        logger: Logger instance

    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    logger.info(f"Cleaning metadata (initial rows: {initial_rows})...")
    
    # Check NaN counts in key columns
    logger.info("")
    logger.info("  NaN counts in key columns:")
    nan_counts = df[NOT_ALLOWED_TO_HAVE_NAN_COLUMNS].isna().sum()
    for col, count in nan_counts.items():
        logger.info(f"    {col}: {count}")
    
    # Drop rows with NaN in key columns
    logger.info("")
    logger.info("  Dropping rows with NaN in key columns...")
    df_clean = df.dropna(subset=NOT_ALLOWED_TO_HAVE_NAN_COLUMNS).copy()
    dropped_nan = initial_rows - len(df_clean)
    logger.info(f"    Dropped {dropped_nan} rows ({dropped_nan/initial_rows*100:.2f}%)")
    logger.info(f"    Remaining rows: {len(df_clean)}")

    # Remove duplicates based on unique series identifiers
    logger.info("")
    logger.info("  Removing duplicate rows...")
    logger.info(f"    Deduplication criteria: {NOT_ALLOWED_TO_HAVE_DUPLICATES_COLUMNS}")
    
    rows_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=NOT_ALLOWED_TO_HAVE_DUPLICATES_COLUMNS, keep='first')
    duplicates_removed = rows_before_dedup - len(df_clean)
    
    if rows_before_dedup > 0:
        logger.info(f"    Removed {duplicates_removed} duplicates ({duplicates_removed/rows_before_dedup*100:.2f}%)")
    logger.info(f"    Final rows: {len(df_clean)}")
    
    # Summary
    total_removed = initial_rows - len(df_clean)
    logger.info("")
    logger.info("  Cleaning summary:")
    logger.info(f"    Initial rows: {initial_rows}")
    logger.info(f"    Final rows: {len(df_clean)}")
    logger.info(f"    Total removed: {total_removed} ({total_removed/initial_rows*100:.2f}%)")
    
    return df_clean


def save_metadata(df: pd.DataFrame, output_path: Path, logger: logging.Logger):
    """
    Save cleaned metadata to CSV file.
    
    Args:
        df: Cleaned metadata DataFrame
        output_path: Path to output CSV file
        logger: Logger instance
    """
    logger.info("")
    logger.info(f"Saving cleaned metadata to: {output_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"  Successfully saved {len(df)} rows")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter and clean a PACS metadata CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Process a PACS metadata CSV
    python src/mr_rate_preprocessing/mri_preprocessing/pacs_metadata_filtering.py \\
        --input-csv data/raw/batch00/batch00_pacs_metadata.csv \\
        --output-csv data/interim/batch00/batch00_raw_metadata.csv \\
        --log-dir logs/batch00
        """
    )

    parser.add_argument(
        "--input-csv", "-i",
        type=Path,
        required=True,
        help="CSV file containing raw PACS metadata"
    )
    
    parser.add_argument(
        "--output-csv", "-o",
        type=Path,
        required=True,
        help="Output path for cleaned metadata CSV"
    )
    
    parser.add_argument(
        "--log-dir", "-l",
        type=Path,
        default=Path("logs"),
        help="Directory for log files (default: logs)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print logs to terminal in addition to log file (default: True)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    script_name = Path(__file__).stem
    logger = setup_logging(args.log_dir, script_name=script_name, verbose=args.verbose)
    
    logger.info("=" * 60)
    logger.info("MR-RATE Image Processing Pipeline - PACS Metadata Filtering Block")
    logger.info("=" * 60)
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Metadata columns config: {METADATA_COLUMNS_CONFIG_PATH}")
    logger.info(f"Output CSV: {args.output_csv}")
    logger.info(f"Log directory: {args.log_dir}")
    logger.info("")

    # Validate inputs
    if not args.log_dir.exists():
        logger.error(f"Log directory not found: {args.log_dir}")
        return 1

    if args.output_csv.exists():
        logger.error(f"Output CSV already exists: {args.output_csv}")
        return 1

    if not args.input_csv.exists():
        logger.error(f"Input CSV not found: {args.input_csv}")
        return 1

    if not METADATA_COLUMNS_CONFIG_PATH.exists():
        logger.error(f"Metadata columns config not found: {METADATA_COLUMNS_CONFIG_PATH}")
        return 1

    try:
        # Load metadata columns config
        required_columns, optional_columns = load_metadata_columns_config(METADATA_COLUMNS_CONFIG_PATH, logger)

        logger.info("")

        # Load CSV
        metadata_df = load_csv(args.input_csv, required_columns, optional_columns, logger)
        
        logger.info("")
        
        # Clean metadata
        cleaned_df = clean_metadata(metadata_df, logger)
        
        # Save cleaned metadata
        save_metadata(cleaned_df, args.output_csv, logger)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("PACS Metadata Filtering Complete")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
