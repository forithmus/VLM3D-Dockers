"""
MR-RATE MRI Preprocessing Pipeline - Metadata Preparation Block

This script validates processed data completeness, filters metadata to valid rows,
adds patient IDs from a mapping file, converts study IDs to anonymized UIDs,
and produces a clean metadata CSV. Optionally uploads the CSV to Hugging Face.

Multiple input triplets (--processed-dir, --modalities-json, --input-csv) can be
provided as parallel lists; studies from all of them are treated as the same batch
and land in the same output CSV. The N-th modalities JSON and input CSV are paired
with the N-th processed-dir.

Input:
    - One or more processed data directories (output of brain_segmentation_and_defacing.py)
    - One or more modalities JSONs (output of modality_filtering.py)
    - One or more filtered metadata CSVs (output of modality_filtering.py)
    - Patient mapping CSV (original_accession -> anon_patient_id)
    - Study date mapping Excel (Accession -> anon_study_date)

Output:
    - Clean metadata CSV with only valid (processed) rows, patient_id added,
      study_id replaced with an anonymized UID, anon_study_date added,
      sensitive columns dropped

The metadata columns config is loaded automatically from:
    configs/config_metadata_columns.json  (relative to mr_rate_preprocessing package root)

Completeness check logic:
    - A modality is complete if all 3 files exist: image, brain mask, defacing mask
    - If ANY modality (center or moving) is incomplete: drop the ENTIRE study

Example Usage:
    python src/mr_rate_preprocessing/mri_preprocessing/prepare_metadata.py \\
        --processed-dir data/processed/batch00 \\
        --modalities-json data/interim/batch00/batch00_modalities_to_process.json \\
        --input-csv data/interim/batch00/batch00_modalities_to_process_metadata.csv \\
        --patient-mapping-csv data/raw/batch00_accession_to_anon_patient.xlsx \\
        --study-date-mapping data/raw/batch00_accession_to_anon_study_date.xlsx \\
        --output-csv data/processed/batch00_metadata.csv \\
        --batch-id batch00 \\
        --log-dir logs/batch00 \\
        --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from utils import setup_logging, accession_to_uid
from mr_rate_preprocessing.configs.config_mri_preprocessing import METADATA_COLUMNS_CONFIG_PATH


def check_modality_complete(
    processed_dir: Path,
    study_id: str,
    modality_id: str,
) -> bool:
    """
    Check if a modality has all required processed files.
    
    Required files:
        - img/{study_id}_{modality_id}.nii.gz
        - seg/{study_id}_{modality_id}_brain-mask.nii.gz
        - seg/{study_id}_{modality_id}_defacing-mask.nii.gz

    Args:
        processed_dir: Base directory of processed data
        study_id: Study identifier
        modality_id: Modality identifier (e.g., "t1w-raw-sag")

    Returns:
        True if all 3 files exist, False otherwise
    """
    study_dir = processed_dir / study_id

    img_path = study_dir / "img" / f"{study_id}_{modality_id}.nii.gz"
    brain_mask_path = study_dir / "seg" / f"{study_id}_{modality_id}_brain-mask.nii.gz"
    defacing_mask_path = study_dir / "seg" / f"{study_id}_{modality_id}_defacing-mask.nii.gz"
    
    return img_path.exists() and brain_mask_path.exists() and defacing_mask_path.exists()


def collect_valid_series_uids(
    processed_dir: Path,
    modalities_data: Dict,
    logger,
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Iterate through modalities JSON, check completeness, collect valid SeriesInstanceUIDs.
    
    Logic:
        - If ANY modality (center or moving) is incomplete: drop the ENTIRE study
        - Collect SeriesInstanceUID for all modalities in fully complete studies only
    
    Args:
        processed_dir: Base directory of processed data
        modalities_data: Dictionary from modalities JSON
        logger: Logger instance
        
    Returns:
        Tuple of (set of valid SeriesInstanceUIDs, stats dict)
    """
    valid_uids = set()
    
    stats = {
        'total_studies': len(modalities_data),
        'valid_studies': 0,
        'studies_dropped_incomplete': 0,
        'total_modalities_checked': 0,
        'valid_modalities': 0,
    }
    
    for study_id, study_data in modalities_data.items():        
        # Build a flat dict of all modalities for this study
        center_modality = study_data.get("center_modality", {})
        center_id = list(center_modality.keys())[0]
        center_info = center_modality[center_id]
        all_modalities = {center_id: center_info}

        moving_modalities = study_data.get("moving_modality", {})
        all_modalities.update(moving_modalities)
        stats['total_modalities_checked'] += len(all_modalities)
        
        # All-or-nothing: drop entire study if any modality is incomplete
        study_complete = True
        for mod_id in all_modalities:
            if not check_modality_complete(processed_dir, study_id, mod_id):
                logger.warning(
                    f"Study {study_id}: Modality {mod_id} incomplete, dropping entire study"
                )
                study_complete = False
                break
        
        if not study_complete:
            stats['studies_dropped_incomplete'] += 1
            continue
        
        # All modalities complete - collect all SeriesInstanceUIDs
        stats['valid_studies'] += 1
        stats['valid_modalities'] += len(all_modalities)
        for mod_info in all_modalities.values():
            valid_uids.add(mod_info['series_instance_uid'])
    
    return valid_uids, stats


def load_patient_mapping(
    patient_mapping_csv: Path,
    logger,
) -> Dict[str, str]:
    """
    Load patient mapping CSV and create study_id -> patient_id mapping.
    
    The CSV has columns: original_accession, full_accession, anon_patient_id, study_instance_uid
    
    Processing:
        - Use full_accession as study_id as-is
        - Format patient_id by removing "patient_" prefix
    
    Args:
        patient_mapping_csv: Path to patient mapping CSV
        logger: Logger instance
        
    Returns:
        Dictionary mapping study_id -> formatted patient_id
    """
    logger.info(f"Loading patient mapping from: {patient_mapping_csv}")
    
    df = pd.read_excel(patient_mapping_csv)
    logger.info(f"  Loaded {len(df)} rows")
    
    # Build mapping: study_id -> patient_id
    mapping = {}
    errors = 0
    duplicates = 0
    
    for _, row in df.iterrows():
        try:
            study_id = str(row['Accession'])
            patient_id = str(row['Anon Patient ID']).replace("patient_", "")
            if study_id in mapping:
                if mapping[study_id] == patient_id:
                    duplicates += 1
                    continue
                else:
                    raise ValueError(f"Study ID {study_id} has multiple patient IDs: {mapping[study_id]} and {patient_id}")
            else:
                mapping[study_id] = patient_id
        except (ValueError, KeyError) as e:
            errors += 1
            if errors <= 5:  # Only log first 5 errors
                logger.warning(f"  Failed to process row: {e}")
    
    if errors > 5:
        logger.warning(f"  ... and {errors - 5} more errors")

    if errors > 0:
        raise ValueError(f"Errors encountered while loading patient mapping")
    else:
        logger.info(f"  Created mapping for {len(mapping)} studies (no errors, {duplicates} duplicates)")
        return mapping


def load_study_date_mapping(
    study_date_mapping_path: Path,
    logger,
) -> Dict[str, str]:
    """
    Load study date mapping from Excel and return study_uid -> anon_study_date dict.

    Expects columns: Accession, Anonymized Study Date.
    Derives study_uid from each Accession via accession_to_uid (same transform used
    in Step 4 of this pipeline), so the resulting keys are directly comparable to
    the study_uid values already in the dataframe.

    Args:
        study_date_mapping_path: Path to the Excel file
        logger: Logger instance

    Returns:
        Dictionary mapping study_uid -> anon_study_date string
    """
    logger.info(f"Loading study date mapping from: {study_date_mapping_path}")

    df = pd.read_excel(study_date_mapping_path, usecols=["Accession", "Anonymized Study Date"])
    logger.info(f"  Loaded {len(df)} rows")

    df["study_uid"] = df["Accession"].apply(lambda acc: accession_to_uid(str(acc)))

    mapping: Dict[str, str] = dict(zip(df["study_uid"], df["Anonymized Study Date"].astype(str)))
    logger.info(f"  Built {len(mapping)} study_uid -> anon_study_date mappings")

    return mapping


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare metadata for segmented and defaced dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single input
    python src/mr_rate_preprocessing/mri_preprocessing/prepare_metadata.py \\
        --processed-dir data/processed/batch00 \\
        --modalities-json data/interim/batch00/batch00_modalities_to_process.json \\
        --input-csv data/interim/batch00/batch00_modalities_to_process_metadata.csv \\
        --patient-mapping-csv data/raw/batch00_accession_to_anon_patient.xlsx \\
        --study-date-mapping data/raw/batch00_accession_to_anon_study_date.xlsx \\
        --output-csv data/processed/batch00_metadata.csv \\
        --batch-id batch00 \\
        --verbose
        """
    )
    
    parser.add_argument(
        "--processed-dir", "-p",
        type=str,
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            "One or more paths to processed data directories (output of brain_segmentation_and_defacing.py). "
            "Must be supplied in the same order as --modalities-json and --input-csv."
        ),
    )
    
    parser.add_argument(
        "--modalities-json", "-m",
        type=str,
        nargs="+",
        required=True,
        metavar="JSON",
        help=(
            "One or more paths to modalities JSON files (output of modality_filtering.py). "
            "Must be supplied in the same order as --processed-dir and --input-csv."
        ),
    )
    
    parser.add_argument(
        "--input-csv", "-i",
        type=str,
        nargs="+",
        required=True,
        metavar="CSV",
        help=(
            "One or more paths to filtered metadata CSVs (output of modality_filtering.py). "
            "Must be supplied in the same order as --processed-dir and --modalities-json."
        ),
    )
    
    parser.add_argument(
        "--patient-mapping-csv", "-pm",
        type=Path,
        required=True,
        help="Path to patient mapping CSV (original_accession -> anon_patient_id)"
    )

    parser.add_argument(
        "--study-date-mapping",
        type=Path,
        required=True,
        help=(
            "Path to study date mapping Excel file. "
            "Must contain columns: Accession, Anonymized Study Date. "
            "Used to add an anon_study_date column to the output metadata."
        ),
    )

    parser.add_argument(
        "--output-csv", "-o",
        type=Path,
        required=True,
        help="Output path for the prepared metadata CSV"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face dataset repo ID to upload to",
    )

    parser.add_argument(
        "--batch-id", "-b",
        type=str,
        required=True,
        help="Batch identifier used for HF upload path, e.g. batch00"
    )
    
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        default=False,
        help="Skip uploading the output CSV to Hugging Face (default: upload)"
    )
    
    parser.add_argument(
        "--num-hf-workers",
        type=int,
        default=4,
        help="Number of parallel workers for Hugging Face upload (default: 4)"
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
        default=False,
        help="Print logs to terminal in addition to log file (default: silent)"
    )

    parser.add_argument(
        "--hf-timeout",
        type=int,
        default=120,
        help="Override HF_HUB_DOWNLOAD_TIMEOUT for this run in seconds (default: 120)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the dataset preparation script."""
    args = parse_args()

    # Apply HF environment overrides before any huggingface_hub imports
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_timeout)

    script_name = Path(__file__).stem
    
    # Setup logging
    logger = setup_logging(args.log_dir, script_name, verbose=args.verbose)

    # Convert parallel string lists to Path lists
    processed_dirs: List[Path] = [Path(p) for p in args.processed_dir]
    modalities_jsons: List[Path] = [Path(p) for p in args.modalities_json]
    input_csvs: List[Path] = [Path(p) for p in args.input_csv]
    
    logger.info("=" * 60)
    logger.info("MR-RATE Image Processing Pipeline - Dataset Preparation Block")
    logger.info("=" * 60)
    for idx, (pdir, mj, icsv) in enumerate(zip(processed_dirs, modalities_jsons, input_csvs)):
        logger.info(f"Input set [{idx}]:")
        logger.info(f"  Processed data directory : {pdir}")
        logger.info(f"  Modalities JSON          : {mj}")
        logger.info(f"  Input metadata CSV       : {icsv}")
    logger.info(f"Patient mapping CSV: {args.patient_mapping_csv}")
    logger.info(f"Study date mapping : {args.study_date_mapping}")
    logger.info(f"Output CSV: {args.output_csv}")
    logger.info(f"HF repo ID: {args.repo_id}")
    logger.info(f"Batch ID: {args.batch_id}")
    logger.info(f"HF timeout (s): {args.hf_timeout}")
    logger.info("")
    
    # Load metadata columns config
    if not METADATA_COLUMNS_CONFIG_PATH.exists():
        logger.error(f"Metadata columns config not found: {METADATA_COLUMNS_CONFIG_PATH}")
        return 1
    with open(METADATA_COLUMNS_CONFIG_PATH, 'r') as f:
        col_cfg = json.load(f)
    columns_to_drop = col_cfg["columns_to_drop"]
    column_rename_map = col_cfg["column_rename_map"]
    modality_value_map = col_cfg["modality_value_map"]
    column_order_prefix = col_cfg["column_order_prefix"]
    logger.info(f"Metadata columns config: {METADATA_COLUMNS_CONFIG_PATH}")
    logger.info("")

    # Validate parallel list lengths
    if not (len(processed_dirs) == len(modalities_jsons) == len(input_csvs)):
        logger.error(
            f"--processed-dir, --modalities-json, and --input-csv must have the same number of values "
            f"({len(processed_dirs)} processed dir(s), {len(modalities_jsons)} JSON(s), {len(input_csvs)} CSV(s))."
        )
        return 1

    # Validate all individual inputs
    for idx, (pdir, mj, icsv) in enumerate(zip(processed_dirs, modalities_jsons, input_csvs)):
        if not pdir.exists():
            logger.error(f"Processed data directory [{idx}] not found: {pdir}")
            return 1
        if not mj.exists():
            logger.error(f"Modalities JSON [{idx}] not found: {mj}")
            return 1
        if not icsv.exists():
            logger.error(f"Input metadata CSV [{idx}] not found: {icsv}")
            return 1

    if not args.patient_mapping_csv.exists():
        logger.error(f"Patient mapping CSV not found: {args.patient_mapping_csv}")
        return 1

    if not args.study_date_mapping.exists():
        logger.error(f"Study date mapping file not found: {args.study_date_mapping}")
        return 1

    if args.output_csv.exists():
        logger.error(f"Output CSV already exists: {args.output_csv}")
        return 1
    
    # Step 1: For each input set, load modalities JSON, check completeness, and
    #         collect valid SeriesInstanceUIDs; accumulate across all sets.
    valid_uids: Set[str] = set()
    combined_stats = {
        'total_studies': 0,
        'valid_studies': 0,
        'studies_dropped_incomplete': 0,
        'total_modalities_checked': 0,
        'valid_modalities': 0,
    }

    for idx, (pdir, mj) in enumerate(zip(processed_dirs, modalities_jsons)):
        logger.info(f"Loading modalities JSON [{idx}]: {mj}")
        with open(mj, 'r') as f:
            modalities_data = json.load(f)
        logger.info(f"  Loaded {len(modalities_data)} studies")

        logger.info(f"Checking processed data completeness for set [{idx}]...")
        set_uids, set_stats = collect_valid_series_uids(pdir, modalities_data, logger)

        logger.info(f"  Completeness results for set [{idx}]:")
        logger.info(f"    Total studies in JSON          : {set_stats['total_studies']}")
        logger.info(f"    Valid studies                  : {set_stats['valid_studies']}")
        logger.info(f"    Studies dropped (incomplete)   : {set_stats['studies_dropped_incomplete']}")
        logger.info(f"    Total modalities checked       : {set_stats['total_modalities_checked']}")
        logger.info(f"    Valid modalities               : {set_stats['valid_modalities']}")
        logger.info(f"    Valid SeriesInstanceUIDs       : {len(set_uids)}")

        duplicate_uids = valid_uids & set_uids
        if duplicate_uids:
            logger.error(
                f"Duplicate SeriesInstanceUIDs detected: set [{idx}] shares {len(duplicate_uids)} "
                f"UID(s) already present from a previous input set. "
                f"Each study must appear in exactly one input set. Stopping."
            )
            return 1

        valid_uids.update(set_uids)
        for key in combined_stats:
            combined_stats[key] += set_stats[key]

    logger.info("")
    logger.info("Combined completeness check results:")
    logger.info(f"  Total studies across all sets  : {combined_stats['total_studies']}")
    logger.info(f"  Valid studies                  : {combined_stats['valid_studies']}")
    logger.info(f"  Studies dropped (incomplete)   : {combined_stats['studies_dropped_incomplete']}")
    logger.info(f"  Total modalities checked       : {combined_stats['total_modalities_checked']}")
    logger.info(f"  Valid modalities               : {combined_stats['valid_modalities']}")
    logger.info(f"  Valid SeriesInstanceUIDs total : {len(valid_uids)}")

    if not valid_uids:
        logger.error("No valid modalities found. Check processed data directories.")
        return 1
    
    # Step 2: Load and filter each input CSV, then concatenate
    logger.info("")
    logger.info("Loading and filtering input metadata CSVs...")
    filtered_parts: List[pd.DataFrame] = []
    for idx, icsv in enumerate(input_csvs):
        try:
            df_part = pd.read_csv(icsv, encoding="utf-8", low_memory=False)
        except pd.errors.EmptyDataError:
            logger.warning(f"  [{idx}] CSV is empty (no columns), skipping: {icsv}")
            continue
        logger.info(f"  [{idx}] Loaded {len(df_part)} rows from {icsv}")
        df_part = df_part[df_part['SeriesInstanceUID'].isin(valid_uids)].copy()
        logger.info(f"  [{idx}] Filtered to {len(df_part)} valid rows")
        filtered_parts.append(df_part)

    if not filtered_parts:
        logger.error("All input CSVs were empty or skipped. Nothing to process.")
        return 1

    df_filtered = pd.concat(filtered_parts, ignore_index=True).copy()
    logger.info(f"  Combined: {len(df_filtered)} rows across {len(input_csvs)} CSV(s)")
    
    if len(df_filtered) == 0:
        logger.error("No rows remaining after filtering. Check SeriesInstanceUID matching.")
        return 1
    
    # Step 3: Load patient mapping and add patient_id column
    logger.info("")
    patient_mapping = load_patient_mapping(args.patient_mapping_csv, logger)
    
    # Map patient_id using study_id
    logger.info("")
    logger.info("Adding patient_id column...")
    
    # Map study_id -> patient_id
    df_filtered['patient_id'] = df_filtered['study_id'].apply(
        lambda x: patient_mapping.get(str(x), pd.NA)
    )
    
    missing_patient_ids = df_filtered['patient_id'].isna().sum()
    if missing_patient_ids > 0:
        logger.error(f"  {missing_patient_ids} rows have no patient_id mapping")
        return 1
    
    df_filtered['patient_id'] = df_filtered['patient_id'].astype(str)
    logger.info(f"  Successfully mapped all study IDs to patient IDs")
    
    # Step 4: Convert study_id to anonymized UID
    logger.info("")
    len_unique_study_ids = df_filtered['study_id'].nunique()
    logger.info(f"Converting {len_unique_study_ids} unique study_ids to anonymized UIDs...")
    df_filtered['study_id'] = df_filtered['study_id'].apply(
        lambda x: accession_to_uid(str(x))
    )
    len_unique_uids = df_filtered['study_id'].nunique()
    logger.info(f"  Unique UIDs: {len_unique_uids}")
    
    if len_unique_study_ids != len_unique_uids:
        logger.error(f"Number of unique study_ids does not match number of unique UIDs: {len_unique_study_ids} != {len_unique_uids}")
        return 1

    # Step 5: Add anon_study_date column
    logger.info("")
    study_date_mapping = load_study_date_mapping(args.study_date_mapping, logger)
    logger.info("Adding anon_study_date column...")
    # study_id column now holds anonymized UIDs (from Step 4), so keys match directly
    df_filtered['anon_study_date'] = df_filtered['study_id'].map(study_date_mapping)
    missing_dates = df_filtered['anon_study_date'].isna().sum()
    if missing_dates > 0:
        missing_uids = df_filtered.loc[df_filtered['anon_study_date'].isna(), 'study_id'].unique().tolist()
        logger.error(
            f"  {missing_dates} row(s) across {len(missing_uids)} unique study_uid(s) have no "
            f"anon_study_date mapping. First 5 missing UIDs: {missing_uids[:5]}"
        )
        return 1
    logger.info(f"  Successfully mapped all {len_unique_uids} study_uid(s) to anon_study_date")

    # Step 6: Rename columns
    # !!! Be carefull, study_id and patient_id becomes study_uid and patient_uid !!!
    logger.info("")
    logger.info("Renaming columns...")
    cols_to_rename = {old: new for old, new in column_rename_map.items() if old in df_filtered.columns}
    if cols_to_rename:
        df_filtered = df_filtered.rename(columns=cols_to_rename)
        logger.info(f"  Renamed: {list(cols_to_rename.items())}")
    else:
        logger.warning("  No columns to rename")
    
    # Step 7: Map classified_modality values
    logger.info("")
    logger.info("Mapping classified_modality values...")
    if 'classified_modality' in df_filtered.columns:
        original_values = df_filtered['classified_modality'].unique()
        df_filtered['classified_modality'] = df_filtered['classified_modality'].replace(modality_value_map)
        mapped_count = sum(df_filtered['classified_modality'].isin(modality_value_map.values()))
        logger.info(f"  Original unique values: {sorted(original_values)}")
        logger.info(f"  Mapped {mapped_count} rows using: {modality_value_map}")
    else:
        logger.error("  'classified_modality' column not found")
        return 1
    
    # Step 8: Drop sensitive/unnecessary columns
    logger.info("")
    logger.info("Dropping columns...")
    cols_to_drop = [col for col in columns_to_drop if col in df_filtered.columns]
    cols_not_found = [col for col in columns_to_drop if col not in df_filtered.columns]
    
    if cols_not_found:
        logger.warning(f"  Some column to drop not found (skipped): {cols_not_found}")
    
    df_filtered = df_filtered.drop(columns=cols_to_drop)
    logger.info(f"  Dropped {len(cols_to_drop)} columns")
    logger.info(f"  Remaining columns: {len(df_filtered.columns)}")
    
    # Step 9: Reorder columns
    logger.info("")
    logger.info("Reordering columns...")
    current_cols = list(df_filtered.columns)
    
    ordered_cols = [col for col in column_order_prefix if col in current_cols]
    remaining_cols = sorted([col for col in current_cols if col not in column_order_prefix])
    ordered_cols.extend(remaining_cols)
    
    df_filtered = df_filtered[ordered_cols]
    logger.info(f"  First columns: {ordered_cols[:len(column_order_prefix)]}")
    logger.info(f"  Remaining {len(remaining_cols)} columns sorted alphabetically")
    
    # Step 10: Sort rows by patient_uid, study_uid, SeriesNumber
    logger.info("")
    logger.info("Sorting rows...")
    sort_cols = ['patient_uid', 'study_uid', 'SeriesNumber']
    df_filtered = df_filtered.sort_values(by=sort_cols).reset_index(drop=True)
    logger.info(f"  Sorted by: {sort_cols}")
    
    # Step 11: Save output CSV
    logger.info("")
    logger.info("Saving output CSV...")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(args.output_csv, index=False)
    logger.info(f"  Saved to: {args.output_csv}")
    logger.info(f"  Total rows: {len(df_filtered)}")
    logger.info(f"  Unique studies: {df_filtered['study_uid'].nunique()}")
    logger.info(f"  Unique patients: {df_filtered['patient_uid'].nunique()}")
    
    # Step 12: Upload to Hugging Face
    logger.info("")
    if args.skip_upload:
        logger.info("--skip-upload set; skipping Hugging Face upload.")
    else:
        hf_path = f"metadata/{args.batch_id}_metadata.csv"
        logger.info(f"Uploading to Hugging Face as {hf_path}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(args.output_csv),
                path_in_repo=hf_path,
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            logger.info(f"  Uploaded to {args.repo_id}/{hf_path}")
        except Exception as e:
            logger.error(f"  Upload failed: {e}")
            return 1
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset Preparation Complete")
    logger.info("=" * 60)
    logger.info(f"Input studies: {combined_stats['total_studies']}")
    logger.info(f"Output studies: {df_filtered['study_uid'].nunique()}")
    logger.info(f"Output modalities: {len(df_filtered)}")
    logger.info(f"Output patients: {df_filtered['patient_uid'].nunique()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
