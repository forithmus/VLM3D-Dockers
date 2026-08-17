"""
MR-RATE MRI Preprocessing Pipeline - Modality Filtering Block

This script filters series of MRI studies based on rule-based classification
and creates outputs for further processing in the MR pipeline.

Input:
    - Classified metadata CSV (output of series_classification.py)
    - Raw MRI data directory (contains study folders with NIfTI files)

Output:
    - Filtered CSV with additional image properties (shape, spacing, FOV)
    - JSON file with center/moving modality structure for downstream processing

Constructs standardized modality IDs from classification fields
(format: {modality}-{role}-{plane}, e.g., "t1w-raw-sag")
and handles multiple runs with the same configuration by adding numeric suffixes (-2, -3, etc.).

Supports parallel processing with configurable number of processes.

Example Usage:
    python src/mr_rate_preprocessing/mri_preprocessing/modality_filtering.py \\
        --raw-data-dir data/raw/batch00/batch00_raw_niftis \\
        --classified-csv data/interim/batch00/batch00_raw_metadata_classified.csv \\
        --output-json data/interim/batch00/batch00_modalities_to_process.json \\
        --output-csv data/interim/batch00/batch00_modalities_to_process_metadata.csv \\
        --num-processes 4 \\
        --log-dir logs/batch00 \\
        --verbose
"""

import argparse
import json
import logging
import sys
import time
from multiprocessing import Pool, Queue
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from mr_rate_preprocessing.configs.config_mri_preprocessing import (
    ACCEPTED_CLASSIFIED_MODALITIES,
    ACCEPTED_ACQUISITION_PLANES,
    CENTER_MODALITY_CRITERIA,
    EXCLUDED_DWI_SUBTYPES,
    FIELD_ABBREVIATIONS,
    INCLUDE_DERIVED_SERIES,
    MIN_PATIENT_AGE,
    MIN_SHAPE,
    MIN_FOV,
    MAX_FOV,
    REQUIRED_METADATA_COLUMNS
)
from utils import (
    setup_parallel_logging,
    BufferedStudyLogger,
)


# Global variable for worker processes (set by worker_init)
# Cannot be passed as arguments because Queue objects can't be pickled
_worker_log_queue = None


def parse_patient_age(age_str: str) -> Optional[float]:
    """
    Parse patient age from DICOM format (e.g., '037Y', '063Y', '006M').
    
    Handles both years ('Y') and months ('M') suffixes.
    Converts months to years (e.g., '006M' -> 0.5 years).
    
    Args:
        age_str: Age string from DICOM metadata
        
    Returns:
        Age in years as float, or None if parsing fails
    """
    if pd.isna(age_str) or not age_str:
        return None
    
    age_str = str(age_str).strip().upper()

    if len(age_str) != 4:
        return None
    
    unit = age_str[-1]  # 'Y' or 'M'
    
    try:
        value = int(age_str[:-1])
        
        if unit == 'Y':
            return int(value)
        elif unit == 'M':
            return int(value / 12.0)  # Convert months to years
        else:
            return None
    except (ValueError, AttributeError):
        return None


def get_dcm2niix_filename(series_description: str, series_number: float) -> str:
    """
    Get dcm2niix filename from series description.
    
    Args:
        series_description: Series description
        series_number: Series number
        
    Returns:
        Predicted NIfTI filename (e.g., "6_t1_mprage_sag_p2_iso.nii.gz")
    """
    return f"{int(series_number)}_{('_').join(str(series_description).split())}.nii.gz"


def get_acquisition_plane(iop_value, oblique_threshold: float = 0.9) -> str:
    """
    Get acquisition plane from ImageOrientation(Patient) value.
    
    Uses the cross product of row and column direction cosines to determine
    the normal vector to the image plane, then identifies the dominant axis.

    Reference:
    - https://stackoverflow.com/questions/34782409/understanding-dicom-image-attributes-to-get-axial-coronal-sagittal-cuts
    - https://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

    The ImageOrientationPatient attribute contains 6 values:
    [row_x, row_y, row_z, col_x, col_y, col_z]
    
    The row vector defines the direction of increasing column index.
    The col vector defines the direction of increasing row index.
    The cross product (row × col) gives the normal to the image plane.
    
    The dominant component of the normal indicates the plane:
    - X-dominant (|1, 0, 0|) → SAGITTAL (left-right view)
    - Y-dominant (|0, 1, 0|) → CORONAL (front-back view)  
    - Z-dominant (|0, 0, 1|) → AXIAL/Transverse (top-bottom view)
    
    Args:
        iop_value: ImageOrientation(Patient) value from DICOM metadata
        oblique_threshold: Threshold for determining oblique planes. If the dominant
                          component is below this threshold, the plane is considered oblique.
                          Default 0.9 means arccos(0.9) ≈ 25.8° from cardinal axis.
        
    Returns:
        Acquisition plane string (AXIAL, SAGITTAL, CORONAL, OBLIQUE, UNKNOWN)
    """
    if pd.isna(iop_value) or not iop_value:
        return "UNKNOWN"
    
    try:
        iop_str = str(iop_value)
        vals = [float(v) for v in iop_str.replace('\\', ',').split(',')]
        if len(vals) == 6:
            row_vec = np.array(vals[0:3])
            col_vec = np.array(vals[3:6])
            normal = np.cross(row_vec, col_vec)
            normal = np.abs(normal)
            
            # Find the dominant axis (largest component)
            max_val = np.max(normal)
            
            # If the normal isn't strongly aligned to any cardinal axis, call it oblique
            if max_val < oblique_threshold:
                return "OBLIQUE"
            
            max_idx = np.argmax(normal)
            if max_idx == 0:
                return "SAGITTAL"
            elif max_idx == 1:
                return "CORONAL"
            else:
                return "AXIAL"
    except (ValueError, TypeError, IndexError):
        pass
    
    return "UNKNOWN"


def load_classified_metadata(
    classified_csv: Path, 
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load classified metadata CSV and prepare it for filtering.
    
    Args:
        classified_csv: Path to classified metadata CSV
        logger: Logger instance
        
    Returns:
        DataFrame with added study_id, raw_nifti_filename, and patient_age columns
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info(f"Loading classified metadata from: {classified_csv}")
    
    df = pd.read_csv(classified_csv, encoding="utf-8", low_memory=False)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Verify required columns exist
    missing_cols = [col for col in REQUIRED_METADATA_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Make sure SeriesNumber is a float and not an string as it is used for sorting 
    # (we don't want lexicographic sorting where 10 is before 2 for example)
    df['SeriesNumber'] = df['SeriesNumber'].astype(float)

    # Copy accession numbers to study_id column
    df['study_id'] = df['AccessionNumber'].astype(str)
    
    # Parse patient ages
    df['patient_age'] = df["Patient'sAge"].apply(parse_patient_age)
    
    # Generate dcm2niix filenames
    df['raw_nifti_filename'] = df.apply(
        lambda row: get_dcm2niix_filename(row['SeriesDescription'], row['SeriesNumber']), 
        axis=1
    )
    
    # Calculate acquisition plane from ImageOrientation(Patient)
    df['acquisition_plane'] = df["ImageOrientation(Patient)"].apply(get_acquisition_plane)
    
    logger.info(f"  Found {df['study_id'].nunique()} unique studies")
    
    return df


def filter_by_config(
    df: pd.DataFrame, 
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter DataFrame by processing config criteria using vectorized operations.
    
    Args:
        df: DataFrame to filter
        logger: Logger instance
        
    Returns:
        Tuple of (filtered DataFrame, stats dict with total and included counts)
    """
    logger.info("Filtering by processing config criteria...")
    total = len(df)
    
    # Apply filters

    logger.info(f"  Accepted modalities: {ACCEPTED_CLASSIFIED_MODALITIES}")
    modality_mask = df['classified_modality'].isin(ACCEPTED_CLASSIFIED_MODALITIES)
    logger.info(f"  classified_modality filter passed: {modality_mask.sum()} / {len(df)}")

    logger.info(f"  Excluding localizer (is_localizer = True) and subtraction (is_subtraction = True) series")
    localizer_mask = ~(df['is_localizer'].astype(bool))
    subtraction_mask = ~(df['is_subtraction'].astype(bool))
    logger.info(f"  is_localizer filter passed: {localizer_mask.sum()} / {len(df)}")
    logger.info(f"  is_subtraction filter passed: {subtraction_mask.sum()} / {len(df)}")
    
    logger.info(f"  Include derived: {INCLUDE_DERIVED_SERIES}")
    derived_mask = (pd.Series([True] * len(df), index=df.index) if INCLUDE_DERIVED_SERIES 
                   else ~(df['is_derived'].astype(bool)))
    logger.info(f"  is_derived filter passed: {derived_mask.sum()} / {len(df)}")
    
    logger.info(f"  Accepted acquisition planes: {ACCEPTED_ACQUISITION_PLANES}")
    plane_mask = df['acquisition_plane'].isin(ACCEPTED_ACQUISITION_PLANES)
    logger.info(f"  acquisition_plane filter passed: {plane_mask.sum()} / {len(df)}")

    # DWI sub-type filter (if applicable)
    logger.info(f"  Excluded DWI sub-types: {EXCLUDED_DWI_SUBTYPES}")
    if 'DWI' in ACCEPTED_CLASSIFIED_MODALITIES and EXCLUDED_DWI_SUBTYPES:
        dwi_subtype_mask = ~((df['classified_modality'] == 'DWI') & df['dwi_sub_type'].isin(EXCLUDED_DWI_SUBTYPES))
    else:
        dwi_subtype_mask = pd.Series([True] * len(df), index=df.index)
    logger.info(f"  dwi_subtype filter passed: {dwi_subtype_mask.sum()} / {len(df)}")
    
    logger.info(f"  Minimum patient age: {MIN_PATIENT_AGE} years")
    age_mask = (df['patient_age'].notna()) & (df['patient_age'] >= MIN_PATIENT_AGE)
    logger.info(f"  patient_age filter passed: {age_mask.sum()} / {len(df)}")
    
    # Combine all filters
    combined_mask = modality_mask & localizer_mask & subtraction_mask & derived_mask & plane_mask & dwi_subtype_mask & age_mask
    filtered_df = df[combined_mask].copy()
    
    stats = {
        'total': total,
        'included': len(filtered_df),
    }
    
    logger.info(f"  Total series: {stats['total']}")
    logger.info(f"  Included: {stats['included']}")
    logger.info(f"  Excluded: {stats['total'] - stats['included']}")
    
    return filtered_df, stats


def load_image_properties(nifti_path: Path) -> Optional[Dict]:
    """
    Load shape, spacing, FOV, and max intensity from NIfTI file.
    
    Args:
        nifti_path: Path to NIfTI file
        
    Returns:
        Dict with shape, spacing, fov, or None if loading fails
    """
    try:
        img = nib.load(str(nifti_path))
        shape = np.array(img.shape)
        spacing = np.array(img.header.get_zooms()[:3])
        assert len(shape) == 3 and len(spacing) == 3
        fov = shape * spacing
        
        return {
            'shape': shape,
            'spacing': spacing,
            'fov': fov,
        }
    except Exception:
        return None


def check_image_quality(img_props: Dict) -> Tuple[bool, str]:
    """
    Check if image meets quality criteria.
    
    Args:
        img_props: Dictionary containing image properties
        
    Returns:
        Tuple of (passes_check, reason_if_failed)
    """
    # Check shape >= MIN_SHAPE in all dimensions
    if np.any(img_props['shape'] < MIN_SHAPE):
        return False, f"shape too small (< {MIN_SHAPE}): {img_props['shape'].tolist()}"
    
    # Check FOV >= MIN_FOV in all dimensions
    if np.any(img_props['fov'] < MIN_FOV):
        return False, f"FOV too small (< {MIN_FOV}mm): {img_props['fov'].tolist()}"

    # Check FOV <= MAX_FOV in all dimensions
    if np.any(img_props['fov'] > MAX_FOV):
        return False, f"FOV too large (> {MAX_FOV}mm): {img_props['fov'].tolist()}"
    
    return True, ""


def construct_modality_name(row: pd.Series) -> str:
    """
    Construct a standardized modality name from classification fields.

    Format: {modality}-{role}-{plane}
    Example: "t1w-raw-sag"

    Args:
        row: DataFrame row with classification columns

    Returns:
        Abbreviated modality name string
    """
    modality_abbr = FIELD_ABBREVIATIONS['classified_modality'][row['classified_modality']]
    role_abbr = FIELD_ABBREVIATIONS['is_derived'][row['is_derived']]
    plane_abbr = FIELD_ABBREVIATIONS['acquisition_plane'][row['acquisition_plane']]

    return f"{modality_abbr}-{role_abbr}-{plane_abbr}"


def find_center_modality(series_df: pd.DataFrame, logger: Optional[BufferedStudyLogger] = None) -> Optional[str]:
    """
    Find the modality ID matching center modality criteria.
    
    Args:
        series_df: DataFrame with series for a single study (must have modality_id column)
        logger: Optional BufferedStudyLogger instance (currently unused)
        
    Returns:
        modality_id of center modality, or None if not found
    """
    # Filter series matching center criteria
    mask = pd.Series([True] * len(series_df), index=series_df.index)
    
    for field, value in CENTER_MODALITY_CRITERIA.items():
        if field in series_df.columns:
            mask &= (series_df[field] == value)
    
    matching = series_df[mask]
    
    if len(matching) == 0:
        return None
    
    if len(matching) >= 1:
        return matching.iloc[0]['modality_id']


def process_study(
    study_id: str,
    study_df: pd.DataFrame,
    raw_data_dir: Path,
) -> Tuple[str, bool, str, Optional[pd.DataFrame], Optional[Dict], Dict]:
    """
    Process a single study: load images, check quality, build outputs.
    
    Series are sorted by SeriesNumber to ensure deterministic processing order
    and center modality selection (earliest series gets priority).
    
    Args:
        study_id: Study ID
        study_df: DataFrame rows for this study
        raw_data_dir: Path to raw data directory
        
    Returns:
        Tuple of (study_id, success, message, enriched DataFrame, JSON dict, stats)
        Returns (study_id, False, message, None, None, stats) if no valid center modality found
        
    Note:
        Uses global _worker_log_queue set by worker_init.
        Logs are buffered and flushed at the end to prevent interleaving.
    """
    global _worker_log_queue
    logger = BufferedStudyLogger(_worker_log_queue, study_id)
    
    start_time = time.time()
    logger.info(f"Processing study: {study_id}")
    
    # Initialize per-study stats
    stats = {
        'missing_study_dir': 0,
        'no_valid_series': 0,
        'no_center': 0,
        'studies_processed': 0,
        'missing_nifti': 0,
        'failed_load': 0,
        'quality_failed': 0,
        'phase_mag_series': 0,
        'multiple_runs': 0,
        'valid_series': 0,
        'series_to_process': 0,
    }
    
    try:
        # Sort by SeriesNumber to ensure deterministic order (earliest series first)
        study_df = study_df.sort_values('SeriesNumber', ascending=True, inplace=False).reset_index(drop=True)
        
        study_dir = raw_data_dir / study_id
        
        if not study_dir.exists():
            logger.warning(f"  Study directory not found: {study_dir}")
            stats['missing_study_dir'] += 1
            return (study_id, False, "Study directory not found", None, None, stats)
        
        valid_rows = []
        modality_run_counts = {}  # Track how many runs share the same modality name
        
        for _, row in study_df.iterrows():
            filename = row['raw_nifti_filename']
            nifti_path = study_dir / filename
            
            # Skip phase/magnitude series
            if any(pattern in filename for pattern in ['_Mag_', '_Pha_', '_Mag.', '_Pha.']):
                logger.warning(f"    Skipping phase/magnitude: {filename}")
                stats['phase_mag_series'] += 1
                continue
            
            # Check if NIfTI file exists
            if not nifti_path.exists():
                logger.warning(f"    Missing NIfTI file: {filename}")
                stats['missing_nifti'] += 1
                continue

            # Load image properties
            props = load_image_properties(nifti_path)
            if props is None:
                logger.warning(f"    Failed to load image: {filename}")
                stats['failed_load'] += 1
                continue
            
            # Check quality
            passes, reason = check_image_quality(props)
            if not passes:
                logger.warning(f"    Quality check failed for {filename}: {reason}")
                stats['quality_failed'] += 1
                continue            
            
            # Construct modality name
            base_modality_name = construct_modality_name(row)
            
            # Handle multiple runs with the same modality name
            count = modality_run_counts.get(base_modality_name, 0)
            modality_run_counts[base_modality_name] = count + 1

            if count > 0:
                modality_id = f"{base_modality_name}-{count + 1}"
                logger.warning(f"    Multiple runs detected: {filename} -> {modality_id}")
                stats['multiple_runs'] += 1
            else:
                modality_id = base_modality_name
            
            # Build enriched row
            enriched_row = row.copy()
            enriched_row['modality_id'] = modality_id
            enriched_row['ras_array_shape'] = props['shape'].tolist()
            enriched_row['ras_array_spacing_mm'] = props['spacing'].tolist()
            enriched_row['ras_array_fov_mm'] = props['fov'].tolist()
            
            valid_rows.append(enriched_row)
            stats['valid_series'] += 1
        
        if not valid_rows:
            logger.warning(f"  No valid series for study {study_id}")
            stats['no_valid_series'] += 1
            return (study_id, False, "No valid series", None, None, stats)
        
        # Create DataFrame from valid rows
        valid_df = pd.DataFrame(valid_rows)
        
        # Find center modality
        center_modality_id = find_center_modality(valid_df, logger)
        if center_modality_id is None:
            logger.warning(f"  No center modality found for study {study_id}")
            stats['no_center'] += 1
            return (study_id, False, "No center modality found", None, None, stats)
        
        # Mark center modality
        valid_df['is_center_modality'] = valid_df['modality_id'] == center_modality_id
        
        # Get StudyInstanceUID
        study_instance_uid = valid_df.iloc[0].get('StudyInstanceUID')
        
        # Build JSON structure
        center_row = valid_df[valid_df['is_center_modality']].iloc[0]
        moving_rows = valid_df[~valid_df['is_center_modality']]
        
        def row_to_json(r):
            """Convert a row to JSON-compatible dict."""
            return {
                'series_instance_uid': str(r['SeriesInstanceUID']),
                'img_path': r['raw_nifti_filename'],
            }
        
        json_output = {
            'study_instance_uid': str(study_instance_uid),
            'center_modality': {
                center_modality_id: row_to_json(center_row)
            },
            'moving_modality': {
                row['modality_id']: row_to_json(row)
                for _, row in moving_rows.iterrows()
            }
        }
        
        elapsed = time.time() - start_time
        logger.info(f"  Study {study_id}: {len(valid_df)} valid series, center={center_modality_id}")
        logger.info(f"  Completed {study_id} in {elapsed:.1f}s")
        
        stats['studies_processed'] += 1
        stats['series_to_process'] += len(valid_df)
        
        return (study_id, True, f"Success ({elapsed:.1f}s)", valid_df, json_output, stats)
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        logger.error(f"  Failed {study_id} after {elapsed:.1f}s: {error_msg}")
        return (study_id, False, error_msg, None, None, stats)
    
    finally:
        # Always flush buffered logs when study processing completes
        logger.flush()


def worker_init(log_queue: Queue):
    """
    Initialize worker process with logging queue.
    
    Args:
        log_queue: Queue for logging (inherited, not pickled)
    """
    global _worker_log_queue
    
    # Store log queue in global (Queue can't be passed as function args)
    _worker_log_queue = log_queue


def worker_process_study(args):
    """
    Wrapper for process_study to work with Pool.map.
    
    Args:
        args: Tuple of arguments for process_study
        
    Returns:
        Result from process_study
    """
    return process_study(*args)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter MRI series based on rule-based classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process with 4 parallel workers
    python src/mr_rate_preprocessing/mri_preprocessing/modality_filtering.py \\
        --raw-data-dir data/raw/batch00/batch00_raw_niftis \\
        --classified-csv data/interim/batch00/batch00_raw_metadata_classified.csv \\
        --output-json data/interim/batch00/batch00_modalities_to_process.json \\
        --output-csv data/interim/batch00/batch00_modalities_to_process_metadata.csv \\
        --num-processes 4 \\
        --verbose
        """
    )
    
    parser.add_argument(
        "--raw-data-dir", "-d",
        type=Path,
        required=True,
        help="Path to raw MRI data directory (contains study folders with NIfTI files)",
    )
    
    parser.add_argument(
        "--classified-csv", "-c",
        type=Path,
        required=True,
        help="Path to classified metadata CSV (output of series_classification.py)",
    )
    
    parser.add_argument(
        "--output-json", "-oj",
        type=Path,
        required=True,
        help="Output JSON file path (modalities to process)",
    )
    
    parser.add_argument(
        "--output-csv", "-oc",
        type=Path,
        required=True,
        help="Output CSV file path (filtered metadata with image properties)",
    )
    
    parser.add_argument(
        "--num-processes", "-n",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    
    parser.add_argument(
        "--log-dir", "-l",
        type=Path,
        default=Path("logs"),
        help="Directory for log files (default: logs)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print logs to terminal in addition to log file (default: silent)",
    )
    
    args = parser.parse_args()
    
    script_name = Path(__file__).stem
    
    # Setup logging with queue for multiprocessing support
    logger, log_queue, queue_listener = setup_parallel_logging(
        args.log_dir, script_name, verbose=args.verbose
    )
    
    try:
        logger.info("=" * 60)
        logger.info("MR-RATE Image Processing Pipeline - Modality Filtering (Rule-Based Classification Block")
        logger.info("=" * 60)
        logger.info(f"Raw data directory: {args.raw_data_dir}")
        logger.info(f"Classified CSV: {args.classified_csv}")
        logger.info(f"Output JSON: {args.output_json}")
        logger.info(f"Output CSV: {args.output_csv}")
        logger.info(f"Log directory: {args.log_dir}")
        logger.info(f"Parallel processes: {args.num_processes}")
        logger.info("")
        logger.info("Quality Control Thresholds:")
        logger.info(f"  Minimum age: {MIN_PATIENT_AGE} years")
        logger.info(f"  Minimum shape: {MIN_SHAPE} voxels")
        logger.info(f"  Minimum FOV: {MIN_FOV} mm")
        logger.info(f"  Maximum FOV: {MAX_FOV} mm")
        logger.info("")
        
        # Validate inputs
        if not args.raw_data_dir.exists():
            logger.error(f"Raw data directory not found: {args.raw_data_dir}")
            return 1
        
        if not args.classified_csv.exists():
            logger.error(f"Classified CSV not found: {args.classified_csv}")
            return 1
        
        if args.output_json.exists():
            logger.error(f"Output JSON already exists: {args.output_json}")
            return 1
        
        if args.output_csv.exists():
            logger.error(f"Output CSV already exists: {args.output_csv}")
            return 1
        
        # Load classified metadata
        try:
            df = load_classified_metadata(args.classified_csv, logger)
        except ValueError as e:
            logger.error(f"Failed to load metadata: {e}")
            return 1
        
        # Filter by classification criteria
        df_filtered, _ = filter_by_config(df, logger)
        
        # Initialize processing statistics
        stats = {
            # Study-level
            'total_studies': df_filtered['study_id'].nunique(),
            'missing_study_dir': 0,
            'no_valid_series': 0,
            'no_center': 0,
            'studies_processed': 0,
            
            # Series-level
            'total_series_filtered': len(df_filtered),
            'missing_nifti': 0,
            'failed_load': 0,
            'quality_failed': 0,
            'phase_mag_series': 0,
            'multiple_runs': 0,
            'valid_series': 0,
            'series_to_process': 0,
        }
        
        logger.info("")
        logger.info("Processing studies...")
        logger.info("")
        
        # Get study groups as a list
        study_groups = list(df_filtered.groupby('study_id'))
        
        # Prepare arguments for each study
        study_args = [
            (
                study_id,
                study_df,
                args.raw_data_dir,
            )
            for study_id, study_df in study_groups
        ]
        
        # Process studies with progress bar
        logger.info(f"Starting processing with {args.num_processes} process(es)...")
        total_start_time = time.time()
        
        all_valid_dfs = []
        json_results = {}
        results = []
        successful = 0
        failed = 0
        
        if args.num_processes == 1:
            # Single process mode with tqdm progress bar
            worker_init(log_queue)
            
            with tqdm(total=len(study_args), desc="Processing studies", unit="study", disable=not args.verbose) as pbar:
                for study_arg in study_args:
                    result = worker_process_study(study_arg)
                    results.append(result)
                    
                    study_id, success, msg, valid_df, json_output, study_stats = result
                    
                    # Aggregate stats
                    for key in study_stats:
                        stats[key] += study_stats[key]
                    
                    if success and valid_df is not None and json_output is not None:
                        all_valid_dfs.append(valid_df)
                        json_results[study_id] = json_output
                        successful += 1
                    else:
                        failed += 1
                    
                    pbar.set_postfix({"OK": successful, "FAIL": failed})
                    pbar.update(1)
        else:
            # Multi-process mode with Pool and tqdm
            with Pool(
                processes=args.num_processes,
                initializer=worker_init,
                initargs=(log_queue,)
            ) as pool:
                # Use imap_unordered for streaming results with progress bar
                with tqdm(total=len(study_args), desc="Processing studies", unit="study", disable=not args.verbose) as pbar:
                    for result in pool.imap_unordered(worker_process_study, study_args):
                        results.append(result)
                        
                        study_id, success, msg, valid_df, json_output, study_stats = result
                        
                        # Aggregate stats
                        for key in study_stats:
                            stats[key] += study_stats[key]
                        
                        if success and valid_df is not None and json_output is not None:
                            all_valid_dfs.append(valid_df)
                            json_results[study_id] = json_output
                            successful += 1
                        else:
                            failed += 1
                        
                        pbar.set_postfix({"OK": successful, "FAIL": failed})
                        pbar.update(1)
        
        total_elapsed = time.time() - total_start_time
        
        # Combine all valid DataFrames
        if all_valid_dfs:
            output_df = pd.concat(all_valid_dfs, ignore_index=True)
        else:
            output_df = pd.DataFrame()
        
        # Save outputs
        logger.info("")
        logger.info("=" * 60)
        logger.info("Results")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Study-level statistics:")
        logger.info(f"  Total studies (after all filters): {stats['total_studies']}")
        logger.info(f"  Missing study directory: {stats['missing_study_dir']}")
        logger.info(f"  No valid series: {stats['no_valid_series']}")
        logger.info(f"  No center modality: {stats['no_center']}")
        logger.info(f"  Studies processed successfully: {stats['studies_processed']}")
        logger.info("")
        logger.info("Series-level statistics:")
        logger.info(f"  Total series (after classification filter): {stats['total_series_filtered']}")
        logger.info(f"  Missing NIfTI files: {stats['missing_nifti']}")
        logger.info(f"  Failed to load: {stats['failed_load']}")
        logger.info(f"  Quality check failed: {stats['quality_failed']}")
        logger.info(f"  Phase/magnitude skipped: {stats['phase_mag_series']}")
        logger.info(f"  Multiple runs (suffixed): {stats['multiple_runs']}")
        logger.info(f"  Valid series: {stats['valid_series']}")
        logger.info(f"  Series to process: {stats['series_to_process']}")
        logger.info("")
        logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        
        if successful > 0:
            avg_time = total_elapsed / len(study_groups)
            logger.info(f"Average time per study: {avg_time:.2f}s")
        
        # Log failed studies
        if failed > 0:
            logger.info("")
            logger.info("Failed studies:")
            for study_id, success, msg, _, _, _ in results:
                if not success:
                    logger.info(f"  {study_id}: {msg}")
        
        # Save JSON output
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(json_results, f, indent=4)
        logger.info(f"JSON output saved to: {args.output_json}")
        
        # Save CSV output
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output_csv, index=False)
        logger.info(f"CSV output saved to: {args.output_csv} ({len(output_df)} rows)")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Done")
        logger.info("=" * 60)
        
        return 0
    
    finally:
        # Clean up queue listener
        queue_listener.stop()


if __name__ == "__main__":
    sys.exit(main())