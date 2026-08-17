"""
MR-RATE Processing Pipeline - Registration Block

This script applies atlas and co-registration to processed MRI outputs.
For each study, it takes defaced images and segmentation masks in native space, then:
1. Registers center modality to MNI152 atlas
2. Co-registers moving modalities to center modality
3. Transforms moving modalities to atlas space
4. Copies center modality files to coreg space
5. Transforms center modality segmentation masks to atlas space

Output directory structure within output_dir:
    MR-RATE-coreg_{input_dir.name}/
        mri/
            {input_dir.name}/
                {study_uid}/
                    coreg_img/
                        {study_uid}_{center_modality_series_id}.nii.gz        # Center (copy, no coreg prefix)
                        {study_uid}_coreg_{moving_modality_series_id}.nii.gz  # Moving registered to center
                    coreg_seg/                                         # Only center modality segmentations (copy, no coreg prefix)
                        {study_uid}_{center_modality_series_id}_brain-mask.nii.gz
                        {study_uid}_{center_modality_series_id}_defacing-mask.nii.gz
                    transform/
                        M_coreg_{moving_modality_series_id}.mat              # Moving→center transforms

    MR-RATE-atlas_{input_dir.name}/
        mri/
            {input_dir.name}/
                {study_uid}/
                    atlas_img/
                        {study_uid}_atlas_{all_modality_series_id}.nii.gz         # All modalities in atlas space
                    atlas_seg/                                         # Only center modality segmentations
                        {study_uid}_atlas_{center_modality_series_id}_brain-mask.nii.gz
                        {study_uid}_atlas_{center_modality_series_id}_defacing-mask.nii.gz
                    transform/
                        M_atlas_{center_modality_series_id}.mat              # Center→atlas transform

Supports parallel processing with configurable number of processes and threads. 
Safe to rerun, resumes processing from previous runs by cleaning up incomplete outputs and skipping completed studies.
Supports partitioning studies across multiple independent jobs (e.g., SLURM array jobs)
via --total-partitions and --partition-index.

Usage:
    python registration.py \\
        --input-dir /path/to/processed/data \\
        --metadata-csv /path/to/metadata.csv \\
        --output-dir /path/to/output \\
        --num-processes n \\
        --threads-per-process t \\
        --total-partitions N \\
        --partition-index I \\
        --log-dir /path/to/logs \\
        --verbose

"""

import argparse
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from multiprocessing import Pool, Queue

from tqdm import tqdm
import ants
import pandas as pd

from mr_rate_preprocessing.mri_preprocessing.utils import (
    BufferedStudyLogger,
    setup_logging,
    setup_parallel_logging,
    fetch_atlas,
)


# Global variable for worker processes (set by worker_init)
# Cannot be passed as arguments because Queue objects can't be pickled
_worker_log_queue = None


def load_metadata_csv(
    metadata_csv: Path,
    logger: logging.Logger,
) -> Dict[str, Dict]:
    """
    Load metadata CSV and build study dictionary with center/moving modalities.
    
    Args:
        metadata_csv: Path to metadata CSV file
        logger: Logger instance
        
    Returns:
        Dictionary mapping study_id to study data:
        {
            study_id: {
                'center_modality_id': str,
                'moving_modality_ids': List[str]
            }
        }
    """
    logger.info(f"Loading metadata from: {metadata_csv}")
    col_map = {'study_uid': 'study_id', 'series_id': 'modality_id'}
    required_cols = ['study_uid', 'series_id', 'is_center_modality']

    df = pd.read_csv(
        metadata_csv,
        encoding="utf-8",
        low_memory=False,
        usecols=required_cols,
        dtype={'study_uid': str, 'series_id': str, 'is_center_modality': bool}
    )
    df = df.rename(columns=col_map)
    logger.info(f"  Loaded {len(df)} modalities (rows), {df['study_id'].nunique()} unique studies")
    
    # Verify required columns exist (check post-rename names)
    expected_cols = ['study_id', 'modality_id', 'is_center_modality']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Build study dictionary
    study_data = {}
    
    for study_id, study_df in df.groupby('study_id'):
        study_id = str(study_id)
        
        # Find center modality
        center_rows = study_df[study_df['is_center_modality'] == True]
        if len(center_rows) != 1:
            error_msg = f"Study {study_id}: Expected 1 center modality, found {len(center_rows)}. Please check the metadata CSV for errors."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        center_modality_id = center_rows.iloc[0]['modality_id']
        
        # Find moving modalities
        moving_rows = study_df[study_df['is_center_modality'] != True]
        moving_modality_ids = moving_rows['modality_id'].tolist()
        
        study_data[study_id] = {
            'center_modality_id': center_modality_id,
            'moving_modality_ids': moving_modality_ids,
        }
    
    logger.info(f"  Built study dictionary with {len(study_data)} studies")
    
    return study_data


def get_input_img_path(input_dir: Path, study_id: str, modality_id: str) -> Path:
    """Get input image path for a modality."""
    return input_dir / study_id / "img" / f"{study_id}_{modality_id}.nii.gz"


def get_input_brain_mask_path(input_dir: Path, study_id: str, modality_id: str) -> Path:
    """Get input brain mask path for a modality."""
    return input_dir / study_id / "seg" / f"{study_id}_{modality_id}_brain-mask.nii.gz"


def get_input_defacing_mask_path(input_dir: Path, study_id: str, modality_id: str) -> Path:
    """Get input defacing mask path for a modality."""
    return input_dir / study_id / "seg" / f"{study_id}_{modality_id}_defacing-mask.nii.gz"


def check_already_processed(
    input_dir: Path,
    coreg_output_dir: Path,
    atlas_output_dir: Path,
    study_data: Dict[str, Dict],
    logger: logging.Logger,
) -> Tuple[List[str], int, int, int]:
    """
    Check which studies need processing, skip completed or missing input ones, clean incomplete ones.
    
    A study is considered complete if center modality segmentations exist in atlas_seg/ of the
    atlas output directory.
    
    A study is skipped (missing input) if:
    - Center modality image missing
    - Center modality brain-mask missing
    - Center modality defacing-mask missing
    - Any moving modality image missing
    
    Incomplete outputs are cleaned from both coreg and atlas output directories.
    
    Args:
        input_dir: Input data directory (processed data from 4_5_6_hdbet_seg_def.py)
        coreg_output_dir: Output directory for co-registration data (MR-RATE-coreg_{input_dir.name})
        atlas_output_dir: Output directory for atlas registration data (MR-RATE-atlas_{input_dir.name})
        study_data: Dictionary from load_metadata_csv
        logger: Logger instance
        
    Returns:
        Tuple of (list of study_ids to process, skipped_complete, skipped_missing_input, cleaned)
    """
    to_process = []
    skipped_complete = 0
    skipped_missing_input = 0
    cleaned_count = 0
    
    for study_id, data in study_data.items():
        center_modality_id = data['center_modality_id']
        moving_modality_ids = data['moving_modality_ids']
        
        atlas_study_output = atlas_output_dir / study_id
        atlas_seg_dir = atlas_study_output / "atlas_seg"
        
        if atlas_study_output.exists():
            # Check if outputs from final step all exist (center segmentations in atlas_seg)
            atlas_brain_mask = atlas_seg_dir / f"{study_id}_atlas_{center_modality_id}_brain-mask.nii.gz"
            atlas_defacing_mask = atlas_seg_dir / f"{study_id}_atlas_{center_modality_id}_defacing-mask.nii.gz"
                        
            if atlas_brain_mask.exists() and atlas_defacing_mask.exists():
                # Already processed successfully
                skipped_complete += 1
            else:
                # Incomplete processing - clean up both output dirs and reprocess
                logger.info(f"Cleaning incomplete output for {study_id}")
                shutil.rmtree(atlas_study_output)
                coreg_study_output = coreg_output_dir / study_id
                if coreg_study_output.exists():
                    shutil.rmtree(coreg_study_output)
                cleaned_count += 1
                to_process.append(study_id)
        else:            
            # Check if all expected input files exist as download might have failed
            input_files_exist = True
            
            center_img = get_input_img_path(input_dir, study_id, center_modality_id)
            center_brain_mask = get_input_brain_mask_path(input_dir, study_id, center_modality_id)
            center_defacing_mask = get_input_defacing_mask_path(input_dir, study_id, center_modality_id)
            
            if not center_img.exists():
                input_files_exist = False
            elif not center_brain_mask.exists():
                input_files_exist = False
            elif not center_defacing_mask.exists():
                input_files_exist = False
            else:
                for modality_id in moving_modality_ids:
                    moving_img = get_input_img_path(input_dir, study_id, modality_id)
                    if not moving_img.exists():
                        input_files_exist = False
                        break
            
            if not input_files_exist:
                skipped_missing_input += 1
                continue
            else:
                to_process.append(study_id)
        
    return to_process, skipped_complete, skipped_missing_input, cleaned_count


def register(
    fixed_path: Path,
    moving_path: Path,
    matrix_path: Path,
    output_path: Optional[Path] = None,
    outprefix: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Perform rigid registration using ANTsPy.
    
    Args:
        fixed_path: Path to fixed (reference) image
        moving_path: Path to moving image
        matrix_path: Path to save transformation matrix
        output_path: Optional path to save warped moving image directly from registration result
        outprefix: Optional output prefix for temporary files
        logger: Optional logger instance
        
    Returns:
        True if successful, False otherwise

    Notes:
        - ANTsPy uses antsRegistration (NOT ANTS) to perform registration
        - For documentation of ANTsPy registration (ants.registration) see: https://antspy.readthedocs.io/en/latest/registration.html#ants.registration
        - For source code of ANTsPy registration (ants.registration) see: https://antspy.readthedocs.io/en/latest/_modules/ants/registration/registration.html#registration
        - For explanation of antsRegistration's parameters see: https://github.com/ANTsX/ANTs/wiki/ANTS-and-antsRegistration
        - Some important parameters for registration with ANTsPy (antsRegistration):
                   type_of_transform (transform)          = 'Rigid'   => Selected on purpose rather than default 'SyN'.
                          aff_metric (metric)             = 'Mattes'  => ANTsPy default. Same with MI but has a different name historically.
                                     (metricWeight)       = 1         => Constant value in ANTsPy.
                        aff_sampling (numberOfBins)       = 32        => ANTsPy default.
                                     (samplingStrategy)   = 'regular' => Constant value in ANTsPy.
            aff_random_sampling_rate (samplingPercentage) = 0.2       => ANTsPy default.
                                     (interpolation)      = 'linear'  => This is not passed by ANTsPy, relies on antsRegistration's default.
        - outprefix controls where temporary files are written during registration, see https://github.com/ANTsX/ANTsPy/issues/302
    """
    try:
        fixed = ants.image_read(str(fixed_path))
        moving = ants.image_read(str(moving_path))
        
        # Prepare registration kwargs
        reg_kwargs = {
            "fixed": fixed,
            "moving": moving,
            "type_of_transform": "Rigid",
            "aff_metric": "Mattes",
            "aff_sampling": 32,
            "aff_random_sampling_rate": 0.2,
        }
        
        # Add outprefix if provided
        if outprefix is not None:
            reg_kwargs["outprefix"] = outprefix
        
        result = ants.registration(**reg_kwargs)
        
        # Ensure output directory exists
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the transform matrix
        shutil.move(result["fwdtransforms"][0], str(matrix_path))
        
        # Save warped image directly from registration result if output_path provided
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ants.image_write(result["warpedmovout"].astype('float32'), str(output_path))
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Registration failed: {e}")
        return False


def apply_transform(
    fixed_path: Path,
    moving_path: Path,
    transform_paths: Union[Path, List[Path]],
    output_path: Path,
    interpolator: str = "linear",
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Apply transformation(s) to moving image.
    
    Args:
        fixed_path: Path to fixed (reference) image
        moving_path: Path to moving image
        transform_paths: Path or list of paths to transformation matrices.
            When multiple transforms are provided, they are applied right-to-left.
        output_path: Path to save transformed image
        interpolator: Interpolation method ("linear", "nearestNeighbor", etc.)
        logger: Optional logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        fixed = ants.image_read(str(fixed_path))
        moving = ants.image_read(str(moving_path))
        
        # Normalize to list
        if isinstance(transform_paths, Path):
            transform_paths = [transform_paths]
        
        # Convert paths to strings for ANTsPy
        transform_paths = [str(t) for t in transform_paths]
        
        transformed = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=transform_paths,
            interpolator=interpolator,
        )

        if interpolator == "nearestNeighbor":
            # Ensure output dtype is uint8 for masks
            transformed = transformed.astype('uint8')
        else:
            # Ensure output dtype is float32 for images
            transformed = transformed.astype('float32')
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ants.image_write(transformed, str(output_path))
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Transform application failed: {e}")
        return False


def process_study(
    study_id: str,
    study_data: Dict,
    input_dir: Path,
    coreg_output_dir: Path,
    atlas_output_dir: Path,
    atlas_path: Path,
    tmp_dir: Optional[Path] = None,
) -> Tuple[str, bool, str]:
    """
    Process a single study: register to atlas and co-register modalities.
    
    Processing steps:
        1. Register center modality to atlas (matrix saved to atlas output dir)
        2. Copy center modality files to coreg directories
        3. Transform center masks to atlas space
        4. For each moving modality: co-register to center (matrix saved to coreg output dir),
           apply composite transform to get modality in atlas space
    
    Args:
        study_id: Study identifier
        study_data: Study data with center_modality_id and moving_modality_ids
        input_dir: Input data directory
        coreg_output_dir: Output directory for co-registration data (MR-RATE-coreg_{input_dir.name})
        atlas_output_dir: Output directory for atlas registration data (MR-RATE-atlas_{input_dir.name})
        atlas_path: Path to atlas image
        tmp_dir: Optional temporary directory for ANTs intermediate files
        
    Returns:
        Tuple of (study_id, success, message)
    """
    global _worker_log_queue
    
    # Use buffered logger if in multiprocessing mode, otherwise create simple logger
    if _worker_log_queue is not None:
        logger = BufferedStudyLogger(_worker_log_queue, study_id)
    else:
        logger = None
    
    start_time = time.time()
    
    def log_info(msg):
        if logger:
            logger.info(msg)
    
    def log_error(msg):
        if logger:
            logger.error(msg)
    
    log_info(f"Processing study: {study_id}")
    
    try:
        center_modality_id = study_data['center_modality_id']
        moving_modality_ids = study_data['moving_modality_ids']
        
        # Get input paths
        center_img = get_input_img_path(input_dir, study_id, center_modality_id)
        center_brain_mask = get_input_brain_mask_path(input_dir, study_id, center_modality_id)
        center_defacing_mask = get_input_defacing_mask_path(input_dir, study_id, center_modality_id)
        
        # Create output directory structure
        coreg_study_output = coreg_output_dir / study_id
        coreg_img_dir = coreg_study_output / "coreg_img"
        coreg_seg_dir = coreg_study_output / "coreg_seg"
        coreg_transform_dir = coreg_study_output / "transform"
        
        atlas_study_output = atlas_output_dir / study_id
        atlas_img_dir = atlas_study_output / "atlas_img"
        atlas_seg_dir = atlas_study_output / "atlas_seg"
        atlas_transform_dir = atlas_study_output / "transform"
        
        for d in [coreg_img_dir, coreg_seg_dir, coreg_transform_dir,
                  atlas_img_dir, atlas_seg_dir, atlas_transform_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Register center modality to atlas (matrix goes to atlas transform dir)
        log_info(f"  Registering center ({center_modality_id}) -> atlas")
        
        atlas_matrix_path = atlas_transform_dir / f"M_atlas_{center_modality_id}.mat"
        atlas_center_output = atlas_img_dir / f"{study_id}_atlas_{center_modality_id}.nii.gz"
        
        # Create unique outprefix for this registration if tmp_dir provided
        outprefix = None
        if tmp_dir is not None:
            outprefix = str(tmp_dir / f"{study_id}_atlas_{center_modality_id}_")
        
        success = register(
            fixed_path=atlas_path,
            moving_path=center_img,
            matrix_path=atlas_matrix_path,
            output_path=atlas_center_output,
            outprefix=outprefix,
            logger=logger,
        )
        
        if not success:
            raise Exception("Atlas registration failed")
        
        # Step 2: Co-register moving modalities to center
        log_info(f"  Processing {len(moving_modality_ids)} moving modalities")
        
        for modality_id in moving_modality_ids:
            moving_img = get_input_img_path(input_dir, study_id, modality_id)
            
            # Co-reg matrix goes to coreg transform dir; atlas moving output goes to atlas img dir
            coreg_matrix_path = coreg_transform_dir / f"M_coreg_{modality_id}.mat"
            coreg_moving_output = coreg_img_dir / f"{study_id}_coreg_{modality_id}.nii.gz"
            atlas_moving_output = atlas_img_dir / f"{study_id}_atlas_{modality_id}.nii.gz"
            
            # Co-register moving to center
            log_info(f"    Registering {modality_id} -> center ({center_modality_id})")
            
            outprefix = None
            if tmp_dir is not None:
                outprefix = str(tmp_dir / f"{study_id}_coreg_{modality_id}_")
            
            success = register(
                fixed_path=center_img,
                moving_path=moving_img,
                matrix_path=coreg_matrix_path,
                output_path=coreg_moving_output,
                outprefix=outprefix,
                logger=logger,
            )
            
            if not success:
                raise Exception(f"Co-registration failed for {modality_id}")
            
            # Apply composite transform (coreg + atlas) to get moving modality in atlas space
            log_info(f"    Transforming {modality_id} -> atlas")
            
            # Transforms applied right-to-left: first coreg (moving->center), then atlas (center->atlas)
            success = apply_transform(
                fixed_path=atlas_path,
                moving_path=moving_img,
                transform_paths=[atlas_matrix_path, coreg_matrix_path],
                output_path=atlas_moving_output,
                interpolator="linear",
                logger=logger,
            )
            
            if not success:
                raise Exception(f"Composite transform failed for {modality_id}")
        
        # Step 3: Copy center modality files to coreg directories (no transformation needed)
        log_info(f"  Copying center modality files to coreg directories")
        
        coreg_center_img = coreg_img_dir / f"{study_id}_{center_modality_id}.nii.gz"
        coreg_center_brain_mask = coreg_seg_dir / f"{study_id}_{center_modality_id}_brain-mask.nii.gz"
        coreg_center_defacing_mask = coreg_seg_dir / f"{study_id}_{center_modality_id}_defacing-mask.nii.gz"
        
        shutil.copy(center_img, coreg_center_img)
        shutil.copy(center_brain_mask, coreg_center_brain_mask)
        shutil.copy(center_defacing_mask, coreg_center_defacing_mask)
        
        # Step 4: Transform center masks to atlas space
        log_info(f"  Transforming center masks to atlas space")
        
        atlas_brain_mask = atlas_seg_dir / f"{study_id}_atlas_{center_modality_id}_brain-mask.nii.gz"
        atlas_defacing_mask = atlas_seg_dir / f"{study_id}_atlas_{center_modality_id}_defacing-mask.nii.gz"
        
        success = apply_transform(
            fixed_path=atlas_path,
            moving_path=center_brain_mask,
            transform_paths=atlas_matrix_path,
            output_path=atlas_brain_mask,
            interpolator="nearestNeighbor",
            logger=logger,
        )
        if not success:
            raise Exception("Center brain mask atlas transform failed")
        
        success = apply_transform(
            fixed_path=atlas_path,
            moving_path=center_defacing_mask,
            transform_paths=atlas_matrix_path,
            output_path=atlas_defacing_mask,
            interpolator="nearestNeighbor",
            logger=logger,
        )
        if not success:
            raise Exception("Center defacing mask atlas transform failed")
        
        elapsed = time.time() - start_time
        log_info(f"  Completed {study_id} in {elapsed:.1f}s")
        
        return (study_id, True, f"Success ({elapsed:.1f}s)")
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        log_error(f"  Failed {study_id} after {elapsed:.1f}s: {error_msg}")
        
        # Clean up incomplete outputs from both directories
        for study_output in [coreg_output_dir / study_id, atlas_output_dir / study_id]:
            if study_output.exists():
                shutil.rmtree(study_output)
                log_info(f"  Cleaned up incomplete output: {study_output}")
        
        return (study_id, False, error_msg)
    
    finally:
        # Always flush buffered logs when study processing completes
        if logger and hasattr(logger, 'flush'):
            logger.flush()


def worker_init(n_threads: int, log_queue: Queue):
    """
    Initialize worker process with thread limits and logging queue.
    
    Args:
        n_threads: Number of threads for ANTs/ITK
        log_queue: Queue for logging (inherited, not pickled)
    """
    global _worker_log_queue
    
    # Store log queue in global (Queue can't be passed as function args)
    _worker_log_queue = log_queue
    
    # Set thread limits for ANTs/ITK
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(n_threads)
    os.environ["ANTS_RANDOM_SEED"] = "42"
    
    # Also set common thread-limiting environment variables
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)


def worker_process_study(args):
    """
    Wrapper for process_study to work with Pool.map.
    
    Args:
        args: Tuple of arguments for process_study
        
    Returns:
        Result from process_study
    """
    return process_study(*args)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MR-RATE Registration Block - Apply atlas and co-registration to processed MRI outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process with 2 parallel workers, 2 threads each (Run from MR-RATE root directory)
    python registration.py \\
        --input-dir data/MR-RATE/mri/batchXX \\
        --metadata-csv data/MR-RATE/metadata/batchXX_metadata.csv \\
        --output-dir data/MR-RATE-reg \\
        --num-processes 2 \\
        --threads-per-process 2 \\
        --log-dir logs \\
        --verbose

    # Split across 4 independent jobs (e.g., SLURM array jobs)
    python registration.py \\
        --input-dir data/MR-RATE/mri/batchXX \\
        --metadata-csv data/MR-RATE/metadata/batchXX_metadata.csv \\
        --output-dir data/MR-RATE-reg \\
        --total-partitions 4 \\
        --partition-index $SLURM_ARRAY_TASK_ID \\
        --log-dir logs

    # Outputs will be written to:
    #   data/MR-RATE-reg/MR-RATE-coreg_batchXX/mri/batchXX/
    #   data/MR-RATE-reg/MR-RATE-atlas_batchXX/mri/batchXX/
        """
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Path to processed data directory"
    )
    
    parser.add_argument(
        "--metadata-csv", "-m",
        type=Path,
        required=True,
        help="Path to metadata CSV file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help=(
            "Root output directory. Two subdirectories are created inside: "
            "MR-RATE-coreg_{input_dir.name}/mri/{input_dir.name} (co-registration outputs) and "
            "MR-RATE-atlas_{input_dir.name}/mri/{input_dir.name} (atlas registration outputs)"
        )
    )
    
    parser.add_argument(
        "--num-processes", "-n",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)"
    )
    
    parser.add_argument(
        "--threads-per-process", "-t",
        type=int,
        default=4,
        help="Number of threads per process for ANTs (default: 4)"
    )
    
    parser.add_argument(
        "--tmp-base-dir",
        type=Path,
        default=None,
        help="Base directory for temporary ANTs files (default: system temp). A unique subdirectory is created per run and cleaned up automatically."
    )
    
    parser.add_argument(
        "--total-partitions",
        type=int,
        default=1,
        help="Total number of partitions to split studies across (default: 1, no partitioning)"
    )

    parser.add_argument(
        "--partition-index",
        type=int,
        default=0,
        help="Zero-based index of this partition (default: 0). Must be in [0, total-partitions)"
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
    
    return parser.parse_args()


def main():
    """Main entry point for the registration pipeline."""
    args = parse_args()
    
    script_name = Path(__file__).stem
    
    # Setup logging based on number of processes
    if args.num_processes == 1:
        logger = setup_logging(args.log_dir, script_name, verbose=args.verbose)
        log_queue = None
        queue_listener = None
    else:
        logger, log_queue, queue_listener = setup_parallel_logging(
            args.log_dir, script_name, verbose=args.verbose
        )
    
    # Validate partition arguments
    if args.total_partitions < 1:
        logger.error(f"--total-partitions must be >= 1, got {args.total_partitions}")
        return 1
    if args.partition_index < 0 or args.partition_index >= args.total_partitions:
        logger.error(
            f"--partition-index must be in [0, {args.total_partitions}), got {args.partition_index}"
        )
        return 1

    batch_name = args.input_dir.name
    coreg_output_dir = args.output_dir / f"MR-RATE-coreg_{batch_name}" / "mri" / batch_name
    atlas_output_dir = args.output_dir / f"MR-RATE-atlas_{batch_name}" / "mri" / batch_name
    coreg_output_dir.mkdir(parents=True, exist_ok=True)
    atlas_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("=" * 60)
        logger.info("MR-RATE Registration Block")
        logger.info("=" * 60)
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Metadata CSV: {args.metadata_csv}")
        logger.info(f"Coreg output directory: {coreg_output_dir}")
        logger.info(f"Atlas output directory: {atlas_output_dir}")
        # Create unique temporary directory for this run
        tmp_base = args.tmp_base_dir if args.tmp_base_dir is not None else Path("ants_tmp")
        tmp_dir = tmp_base / f"registration_{uuid.uuid4().hex[:12]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory: {tmp_dir}")
        logger.info(f"Parallel processes: {args.num_processes}")
        logger.info(f"Threads per process: {args.threads_per_process}")
        if args.total_partitions > 1:
            logger.info(f"Partition: {args.partition_index} of {args.total_partitions}")
        logger.info("")
        
        # Validate inputs
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1
        
        if not args.metadata_csv.exists():
            logger.error(f"Metadata CSV not found: {args.metadata_csv}")
            return 1
        
        # Load metadata
        study_data = load_metadata_csv(args.metadata_csv, logger)

        if not study_data:
            logger.error("No studies found in metadata CSV")
            return 1

        # Apply partitioning on full study list before filtering
        # This ensures deterministic splits regardless of processing state
        if args.total_partitions > 1:
            all_study_ids = sorted(study_data.keys())
            partition_study_ids = set(
                s for idx, s in enumerate(all_study_ids)
                if idx % args.total_partitions == args.partition_index
            )
            study_data = {k: v for k, v in study_data.items() if k in partition_study_ids}
            logger.info(
                f"Partition {args.partition_index}/{args.total_partitions}: "
                f"{len(study_data)} studies assigned to this partition"
            )

        # Check for already processed studies
        logger.info("")
        logger.info("Checking for already processed studies...")
        to_process, skipped_complete, skipped_missing, cleaned = check_already_processed(
            args.input_dir,
            coreg_output_dir,
            atlas_output_dir,
            study_data,
            logger,
        )

        logger.info(f"  Studies already processed (skipped): {skipped_complete}")
        logger.info(f"  Studies missing input (skipped): {skipped_missing}")
        if cleaned > 0:
            logger.info(f"  Incomplete outputs cleaned: {cleaned}")
        logger.info(f"  Studies to process: {len(to_process)}")

        if not to_process:
            logger.info("")
            logger.info("No studies to process. All done!")
            return 0
        
        # Fetch MNI152 atlas
        atlas_path = fetch_atlas(logger=logger)
        
        # Prepare arguments for each study
        study_args = [
            (
                study_id,
                study_data[study_id],
                args.input_dir,
                coreg_output_dir,
                atlas_output_dir,
                atlas_path,
                tmp_dir,
            )
            for study_id in to_process
        ]
        
        # Process studies with progress bar
        logger.info("")
        logger.info(f"Starting processing with {args.num_processes} process(es)...")
        total_start_time = time.time()
        
        results = []
        successful = 0
        failed = 0
        
        if args.num_processes == 1:
            # Single process mode with tqdm progress bar
            # Set thread limits for single process mode
            os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads_per_process)
            os.environ["ANTS_RANDOM_SEED"] = "42"
            os.environ["OMP_NUM_THREADS"] = str(args.threads_per_process)
            os.environ["MKL_NUM_THREADS"] = str(args.threads_per_process)
            
            with tqdm(total=len(study_args), desc="Processing studies", unit="study") as pbar:
                for study_arg in study_args:
                    result = process_study(*study_arg)
                    results.append(result)
                    
                    study_id, success, msg = result
                    
                    # Update counters and progress bar
                    if success:
                        successful += 1
                        logger.info(f"{study_id}: {msg}")
                    else:
                        failed += 1
                        logger.error(f"{study_id}: {msg}")
                    
                    pbar.set_postfix({"OK": successful, "FAIL": failed})
                    pbar.update(1)
        else:
            # Multi-process mode with Pool and tqdm
            with Pool(
                processes=args.num_processes,
                initializer=worker_init,
                initargs=(args.threads_per_process, log_queue)
            ) as pool:
                # Use imap_unordered for streaming results with progress bar
                with tqdm(total=len(study_args), desc="Processing studies", unit="study") as pbar:
                    for result in pool.imap_unordered(worker_process_study, study_args):
                        results.append(result)
                        
                        study_id, success, msg = result
                        
                        # Update counters and progress bar
                        if success:
                            successful += 1
                        else:
                            failed += 1
                        
                        pbar.set_postfix({"OK": successful, "FAIL": failed})
                        pbar.update(1)
        
        total_elapsed = time.time() - total_start_time
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info("=" * 60)
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        
        if successful > 0:
            avg_time = total_elapsed / len(to_process)
            logger.info(f"Average time per study: {avg_time:.1f}s")
        
        # Log failed studies
        if failed > 0:
            logger.info("")
            logger.info("Failed studies:")
            for study_id, success, msg in results:
                if not success:
                    logger.info(f"  {study_id}: {msg}")
        
        return 0 if failed == 0 else 1
        
    finally:
        # Clean up temporary directory
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        
        # Clean up queue listener
        if queue_listener is not None:
            queue_listener.stop()


if __name__ == "__main__":
    sys.exit(main())