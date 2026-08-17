"""
MR-RATE MRI Preprocessing Pipeline - Brain Segmentation and Defacing Block

This script processes all modalities of a study in their native space:
1. Brain segmentation using HD-BET for each modality (center + moving)
2. Defacing mask generation using Quickshear
3. Defacing mask application to the native images

Output directory structure:
    output_dir/
        study_id/
            img/
                {study_id}_{modality_id}.nii.gz               # Defaced native images (np.uint16 or np.float32)
            seg/
                {study_id}_{modality_id}_brain_mask.nii.gz    # Brain masks (np.uint8)
                {study_id}_{modality_id}_defacing_mask.nii.gz # Defacing masks (np.uint8)

Supports parallel GPU processing with multiple GPUs.

Example Usage:
    python src/mr_rate_preprocessing/mri_preprocessing/brain_segmentation_and_defacing.py \\
        --modalities-json data/interim/batch00/batch00_modalities_to_process.json \\
        --raw-dir data/raw/batch00/batch00_raw_niftis \\
        --output-dir data/processed/batch00/ \\
        --device 0 \\
        --log-dir logs/batch00 \\
        --verbose


HD-BET, Quickshear and BrainLesion Suite Attribution
----------------------------------------------------
This script utilizes HD-BET for brain segmentation and Quickshear for defacing, 
both of which are implemented as part of the BrainLesion Suite.
For complete attribution, please refer to the hdbet.py and quickshear.py files.

If you are using this script, please cite:

Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A,
Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P.
Automated brain extraction of multi-sequence MRI using artificial neural networks.
Hum Brain Mapp. 2019; 1-13. https://doi.org/10.1002/hbm.24750

Schimke, Nakeisha, and John Hale. "Quickshear defacing for neuroimages." Proceedings 
of the 2nd USENIX conference on Health security and privacy. USENIX Association, 2011.

Kofler, F., Rosier, M., Astaraki, M., Möller, H., Mekki, I. I., Buchner, J. A., Schmick, 
A., Pfiffer, A., Oswald, E., Zimmer, L., Rosa, E. de la, Pati, S., Canisius, J., Piffer, 
A., Baid, U., Valizadeh, M., Linardos, A., Peeken, J. C., Shit, S., … Menze, B. (2025). 
BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image 
Analysis https://arxiv.org/abs/2507.09036
"""

import argparse
import json
import shutil
import sys
import time
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from utils import setup_logging
from hdbet import BrainSegmentor, parse_devices
from quickshear import generate_defacing_mask, apply_defacing
from mr_rate_preprocessing.configs.config_mri_preprocessing import (
    HDBET_MODE,
    HDBET_DO_TTA,
    HDBET_POSTPROCESS,
    HDBET_COMPILE,
    HDBET_MIXED_PREC,
)


def check_already_processed(
    raw_dir: Path,
    output_dir: Path,
    modalities_data: Dict,
    logger,
) -> Tuple[List[str], int, int, int]:
    """
    Check which studies need processing, skip completed ones, clean incomplete ones.
    
    A study is considered complete if ALL modalities have their defaced images present.
    
    A study is skipped (missing input) if:
    - Study folder missing in raw-dir
    
    Args:
        raw_dir: Raw data directory
        output_dir: Base output directory
        modalities_data: Dictionary from modalities JSON
        logger: Logger instance
        
    Returns:
        Tuple of (list of study_ids to process, skipped_complete, skipped_missing_input, cleaned)
    """
    to_process = []
    skipped_complete = 0
    skipped_missing_input = 0
    cleaned_count = 0
    
    for study_id, data in modalities_data.items():
        study_output = output_dir / study_id
        study_raw = raw_dir / study_id
        img_dir = study_output / "img"
        
        # Check if input directory exists
        if not study_raw.exists():
            skipped_missing_input += 1
            continue
        
        # Get all modality IDs (center + moving)
        center_id = list(data["center_modality"].keys())[0]
        moving_ids = list(data.get("moving_modality", {}).keys())
        all_modality_ids = [center_id] + moving_ids
        
        # Check if ALL defaced images exist
        all_outputs_exist = all(
            (img_dir / f"{study_id}_{mod_id}.nii.gz").exists()
            for mod_id in all_modality_ids
        )
        
        if all_outputs_exist:
            # Already processed successfully
            skipped_complete += 1
        elif study_output.exists():
            # Incomplete processing - clean up and reprocess
            logger.info(f"Cleaning incomplete output for {study_id}")
            shutil.rmtree(study_output)
            cleaned_count += 1
            to_process.append(study_id)
        else:
            # Not processed yet
            to_process.append(study_id)
    
    return to_process, skipped_complete, skipped_missing_input, cleaned_count


def process_modality(
    study_id: str,
    modality_id: str,
    img_path: str,
    raw_dir: Path,
    output_dir: Path,
    brain_segmentor: BrainSegmentor,
) -> Tuple[str, str, bool, str, float]:
    """
    Process a single modality: brain segmentation -> defacing mask -> defacing.
    
    Args:
        study_id: Study identifier
        modality_id: Modality identifier
        img_path: Relative path to image within study folder
        raw_dir: Raw data directory
        output_dir: Output directory
        brain_segmentor: Initialized BrainSegmentor instance
        
    Returns:
        Tuple of (study_id, modality_id, success, error_msg, elapsed)
    """
    start_time = time.time()
    
    try:
        # Define paths
        input_path = raw_dir / study_id / img_path
        study_output = output_dir / study_id
        img_out_dir = study_output / "img"
        seg_out_dir = study_output / "seg"

        brain_mask_path = seg_out_dir / f"{study_id}_{modality_id}_brain-mask.nii.gz"
        defacing_mask_path = seg_out_dir / f"{study_id}_{modality_id}_defacing-mask.nii.gz"
        defaced_img_path = img_out_dir / f"{study_id}_{modality_id}.nii.gz"
        
        # Create output directories
        img_out_dir.mkdir(parents=True, exist_ok=True)
        seg_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Brain segmentation
        brain_segmentor(input_path, brain_mask_path)
        
        # Step 2: Generate defacing mask
        success = generate_defacing_mask(
            brain_mask_path=brain_mask_path,
            defacing_mask_path=defacing_mask_path,
            logger=None,
        )
        if not success:
            raise Exception("Failed to generate defacing mask")
        
        # Step 3: Apply defacing
        success = apply_defacing(
            input_path=input_path,
            defacing_mask_path=defacing_mask_path,
            output_path=defaced_img_path,
            logger=None,
        )
        if not success:
            raise Exception("Failed to apply defacing")
        
        elapsed = time.time() - start_time
        return (study_id, modality_id, True, None, elapsed)
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        return (study_id, modality_id, False, error_msg, elapsed)


def gpu_worker(
    gpu_id: int,
    tasks: List[Tuple[str, str, str]],  # List of (study_id, modality_id, img_path)
    raw_dir: Path,
    output_dir: Path,
    mode: str,
    do_tta: bool,
    postprocess: bool,
    compile_model: bool,
    mixed_prec: bool,
    result_queue: Queue,
):
    """
    Worker function for processing modalities on a single GPU.
    
    Each worker initializes its own BrainSegmentor on the assigned GPU
    and processes its share of modalities. Results are sent to the queue.
    
    Args:
        gpu_id: GPU device ID for this worker
        tasks: List of (study_id, modality_id, img_path) tuples to process
        raw_dir: Raw data directory
        output_dir: Output directory
        mode: 'fast' or 'accurate'
        do_tta: Enable test-time augmentation
        postprocess: Keep only largest connected component
        compile_model: Enable torch.compile()
        mixed_prec: Enable mixed precision
        result_queue: Queue to send results back to main process
    """
    # Set GPU device
    torch.cuda.set_device(gpu_id)
    
    # Initialize BrainSegmentor for this GPU
    brain_segmentor = BrainSegmentor(
        mode=mode,
        device=gpu_id,
        do_tta=do_tta,
        postprocess=postprocess,
        compile=compile_model,
        mixed_prec=mixed_prec,
    )
    
    # Process assigned tasks
    for study_id, modality_id, img_path in tasks:
        result = process_modality(
            study_id=study_id,
            modality_id=modality_id,
            img_path=img_path,
            raw_dir=raw_dir,
            output_dir=output_dir,
            brain_segmentor=brain_segmentor,
        )
        study_id_res, modality_id_res, success, error_msg, elapsed = result
        result_queue.put((study_id_res, modality_id_res, success, error_msg, elapsed, gpu_id))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MR-RATE Segmentation + Defacing Block - Alternative to registration-based pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Multi-GPU processing (2 GPUs in parallel)
    python src/mr_rate_preprocessing/mri_preprocessing/brain_segmentation_and_defacing.py \\
        --modalities-json data/interim/batch00/batch00_modalities_to_process.json \\
        --raw-dir data/raw/batch00/batch00_raw_niftis \\
        --output-dir data/processed/batch00/ \\
        --device 0,1 \\
        --verbose
        """
    )
    
    parser.add_argument(
        "--modalities-json", "-m",
        type=Path,
        required=True,
        help="Path to modalities JSON file (output of modality_filtering.py)"
    )
    
    parser.add_argument(
        "--raw-dir", "-r",
        type=Path,
        required=True,
        help="Path to raw MR data directory (contains study folders with NIfTI files)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for segmented and defaced images"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="0",
        help="Device(s): GPU IDs comma-separated (e.g., '0', '0,1,2') or 'cpu' (default: '0')"
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
    """Main entry point for the segmentation + defacing pipeline."""
    args = parse_args()
    
    script_name = Path(__file__).stem
    
    # Setup logging
    logger = setup_logging(args.log_dir, script_name, verbose=args.verbose)
    
    # Parse devices
    try:
        devices = parse_devices(args.device)
    except ValueError:
        logger.error(f"Invalid device specification: {args.device}")
        return 1
    
    num_devices = len(devices)
    is_multi_gpu = num_devices > 1 and devices[0] != "cpu"
    
    logger.info("=" * 60)
    logger.info("MR-RATE Image Processing Pipeline - Brain Segmentation & Defacing Block")
    logger.info("=" * 60)
    logger.info(f"Modalities JSON: {args.modalities_json}")
    logger.info(f"Raw data directory: {args.raw_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device(s): {args.device} ({num_devices} device{'s' if num_devices > 1 else ''})")
    logger.info(f"Mode: {HDBET_MODE}")
    logger.info(f"Test-time augmentation: {HDBET_DO_TTA}")
    logger.info(f"Postprocessing: {HDBET_POSTPROCESS}")
    logger.info(f"Compile: {HDBET_COMPILE}")
    logger.info(f"Mixed precision (BF16): {HDBET_MIXED_PREC}")
    logger.info("")
    
    # Validate inputs
    if not args.modalities_json.exists():
        logger.error(f"Modalities JSON not found: {args.modalities_json}")
        return 1
    
    if not args.raw_dir.exists():
        logger.error(f"Raw data directory not found: {args.raw_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load modalities JSON
    logger.info("Loading modalities data...")
    with open(args.modalities_json, 'r') as f:
        modalities_data = json.load(f)
    
    logger.info(f"Found {len(modalities_data)} studies in JSON")
    
    # Check for already processed studies
    logger.info("")
    logger.info("Checking for already processed studies...")
    to_process, skipped_complete, skipped_missing, cleaned = check_already_processed(
        args.raw_dir, args.output_dir, modalities_data, logger
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
    
    # Flatten tasks: convert studies to individual (study_id, modality_id, img_path) tuples
    tasks = []
    for study_id in to_process:
        study_data = modalities_data[study_id]
        
        # Add center modality
        center_id = list(study_data["center_modality"].keys())[0]
        center_info = study_data["center_modality"][center_id]
        tasks.append((study_id, center_id, center_info["img_path"]))
        
        # Add moving modalities
        for mod_id, mod_info in study_data.get("moving_modality", {}).items():
            tasks.append((study_id, mod_id, mod_info["img_path"]))
    
    logger.info(f"  Total modalities to process: {len(tasks)}")
    
    total_start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    if is_multi_gpu:
        # Multi-GPU processing with multiprocessing
        logger.info("")
        logger.info(f"Starting multi-GPU processing with {num_devices} GPUs...")
        
        # Set start method to spawn
        mp.set_start_method("spawn", force=True)
        
        # Split tasks among GPUs using round-robin
        chunks = [tasks[i::num_devices] for i in range(num_devices)]
        for i, (gpu_id, chunk) in enumerate(zip(devices, chunks)):
            logger.info(f"  GPU {gpu_id}: {len(chunk)} modalities")
        
        # Create result queue and spawn workers
        result_queue = Queue()
        processes = []
        
        for gpu_id, chunk in zip(devices, chunks):
            if len(chunk) == 0:
                continue
            p = Process(
                target=gpu_worker,
                args=(
                    gpu_id,
                    chunk,
                    args.raw_dir,
                    args.output_dir,
                    HDBET_MODE,
                    HDBET_DO_TTA,
                    HDBET_POSTPROCESS,
                    HDBET_COMPILE,
                    HDBET_MIXED_PREC,
                    result_queue,
                )
            )
            p.start()
            processes.append(p)
        
        # Collect results with progress bar
        logger.info("")
        with tqdm(total=len(tasks), desc="Processing modalities", unit="mod") as pbar:
            collected = 0
            while collected < len(tasks):
                result = result_queue.get()
                study_id, modality_id, success, error_msg, elapsed, gpu_id = result
                results.append(result)
                collected += 1
                
                if success:
                    successful += 1
                    logger.info(f"GPU{gpu_id} - {study_id}/{modality_id} completed in {elapsed:.1f}s")
                else:
                    failed += 1
                    logger.error(f"GPU{gpu_id} - {study_id}/{modality_id} failed after {elapsed:.1f}s: {error_msg}")
                
                pbar.set_postfix({"OK": successful, "FAIL": failed})
                pbar.update(1)
        
        # Wait for all workers to finish
        for p in processes:
            p.join()
    
    else:
        # Single GPU/CPU processing (sequential)
        device = devices[0]
        
        logger.info("")
        logger.info("Initializing BrainSegmentor...")
        logger.info(f"  Loading {5 if HDBET_MODE == 'accurate' else 1} model(s) on {device}...")
        
        brain_segmentor = BrainSegmentor(
            mode=HDBET_MODE,
            device=device,
            do_tta=HDBET_DO_TTA,
            postprocess=HDBET_POSTPROCESS,
            compile=HDBET_COMPILE,
            mixed_prec=HDBET_MIXED_PREC,
        )
        
        logger.info("  Models loaded successfully")
        
        # Process modalities
        logger.info("")
        logger.info(f"Processing {len(tasks)} modalities...")
        
        with tqdm(total=len(tasks), desc="Processing modalities", unit="mod") as pbar:
            for study_id, modality_id, img_path in tasks:
                result = process_modality(
                    study_id=study_id,
                    modality_id=modality_id,
                    img_path=img_path,
                    raw_dir=args.raw_dir,
                    output_dir=args.output_dir,
                    brain_segmentor=brain_segmentor,
                )
                study_id_res, modality_id_res, success, error_msg, elapsed = result
                results.append(result)
                
                if success:
                    successful += 1
                    logger.info(f"{study_id_res}/{modality_id_res} completed in {elapsed:.1f}s")
                else:
                    failed += 1
                    logger.error(f"{study_id_res}/{modality_id_res} failed after {elapsed:.1f}s: {error_msg}")
                
                pbar.set_postfix({"OK": successful, "FAIL": failed})
                pbar.update(1)
    
    total_elapsed = time.time() - total_start_time
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Successful modalities: {successful}")
    logger.info(f"Failed modalities: {failed}")
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    if successful > 0:
        avg_time = total_elapsed / len(tasks)
        logger.info(f"Average time per modality: {avg_time:.1f}s")
        if is_multi_gpu:
            logger.info(f"Effective throughput: {len(tasks) / total_elapsed * 60:.1f} modalities/min")
    
    # Log failed modalities
    if failed > 0:
        logger.info("")
        logger.info("Failed modalities:")
        for result in results:
            if is_multi_gpu:
                study_id, modality_id, success, error_msg, elapsed, gpu_id = result
            else:
                study_id, modality_id, success, error_msg, elapsed = result
            if not success:
                logger.info(f"  {study_id}/{modality_id}: {error_msg}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())