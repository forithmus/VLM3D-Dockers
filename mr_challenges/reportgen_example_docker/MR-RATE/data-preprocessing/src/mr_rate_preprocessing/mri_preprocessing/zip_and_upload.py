"""
MR-RATE MRI Preprocessing Pipeline - Zip & Upload Block

This script copies study folders to a temporary location while replacing the
original study IDs with anonymized UIDs, then zips from that
temp location and uploads to Hugging Face.

Multiple input directories can be provided; studies from all of them are
treated as the same batch and land in the same output folder.

Output directory structure:
    {output_dir}/
        MR-RATE_{batch_id}/
            tmp/                         ← per-study staging (cleaned up after each zip)
            mri/
                {batch_id}/
                    {uid}.zip

Each zip preserves the internal folder structure rooted at uid/:
    uid/
        img/
            {uid}_{modality_id}.nii.gz
        seg/
            ...

--input-dir and --modalities-json are parallel lists: the N-th JSON is used for
studies found in the N-th input directory.

Example Usage:
    python src/mr_rate_preprocessing/mri_preprocessing/zip_and_upload.py \\
        --input-dir data/processed/batch00 \\
        --modalities-json data/interim/batch00/batch00_modalities_to_process.json \\
        --batch-id batch00 \\
        --output-dir data/processed \\
        --num-zip-workers 8 \\
        --log-dir logs/batch00 \\
        --delete-zips \\
        --verbose
"""

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from utils import setup_parallel_logging, BufferedStudyLogger, accession_to_uid


# Global log queue for worker processes (set via Pool initializer)
_worker_log_queue = None


def _worker_init(log_queue) -> None:
    """Initialise worker process with the logging queue."""
    global _worker_log_queue
    _worker_log_queue = log_queue


def check_modality_complete(study_dir: Path, study_id: str, modality_id: str) -> bool:
    """
    Check whether all three expected files for a modality are present:
        img/{study_id}_{modality_id}.nii.gz
        seg/{study_id}_{modality_id}_brain-mask.nii.gz
        seg/{study_id}_{modality_id}_defacing-mask.nii.gz
    """
    img = study_dir / "img" / f"{study_id}_{modality_id}.nii.gz"
    brain_mask = study_dir / "seg" / f"{study_id}_{modality_id}_brain-mask.nii.gz"
    defacing_mask = study_dir / "seg" / f"{study_id}_{modality_id}_defacing-mask.nii.gz"
    return img.exists() and brain_mask.exists() and defacing_mask.exists()


def check_study_complete(study_dir: Path, study_id: str, study_data: Dict, logger) -> bool:
    """
    Return True only if all modalities (center and moving) are complete.
    Any incomplete modality causes the entire study to be skipped.
    """
    center_modality = study_data.get("center_modality", {})
    if not center_modality:
        logger.warning(f"{study_id}: no center_modality defined in JSON — skipping")
        return False

    center_id = list(center_modality.keys())[0]
    if not check_modality_complete(study_dir, study_id, center_id):
        logger.warning(f"{study_id}: center modality '{center_id}' incomplete — skipping")
        return False

    for mod_id in study_data.get("moving_modality", {}):
        if not check_modality_complete(study_dir, study_id, mod_id):
            logger.warning(f"{study_id}: moving modality '{mod_id}' incomplete — skipping")
            return False

    return True


def _zip_study(args: Tuple) -> Tuple[str, bool, str, float]:
    """
    Copy a study folder to a temporary staging directory (renaming study_id ->
    uid in both the folder name and all filenames), zip from there, then remove
    the staging copy.

    Args:
        args: (study_id, uid, study_dir_str, tmp_study_dir_str, zip_path_str)

    Returns:
        (study_id, success, error_msg, elapsed_seconds)
    """
    study_id, uid, study_dir_str, tmp_study_dir_str, zip_path_str = args
    study_dir = Path(study_dir_str)
    tmp_study_dir = Path(tmp_study_dir_str)
    zip_path = Path(zip_path_str)

    logger = BufferedStudyLogger(_worker_log_queue, study_id)
    start = time.time()

    try:
        # --- Stage: copy with renamed study_id -> uid ---
        tmp_study_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(study_dir.rglob("*")):
            if src.is_file():
                rel = src.relative_to(study_dir)
                new_name = rel.name.replace(study_id, uid, 1)
                dest = tmp_study_dir / rel.parent / new_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

        # --- Zip from staging dir, rooted at uid/ ---
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for file_path in sorted(tmp_study_dir.rglob("*")):
                if file_path.is_file():
                    arcname = Path(uid) / file_path.relative_to(tmp_study_dir)
                    zf.write(file_path, arcname)

        elapsed = time.time() - start
        logger.info(f"Zipped {study_id} -> {uid} -> {zip_path} ({elapsed:.1f}s)")
        logger.flush()
        return study_id, True, "", elapsed

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Failed to zip {study_id}: {e}")
        logger.flush()
        if zip_path.exists():
            zip_path.unlink()
        return study_id, False, str(e), elapsed

    finally:
        # Always remove the staging copy to keep disk usage bounded
        if tmp_study_dir.exists():
            shutil.rmtree(tmp_study_dir, ignore_errors=True)


def collect_studies(
    input_dirs: List[Path],
    modalities_data_list: List[Dict],
    zip_dir: Path,
    logger,
) -> Tuple[List[Path], Dict[str, str], int, int]:
    """
    Discover all study_id/ directories across all input_dirs and derive a
    anonymized UID for each one.

    input_dirs and modalities_data_list are parallel: the N-th modalities dict
    is used exclusively for studies found in the N-th input directory.

    Skips a study if:
    - Its uid-named .zip already exists in zip_dir (already done), or
    - The study's any modality is incomplete per its paired modalities dict.

    If the same study_id appears in more than one input directory, the process
    is stopped with an error.

    Returns:
        (list of study_dirs to process, uid_map {study_id: uid},
         skipped_already_done, skipped_incomplete)
    """
    to_zip: List[Path] = []
    uid_map: Dict[str, str] = {}
    skipped_done = 0
    skipped_incomplete = 0
    seen_study_ids: Dict[str, Path] = {}
    total_found = 0

    for input_dir, modalities_data in zip(input_dirs, modalities_data_list):
        all_study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        total_found += len(all_study_dirs)

        for study_dir in all_study_dirs:
            study_id = study_dir.name

            if study_id in seen_study_ids:
                error_msg = (
                    f"{study_id}: duplicate — already seen in "
                    f"{seen_study_ids[study_id]}, stopping process"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            seen_study_ids[study_id] = input_dir

            uid = accession_to_uid(study_id)
            uid_map[study_id] = uid
            zip_path = zip_dir / f"{uid}.zip"

            if zip_path.exists():
                skipped_done += 1
                continue

            if study_id not in modalities_data:
                logger.warning(f"{study_id}: not found in modalities JSON — skipping")
                skipped_incomplete += 1
                continue
            if not check_study_complete(study_dir, study_id, modalities_data[study_id], logger):
                skipped_incomplete += 1
                continue

            to_zip.append(study_dir)

    logger.info(
        f"Studies found: {total_found} total across {len(input_dirs)} input dir(s) | "
        f"{len(to_zip)} to zip | "
        f"{skipped_done} already done (skipped) | "
        f"{skipped_incomplete} incomplete/duplicate (skipped)"
    )
    return to_zip, uid_map, skipped_done, skipped_incomplete


def upload_to_hf(batch_folder: Path, repo_id: str, num_workers: int, logger, delete_after: bool = False) -> None:
    """
    Validate and upload batch_folder to Hugging Face.

    batch_folder must contain only an mri/ subdirectory (no hidden dirs).
    If delete_after is True, batch_folder is removed once the upload finishes.
    """
    from huggingface_hub import upload_large_folder

    subdirs = [
        d for d in batch_folder.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    if len(subdirs) != 1 or subdirs[0].name != "mri":
        logger.error(
            f"Upload aborted: expected exactly one subdirectory 'mri/' inside "
            f"{batch_folder}, found {[d.name for d in subdirs]}"
        )
        sys.exit(1)

    logger.info(f"Uploading {batch_folder} to {repo_id} with {num_workers} workers...")
    upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(batch_folder),
        num_workers=num_workers,
    )
    logger.info("Upload complete.")

    if delete_after:
        logger.info(f"--delete-zips set; removing {batch_folder}...")
        shutil.rmtree(batch_folder, ignore_errors=True)
        logger.info(f"Deleted {batch_folder}.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zip study folders and upload to Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Zip and upload
    python src/mr_rate_preprocessing/mri_preprocessing/zip_and_upload.py \\
        --input-dir data/processed/batch00 \\
        --modalities-json data/interim/batch00/batch00_modalities_to_process.json \\
        --batch-id batch00 \\
        --output-dir data/processed \\
        --num-zip-workers 8 \\
        --log-dir logs/batch00 \\
        --delete-zips \\
        --verbose
        """
    )
    parser.add_argument(
        "-m", "--modalities-json",
        type=str,
        nargs="+",
        required=True,
        metavar="JSON",
        help=(
            "One or more paths to modalities JSON files (output of modality_filtering.py). "
            "Must be supplied in the same order as --input-dir: the N-th JSON is used for "
            "studies in the N-th input directory. "
            "Studies where any modality (center or moving) is incomplete are skipped."
        ),
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            "One or more directories containing study_id/ subfolders to zip. "
            "Studies from all directories are merged into the same batch."
        ),
    )
    parser.add_argument(
        "-b", "--batch-id",
        type=str,
        required=True,
        help="Batch identifier, e.g. batch00",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Root output directory; zips land at <output-dir>/MR-RATE_<batch-id>/mri/<batch-id>/",
    )
    parser.add_argument(
        "-n", "--num-zip-workers",
        type=int,
        default=4,
        help="Number of parallel zipping workers (default: 4)",
    )
    parser.add_argument(
        "-w", "--num-hf-workers",
        type=int,
        default=16,
        help="Number of parallel upload workers for Hugging Face (default: 16)",
    )
    parser.add_argument(
        "-l", "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Also log to console",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Zip only; skip Hugging Face upload",
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete the batch folder after a successful upload to Hugging Face",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face dataset repo ID to upload to",
    )
    parser.add_argument(
        "--hf-cache",
        type=str,
        default=None,
        help=(
            "Override HF_XET_CACHE for this run (e.g. /scratch/$USER/hf_xet). "
            "Defaults to None (uses the existing HF_XET_CACHE environment variable)."
        ),
    )
    parser.add_argument(
        "--hf-timeout",
        type=int,
        default=120,
        help="Override HF_HUB_DOWNLOAD_TIMEOUT for this run in seconds (default: 120)",
    )
    parser.add_argument(
        "--xet-high-perf",
        action="store_true",
        default=False,
        help=(
            "Set HF_XET_HIGH_PERFORMANCE=1 for this run. "
            "Uses all available CPUs and maximum network bandwidth — "
            "system responsiveness may degrade."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Apply HF environment overrides before any huggingface_hub imports
    if args.hf_cache is not None:
        os.environ["HF_XET_CACHE"] = args.hf_cache
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_timeout)
    if args.xet_high_perf:
        print(
            "\nWARNING: --xet-high-perf is enabled.\n"
            "  HF_XET_HIGH_PERFORMANCE=1 will be set for this session.\n"
            "  This uses all available CPUs and maximum network bandwidth.\n"
            "  System responsiveness may degrade.\n"
        )
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    input_dirs = [Path(p) for p in args.input_dir]
    modalities_json_paths = [Path(p) for p in args.modalities_json]
    output_dir = Path(args.output_dir)
    log_dir = args.log_dir
    batch_id = args.batch_id

    # Validate parallel list lengths
    if len(input_dirs) != len(modalities_json_paths):
        print(
            f"Error: --input-dir and --modalities-json must have the same number of values "
            f"({len(input_dirs)} input dir(s) vs {len(modalities_json_paths)} JSON(s))."
        )
        sys.exit(1)

    # Validate all input directories
    for input_dir in input_dirs:
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Error: input directory does not exist: {input_dir}")
            sys.exit(1)

    # Load all modalities JSONs
    modalities_data_list: List[Dict] = []
    for json_path in modalities_json_paths:
        if not json_path.exists():
            print(f"Error: modalities JSON does not exist: {json_path}")
            sys.exit(1)
        with open(json_path) as f:
            modalities_data_list.append(json.load(f))

    # Build paths
    batch_folder = output_dir / f"MR-RATE_{batch_id}"
    zip_dir = batch_folder / "mri" / batch_id
    tmp_dir = batch_folder / "tmp"
    zip_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up temporary directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logger, log_queue, queue_listener = setup_parallel_logging(
        log_dir=log_dir,
        script_name="zip_and_upload" + "_" + batch_id,
        verbose=args.verbose,
    )

    for i, (input_dir, json_path) in enumerate(zip(input_dirs, modalities_json_paths)):
        logger.info(f"Input dir [{i}]    : {input_dir}")
        logger.info(f"Modalities JSON [{i}]: {json_path}")
    logger.info(f"Batch ID        : {batch_id}")
    logger.info(f"Output dir      : {output_dir}")
    logger.info(f"Zip dir         : {zip_dir}")
    logger.info(f"Tmp dir         : {tmp_dir}")
    logger.info(f"Zip workers     : {args.num_zip_workers}")
    logger.info(f"Skip upload     : {args.skip_upload}")
    logger.info(f"HF repo ID      : {args.repo_id}")
    logger.info(f"HF_XET_CACHE    : {os.environ.get('HF_XET_CACHE', '(default)')}")
    logger.info(f"HF timeout (s)  : {args.hf_timeout}")
    logger.info(f"XET high perf   : {args.xet_high_perf}")

    # Discover studies and compute uid_map
    to_zip, uid_map, skipped_done, skipped_incomplete = collect_studies(
        input_dirs, modalities_data_list, zip_dir, logger
    )

    if not to_zip:
        logger.info(f"Nothing to zip ({skipped_done} already done, {skipped_incomplete} incomplete). Exiting.")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if not args.skip_upload:
            upload_to_hf(batch_folder, args.repo_id, args.num_hf_workers, logger, delete_after=args.delete_zips)
        queue_listener.stop()
        return

    # Build worker args (queue is passed via initializer, not in args)
    worker_args = [
        (
            study_dir.name,                              # study_id
            uid_map[study_dir.name],                     # uid
            str(study_dir),                              # study_dir_str
            str(tmp_dir / uid_map[study_dir.name]),      # tmp_study_dir_str
            str(zip_dir / f"{uid_map[study_dir.name]}.zip"),  # zip_path_str
        )
        for study_dir in to_zip
    ]

    # Parallel zip
    total = len(worker_args)
    success_count = 0
    fail_count = 0

    logger.info(f"Starting parallel zipping of {total} studies with {args.num_zip_workers} workers...")

    with Pool(processes=args.num_zip_workers, initializer=_worker_init, initargs=(log_queue,)) as pool:
        for study_id, success, error_msg, elapsed in tqdm(
            pool.imap_unordered(_zip_study, worker_args),
            total=total,
            desc="Zipping studies",
            unit="study",
        ):
            if success:
                success_count += 1
            else:
                fail_count += 1
                logger.error(f"FAILED [{study_id}]: {error_msg}")

    logger.info(
        f"Zipping complete: {success_count} succeeded, {fail_count} failed, "
        f"{skipped_done} already done, {skipped_incomplete} incomplete"
    )

    # Clean up temporary directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if fail_count > 0:
        logger.error(f"{fail_count} studies failed to zip. Check logs for details.")
        queue_listener.stop()
        return

    # Upload
    if args.skip_upload:
        logger.info("--skip-upload set; skipping Hugging Face upload.")
    else:
        logger.info("Uploading to Hugging Face...")
        upload_to_hf(batch_folder, args.repo_id, args.num_hf_workers, logger, delete_after=args.delete_zips)

    queue_listener.stop()
    return

if __name__ == "__main__":
    main()
