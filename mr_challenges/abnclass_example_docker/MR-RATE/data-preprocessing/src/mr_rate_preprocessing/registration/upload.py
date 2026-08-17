"""
Zip registration output studies and upload to a Hugging Face dataset.

Study directories found under <input-dir>/mri/<batch-subdir>/<study-uid>/ are
zipped in parallel and written to a mirror folder next to the input directory:

    <input-dir>_zipped/
        mri/
            <batch-subdir>/
                <study-uid><zip-suffix>.zip

Each zip preserves the internal folder structure rooted at <study-uid>/:

    <study-uid>/
        coreg_img/
        coreg_seg/
        transform/
    or
    
    <study-uid>/
        atlas_img/
        atlas_seg/
        transform/
    
    or

    <study-uid>/
        seg/

Usage:
    python upload.py \\
        --input-dir data/MR-RATE-reg/MR-RATE-coreg_batchXX \\
        --zip-suffix _coreg \\
        --repo-id Forithmus/MR-RATE-coreg \\
        [--zip-workers 4] \\
        [--hf-workers 16] \\
        [--xet-high-perf] \\
        [--delete-zips] \\
        [--hf-xet-cache /scratch/hf_xet] \\
        [--hf-read-timeout 120] \\
        [--log-dir logs] \\
        [--verbose]
"""

import os
import shutil
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from mr_rate_preprocessing.mri_preprocessing.utils import setup_parallel_logging, BufferedStudyLogger


# Global log queue injected into worker processes via Pool initializer
_worker_log_queue = None


def _worker_init(log_queue) -> None:
    global _worker_log_queue
    _worker_log_queue = log_queue


def _zip_study(args: Tuple) -> Tuple[str, bool, str, float]:
    """
    Zip a single study directory to zip_path, rooted at <study_uid>/.

    Args:
        args: (study_uid, study_dir_str, zip_path_str)

    Returns:
        (study_uid, success, error_msg, elapsed_seconds)
    """
    study_uid, study_dir_str, zip_path_str = args
    study_dir = Path(study_dir_str)
    zip_path = Path(zip_path_str)

    logger = BufferedStudyLogger(_worker_log_queue, study_uid)
    start = time.time()

    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for file_path in sorted(study_dir.rglob("*")):
                if file_path.is_file():
                    arcname = Path(study_uid) / file_path.relative_to(study_dir)
                    zf.write(file_path, arcname)

        elapsed = time.time() - start
        logger.info(f"Zipped {study_uid} -> {zip_path} ({elapsed:.1f}s)")
        logger.flush()
        return study_uid, True, "", elapsed

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Failed to zip {study_uid}: {e}")
        logger.flush()
        if zip_path.exists():
            zip_path.unlink()
        return study_uid, False, str(e), elapsed


def collect_studies(
    input_dir: Path,
    output_dir: Path,
    zip_suffix: str,
    logger,
) -> Tuple[List[Path], int]:
    """
    Discover study directories under <input_dir>/mri/<batch_subdir>/<study_uid>/.

    Skips a study if its <study_uid><zip_suffix>.zip already exists in the
    mirrored output location.

    Returns:
        (list of study_dirs to zip, skipped_already_done count)
    """
    mri_dir = input_dir / "mri"
    if not mri_dir.is_dir():
        logger.error(f"Expected 'mri/' subdirectory not found inside {input_dir}")
        sys.exit(1)

    to_zip: List[Path] = []
    skipped_done = 0
    total_found = 0

    batch_subdirs = sorted([d for d in mri_dir.iterdir() if d.is_dir()])
    if not batch_subdirs:
        logger.error(f"No batch subdirectories found under {mri_dir}")
        sys.exit(1)

    for batch_subdir in batch_subdirs:
        study_dirs = sorted([d for d in batch_subdir.iterdir() if d.is_dir()])
        total_found += len(study_dirs)

        for study_dir in study_dirs:
            study_uid = study_dir.name
            rel = study_dir.parent.relative_to(input_dir)
            zip_path = output_dir / rel / f"{study_uid}{zip_suffix}.zip"

            if zip_path.exists():
                skipped_done += 1
                continue

            to_zip.append(study_dir)

    logger.info(
        f"Studies found: {total_found} total | "
        f"{len(to_zip)} to zip | "
        f"{skipped_done} already done (skipped)"
    )
    return to_zip, skipped_done


def upload_to_hf(
    output_dir: Path,
    repo_id: str,
    hf_workers: int,
    logger,
    delete_after: bool = False,
) -> None:
    """
    Validate output_dir and upload it to Hugging Face via upload_large_folder.

    output_dir must contain only a single 'mri/' subdirectory (hidden dirs
    such as .cache are ignored).
    """
    from huggingface_hub import upload_large_folder

    subdirs = [
        d for d in output_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    if len(subdirs) != 1 or subdirs[0].name != "mri":
        logger.error(
            f"Upload aborted: expected exactly one subdirectory 'mri/' inside "
            f"{output_dir}, found: {[d.name for d in subdirs]}"
        )
        sys.exit(1)

    logger.info(f"Uploading {output_dir} to {repo_id} with {hf_workers} workers...")
    upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        num_workers=hf_workers,
    )
    logger.info("Upload complete.")

    if delete_after:
        logger.info(f"--delete-zips set; removing {output_dir}...")
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.info(f"Deleted {output_dir}.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Zip registration output studies and upload to Hugging Face. "
            "Zip suffix and default repo are inferred from the input dir name "
            "('coreg' or 'atlas')."
        )
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        metavar="DIR",
        help=(
            "Batch folder to process, e.g. data/MR-RATE-reg/MR-RATE-coreg_batchXX. "
            "Must contain mri/<batch-subdir>/<study-uid>/ structure."
        ),
    )
    parser.add_argument(
        "--zip-suffix",
        type=str,
        required=True,
        metavar="SUFFIX",
        help="Suffix appended to each zip filename, e.g. '_coreg' or '_atlas' or '_vista-seg'.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        metavar="REPO",
        help="Hugging Face dataset repo ID to upload to.",
    )
    parser.add_argument(
        "--zip-workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel zipping processes (default: 4)",
    )
    parser.add_argument(
        "--hf-workers",
        type=int,
        default=16,
        metavar="N",
        help="Number of parallel upload workers for Hugging Face (default: 16)",
    )
    parser.add_argument(
        "--xet-high-perf",
        action="store_true",
        default=False,
        help=(
            "Set HF_XET_HIGH_PERFORMANCE=1 to enable the high-performance Xet "
            "transfer backend. WARNING: uses all available CPUs and maximum "
            "bandwidth, which may degrade system responsiveness."
        ),
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete the zipped output folder after a successful upload",
    )
    parser.add_argument(
        "--hf-xet-cache",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override HF_XET_CACHE for this run (e.g. /scratch/$USER/hf_xet). "
            "Defaults to the existing HF_XET_CACHE environment variable."
        ),
    )
    parser.add_argument(
        "--hf-read-timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Override HF_HUB_DOWNLOAD_TIMEOUT for this run in seconds (default: 120)",
    )
    parser.add_argument(
        "-l", "--log-dir",
        type=str,
        default="logs",
        metavar="DIR",
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Also log to console",
    )

    args = parser.parse_args()

    # Apply HF environment overrides before any huggingface_hub imports
    if args.hf_xet_cache is not None:
        os.environ["HF_XET_CACHE"] = args.hf_xet_cache
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_read_timeout)
    if args.xet_high_perf:
        print(
            "\nWARNING: --xet-high-perf is enabled.\n"
            "  HF_XET_HIGH_PERFORMANCE=1 will be set for this session.\n"
            "  This uses all available CPUs and maximum network bandwidth.\n"
            "  System responsiveness may degrade.\n"
        )
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    input_dir = Path(args.input_dir).resolve()
    log_dir = Path(args.log_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    zip_suffix = args.zip_suffix
    repo_id = args.repo_id

    # Derive output dir next to input dir
    output_dir = input_dir.parent / f"{input_dir.name}_zipped"

    # Logging
    logger, log_queue, queue_listener = setup_parallel_logging(
        log_dir=log_dir,
        script_name=f"hf_upload_{input_dir.name}",
        verbose=args.verbose,
    )

    logger.info(f"Input dir         : {input_dir}")
    logger.info(f"Output dir        : {output_dir}")
    logger.info(f"Repo ID           : {repo_id}")
    logger.info(f"Zip suffix        : {zip_suffix}")
    logger.info(f"Zip workers       : {args.zip_workers}")
    logger.info(f"HF workers        : {args.hf_workers}")
    logger.info(f"HF_XET_CACHE      : {os.environ.get('HF_XET_CACHE', '(default)')}")
    logger.info(f"HF read timeout(s): {args.hf_read_timeout}")
    logger.info(f"Xet high-perf     : {'ON' if args.xet_high_perf else 'off'}")
    logger.info(f"Delete after upload: {args.delete_zips}")

    # Discover studies
    to_zip, skipped_done = collect_studies(input_dir, output_dir, zip_suffix, logger)

    if not to_zip:
        logger.info(
            f"Nothing to zip ({skipped_done} already done). Proceeding to upload..."
        )
        upload_to_hf(output_dir, repo_id, args.hf_workers, logger, delete_after=args.delete_zips)
        queue_listener.stop()
        return

    # Build worker args
    worker_args = [
        (
            study_dir.name,  # study_uid
            str(study_dir),
            str(output_dir / study_dir.parent.relative_to(input_dir) / f"{study_dir.name}{zip_suffix}.zip"),
        )
        for study_dir in to_zip
    ]

    # Parallel zip
    total = len(worker_args)
    success_count = 0
    fail_count = 0

    logger.info(f"Starting parallel zipping of {total} studies with {args.zip_workers} processes...")

    with Pool(processes=args.zip_workers, initializer=_worker_init, initargs=(log_queue,)) as pool:
        for study_uid, success, error_msg, elapsed in tqdm(
            pool.imap_unordered(_zip_study, worker_args),
            total=total,
            desc="Zipping studies",
            unit="study",
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),  # disable in non-TTY (HPC/pipe) environments
        ):
            if success:
                success_count += 1
            else:
                fail_count += 1
                logger.error(f"FAILED [{study_uid}]: {error_msg}")

    logger.info(
        f"Zipping complete: {success_count} succeeded, {fail_count} failed, "
        f"{skipped_done} already done"
    )

    if fail_count > 0:
        logger.error(f"{fail_count} studies failed to zip. Check logs for details. Aborting upload.")
        queue_listener.stop()
        sys.exit(1)

    # Upload
    upload_to_hf(output_dir, repo_id, args.hf_workers, logger, delete_after=args.delete_zips)

    queue_listener.stop()


if __name__ == "__main__":
    main()