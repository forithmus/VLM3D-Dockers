"""
MR-RATE MRI Preprocessing Pipeline - DICOM to NIfTI Conversion Block

This script converts DICOM folders to NIfTI format by:
1. Reading a CSV file containing a list of DICOM folder paths
2. Extracting the AccessionNumber from the first DICOM file in each folder
3. Running dcm2niix to produce gzip-compressed NIfTI files and JSON sidecars
4. Writing each study's output directly into a per-accession subfolder under the output directory

Conversion is parallelised across folders using a ProcessPoolExecutor.

Example Usage:
    python src/mr_rate_preprocessing/mri_preprocessing/dcm2nii.py \\
        --input-csv data/raw/batch00/batch00_dicom_folder_paths.csv \\
        --output-dir data/raw/batch00/batch00_raw_niftis \\
        --max-workers 16
"""

import argparse
import os
import pandas as pd
import pydicom
import subprocess
import sys
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import tqdm


def sanitize_filename(filename):
    """
    Sanitizes a string to be used as a valid filename.
    - Replaces spaces and path separators with underscore
    - Removes illegal characters
    - Prevents multiple underscores
    """
    if not isinstance(filename, str):
        filename = str(filename)
    s = re.sub(r'[\\/\s]+', '_', filename)
    s = re.sub(r'[^\w.-]', '', s)
    s = re.sub(r'__+', '_', s)
    return s if s else "unnamed_series"



def process_folder(folder_path, base_output_dir):

    if not os.path.isdir(folder_path):
        return f"SKIPPED: Folder not found -> {folder_path}"

    try:
        dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
        if not dicom_files:
            return f"SKIPPED: No .dcm files found in -> {folder_path}"

        first_dcm_path = os.path.join(folder_path, dicom_files[0])
        ds = pydicom.dcmread(first_dcm_path, stop_before_pixels=True)

        accession_no = getattr(ds, 'AccessionNumber', None)
        if not accession_no:
            return f"SKIPPED: Missing AccessionNumber -> {folder_path}"

        # Keep full accession (only sanitize for filesystem)
        safe_accession = sanitize_filename(str(accession_no))

        accession_output_dir = os.path.join(base_output_dir, safe_accession)
        os.makedirs(accession_output_dir, exist_ok=True)

        output_filename_format = '%s_%d'

        command = [
            'dcm2niix',
            '-o', accession_output_dir,
            '-f', output_filename_format,
            '-z', 'y',
            '-b', 'y',
            folder_path
        ]

        subprocess.run(command, check=True, capture_output=True, text=True)
        return f"SUCCESS: {folder_path} -> {safe_accession}"

    except pydicom.errors.InvalidDicomError:
        return f"ERROR: Invalid DICOM -> {folder_path}"
    except subprocess.CalledProcessError as e:
        return f"ERROR dcm2niix {folder_path}: {e.stderr.strip()}"
    except Exception as e:
        return f"ERROR: Unexpected issue with {folder_path}: {e}"


def convert_folders_to_nifti_parallel(csv_path, base_output_dir, max_workers):

    try:
        cmd = "where" if sys.platform == "win32" else "which"
        subprocess.run([cmd, "dcm2niix"], check=True, capture_output=True)
        print("dcm2niix found.")
    except:
        print("ERROR: dcm2niix not found in PATH")
        return

    try:
        df = pd.read_csv(csv_path)
        folder_paths = df['FolderPath'].tolist()
        print(f"{len(folder_paths)} folders to process.")
    except Exception as e:
        print(f"CSV error: {e}")
        return

    os.makedirs(base_output_dir, exist_ok=True)

    print(f"Output: {base_output_dir}")

    effective_workers = min(max_workers, os.cpu_count()) if max_workers else os.cpu_count()
    print(f"Using {effective_workers} workers")

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(process_folder, path, base_output_dir): idx
            for idx, path in enumerate(folder_paths)
        }

        for future in tqdm.tqdm(as_completed(futures), total=len(folder_paths), desc="Processing"):
            result = future.result()
            if "SUCCESS" not in result:
                tqdm.tqdm.write(result)

    print("Done.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert DICOM folders to NIfTI using dcm2niix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python src/mr_rate_preprocessing/mri_preprocessing/dcm2nii.py \\
        --input-csv data/raw/batch00/batch00_folder_paths.csv \\
        --output-dir data/raw/batch00/batch00_raw_niftis \\
        --max-workers 16
        """
    )

    parser.add_argument(
        "--input-csv", "-i",
        type=Path,
        required=True,
        help="CSV file with a 'FolderPath' column listing DICOM folders to convert"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory where per-accession NIfTI folders will be written"
    )

    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: all available CPUs)"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_folders_to_nifti_parallel(args.input_csv, args.output_dir, args.max_workers)
