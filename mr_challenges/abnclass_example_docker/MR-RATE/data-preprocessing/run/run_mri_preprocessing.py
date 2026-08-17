"""
MR-RATE MRI Preprocessing Pipeline Runner

Runs the five preprocessing steps sequentially for a given batch:
  1. DICOM to NIfTI Conversion     (dcm2nii.py)
  2. PACS Metadata Filtering       (pacs_metadata_filtering.py)
  3. Series Classification         (series_classification.py)
  4. Modality Filtering            (modality_filtering.py)
  5. Brain Segmentation & Defacing (brain_segmentation_and_defacing.py)

Must be run from the data-preprocessing directory:
    python run/run_mri_preprocessing.py --config run/configs/mri_batch00.yaml
"""

import sys

from utils import load_config, parse_args, run_step


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_dcm2nii(cfg: dict) -> None:
    step_cfg     = cfg["dcm2nii"]
    workers_flag = ["--max-workers", str(step_cfg["max_workers"])] if step_cfg.get("max_workers") else []

    run_step(
        "Step 1 / 5 — DICOM to NIfTI Conversion",
        [
            sys.executable,
            "src/mr_rate_preprocessing/mri_preprocessing/dcm2nii.py",
            "--input-csv",  step_cfg["input_csv"],
            "--output-dir", step_cfg["output_dir"],
            *workers_flag,
        ],
    )


def step_pacs_metadata_filtering(cfg: dict) -> None:
    step_cfg     = cfg["pacs_metadata_filtering"]
    verbose_flag = ["--verbose"] if cfg.get("verbose", False) else []

    run_step(
        "Step 2 / 5 — PACS Metadata Filtering",
        [
            sys.executable,
            "src/mr_rate_preprocessing/mri_preprocessing/pacs_metadata_filtering.py",
            "--input-csv",  step_cfg["input_csv"],
            "--output-csv", step_cfg["output_csv"],
            "--log-dir",    cfg["log_dir"],
            *verbose_flag,
        ],
    )


def step_series_classification(cfg: dict) -> None:
    # Input is the output of the previous step — single source of truth.
    input_csv = cfg["pacs_metadata_filtering"]["output_csv"]

    # series_classification.py takes a single positional argument; its output
    # paths are auto-derived as <stem>_classified.csv / <stem>_classification_summary.csv.
    run_step(
        "Step 3 / 5 — Series Classification",
        [
            sys.executable,
            "src/mr_rate_preprocessing/mri_preprocessing/series_classification.py",
            input_csv,
        ],
    )


def step_modality_filtering(cfg: dict) -> None:
    step_cfg     = cfg["modality_filtering"]
    verbose_flag = ["--verbose"] if cfg.get("verbose", False) else []

    # Derive the classified CSV from the upstream output — same logic as series_classification.py.
    classified_csv = str(cfg["pacs_metadata_filtering"]["output_csv"]).replace(".csv", "_classified.csv")

    # raw_data_dir is taken from dcm2nii.output_dir — single source of truth.
    raw_data_dir = cfg["dcm2nii"]["output_dir"]

    run_step(
        "Step 4 / 5 — Modality Filtering",
        [
            sys.executable,
            "src/mr_rate_preprocessing/mri_preprocessing/modality_filtering.py",
            "--raw-data-dir",   raw_data_dir,
            "--classified-csv", classified_csv,
            "--output-json",    step_cfg["output_json"],
            "--output-csv",     step_cfg["output_csv"],
            "--num-processes",  str(step_cfg["num_processes"]),
            "--log-dir",        cfg["log_dir"],
            *verbose_flag,
        ],
    )


def step_brain_segmentation(cfg: dict) -> None:
    step_cfg     = cfg["brain_segmentation"]
    mod_cfg      = cfg["modality_filtering"]
    verbose_flag = ["--verbose"] if cfg.get("verbose", False) else []

    # raw_dir is taken from dcm2nii.output_dir — single source of truth.
    raw_dir = cfg["dcm2nii"]["output_dir"]

    run_step(
        "Step 5 / 5 — Brain Segmentation & Defacing",
        [
            sys.executable,
            "src/mr_rate_preprocessing/mri_preprocessing/brain_segmentation_and_defacing.py",
            "--modalities-json", mod_cfg["output_json"],
            "--raw-dir",         raw_dir,
            "--output-dir",      step_cfg["output_dir"],
            "--device",          str(step_cfg["device"]),
            "--log-dir",         cfg["log_dir"],
            *verbose_flag,
        ],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args("Run the MRI preprocessing pipeline for a single batch.")
    cfg  = load_config(args.config)

    batch_id = cfg["batch_id"]
    print(f"\nStarting MRI preprocessing pipeline — batch: {batch_id}")
    print(f"Config: {args.config}")

    step_dcm2nii(cfg)
    step_pacs_metadata_filtering(cfg)
    step_series_classification(cfg)
    step_modality_filtering(cfg)
    step_brain_segmentation(cfg)

    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  MRI preprocessing pipeline complete — batch: {batch_id}")
    print(f"{separator}\n")


if __name__ == "__main__":
    main()
