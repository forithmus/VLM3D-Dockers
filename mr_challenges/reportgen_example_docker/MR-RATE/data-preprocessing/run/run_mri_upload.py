"""
MR-RATE MRI Upload Pipeline Runner

Runs the two upload steps sequentially for a given batch:
  1. Zip & Upload to Hugging Face   (zip_and_upload.py)
  2. Prepare & Upload Metadata      (prepare_metadata.py)

Input paths are taken directly from the preprocessing sections of the shared
config so there is a single source of truth for inter-step paths.

Must be run from the data-preprocessing directory:
    python run/run_mri_upload.py --config run/configs/mri_batch00.yaml
"""

import sys

from utils import load_config, parse_args, run_step


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_zip_and_upload(cfg: dict) -> None:
    step_cfg     = cfg["zip_and_upload"]
    seg_cfg      = cfg["brain_segmentation"]
    mod_cfg      = cfg["modality_filtering"]
    verbose_flag = ["--verbose"] if cfg.get("verbose", False) else []

    optional_flags = []
    if step_cfg.get("skip_upload", False):
        optional_flags.append("--skip-upload")
    if step_cfg.get("delete_zips", False):
        optional_flags.append("--delete-zips")
    if step_cfg.get("xet_high_perf", False):
        optional_flags.append("--xet-high-perf")

    run_step(
        "Step 1 / 2 — Zip & Upload",
        [
            sys.executable,
            "src/mr_rate_preprocessing/mri_preprocessing/zip_and_upload.py",
            "--input-dir",       seg_cfg["output_dir"],
            "--modalities-json", mod_cfg["output_json"],
            "--batch-id",        cfg["batch_id"],
            "--output-dir",      step_cfg["output_dir"],
            "--repo-id",         step_cfg["repo_id"],
            "--num-zip-workers", str(step_cfg["num_zip_workers"]),
            "--num-hf-workers",  str(step_cfg["num_hf_workers"]),
            "--hf-timeout",      str(step_cfg["hf_timeout"]),
            "--log-dir",         cfg["log_dir"],
            *optional_flags,
            *verbose_flag,
        ],
    )


def step_prepare_metadata(cfg: dict) -> None:
    step_cfg     = cfg["prepare_metadata"]
    seg_cfg      = cfg["brain_segmentation"]
    mod_cfg      = cfg["modality_filtering"]
    verbose_flag = ["--verbose"] if cfg.get("verbose", False) else []

    optional_flags = []
    if step_cfg.get("skip_upload", False):
        optional_flags.append("--skip-upload")

    run_step(
        "Step 2 / 2 — Prepare Metadata",
        [
            sys.executable,
            "src/mr_rate_preprocessing/mri_preprocessing/prepare_metadata.py",
            "--processed-dir",       seg_cfg["output_dir"],
            "--modalities-json",     mod_cfg["output_json"],
            "--input-csv",           mod_cfg["output_csv"],
            "--patient-mapping-csv", step_cfg["patient_mapping_csv"],
            "--study-date-mapping",  step_cfg["study_date_mapping"],
            "--output-csv",          step_cfg["output_csv"],
            "--repo-id",             step_cfg["repo_id"],
            "--batch-id",            cfg["batch_id"],
            "--num-hf-workers",      str(step_cfg["num_hf_workers"]),
            "--hf-timeout",          str(step_cfg["hf_timeout"]),
            "--log-dir",             cfg["log_dir"],
            *optional_flags,
            *verbose_flag,
        ],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args("Run the MRI upload pipeline for a single batch.")
    cfg  = load_config(args.config)

    batch_id = cfg["batch_id"]
    print(f"\nStarting MRI upload pipeline — batch: {batch_id}")
    print(f"Config: {args.config}")

    step_zip_and_upload(cfg)
    step_prepare_metadata(cfg)

    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  MRI upload pipeline complete — batch: {batch_id}")
    print(f"{separator}\n")


if __name__ == "__main__":
    main()
