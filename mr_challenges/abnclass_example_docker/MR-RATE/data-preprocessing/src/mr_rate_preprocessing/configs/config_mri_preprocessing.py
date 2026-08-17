"""
Configuration for MR-RATE MRI Preprocessing Pipeline.
"""
from pathlib import Path


# =============================================================================
# Metadata Columns Configuration (for pacs_metadata_filtering.py and prepare_metadata.py)
# =============================================================================
METADATA_COLUMNS_CONFIG_PATH = (Path(__file__).parent / "config_metadata_columns.json").resolve()

NOT_ALLOWED_TO_HAVE_NAN_COLUMNS = [
    'AccessionNumber', 
    'SeriesDescription', 
    'SeriesNumber', 
    'SeriesInstanceUID', 
    'StudyInstanceUID', 
    "Patient'sAge",
]

NOT_ALLOWED_TO_HAVE_DUPLICATES_COLUMNS = [
    'AccessionNumber',
    'SeriesDescription',
    'SeriesNumber'
]


# =============================================================================
# Rule-Based Modality Filtering Parameters (for modality_filtering.py)
# =============================================================================

# Patient age filter
MIN_PATIENT_AGE = 13  # Minimum patient age in years for inclusion

# Image quality filter
MIN_SHAPE = 16  # Minimum shape in each dimension (voxels)
MIN_FOV = 140  # Minimum field of view in each dimension (mm)
MAX_FOV = 350  # Maximum field of view in each dimension (mm)

# Accepted values for classified_modality column from rule-based classifier
# Options: T1w, T2w, T2-FLAIR, T2star, DWI, MRA, ASL, PDw, UNKNOWN
ACCEPTED_CLASSIFIED_MODALITIES = ["T1w", "T2w", "T2-FLAIR", "SWI", "MRA"]

# Whether to include derived series (is_derived=True)
INCLUDE_DERIVED_SERIES = False

# Accepted acquisition planes (computed from ImageOrientation(Patient) in 3_modality_filtering.py)
# Options: AXIAL, CORONAL, SAGITTAL, OBLIQUE, UNKNOWN
ACCEPTED_ACQUISITION_PLANES = ["AXIAL", "CORONAL", "SAGITTAL", "OBLIQUE"]

# DWI sub-types to exclude (these are computed maps, not raw diffusion data)
# Options: ADC, eADC, FA, TRACE, TENSOR, DIRECTIONAL, B0
EXCLUDED_DWI_SUBTYPES = []

REQUIRED_METADATA_COLUMNS = [
    "AccessionNumber",
    "SeriesNumber",
    "SeriesDescription",
    "Patient'sAge",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "ImageOrientation(Patient)",
    "classified_modality",
    "is_derived",
    "is_localizer",
    "is_subtraction",
]


# =============================================================================
# Field Abbreviation Lookup Table (for modality_filtering.py)
# Maps classification field values to short codes for modality naming
# Modality name format: {modality}-{role}-{plane}
# Example: t1w-raw-sag
# =============================================================================
FIELD_ABBREVIATIONS = {
    # classified_modality -> modality abbreviation
    "classified_modality": {
        "T1w": "t1w",
        "T2w": "t2w",
        "T2-FLAIR": "flair",
        "T2star": "t2star",
        "SWI": "swi",
        "DWI": "dwi",
        "MRA": "mra",
        "ASL": "asl",
        "PDw": "pdw",
        "UNKNOWN": "unk",
    },
    # is_derived boolean -> role abbreviation
    "is_derived": {
        True: "der",
        False: "raw",
    },
    # acquisition_plane -> plane abbreviation
    "acquisition_plane": {
        "AXIAL": "axi",
        "SAGITTAL": "sag",
        "CORONAL": "cor",
        "OBLIQUE": "obl",
        "UNKNOWN": "unk",
    },
}


# =============================================================================
# Center Modality Selection Criteria (for modality_filtering.py)
# Defines which scan should be used as the center (reference) modality
# In case of multiple matches, the one with the least SeriesNumber is selected
# =============================================================================
CENTER_MODALITY_CRITERIA = {
    "classified_modality": "T1w",
    "is_derived": False,
}


# =============================================================================
# Brain Segmentation Configuration (for brain_segmentation_and_defacing.py)
# =============================================================================

# Segmentation mode: 'fast' (1 model) or 'accurate' (5-model ensemble)
HDBET_MODE = "accurate"

# Enable test-time augmentation (mirroring) - improves mask quality at the cost of speed
HDBET_DO_TTA = True

# Keep only the largest connected component in the output brain mask
HDBET_POSTPROCESS = True

# Enable torch.compile() for faster inference (only makes sense if input tensor shape is fixed)
HDBET_COMPILE = False

# Enable BF16 mixed precision (recommended on modern GPUs, disable on CPU or older hardware)
HDBET_MIXED_PREC = True
