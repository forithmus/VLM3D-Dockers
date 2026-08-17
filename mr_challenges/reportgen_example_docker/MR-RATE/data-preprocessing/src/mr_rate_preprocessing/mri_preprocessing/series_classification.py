"""
MR-RATE MRI Preprocessing Pipeline - Rule-Based Series Modality Classification Block

Classifies brain MRI series into contrast-weighted modalities using DICOM metadata
fields — without relying on any existing (potentially incorrect) modality labels.

Target modalities:
  - T1w       : T1-weighted (MPRAGE, TFE, FFE, PETRA, VIBE, BRAVO, SPGR, etc.)
  - T2-FLAIR  : T2-weighted Fluid-Attenuated Inversion Recovery
  - T2w       : T2-weighted (spin-echo / TSE / FSE / SPACE / CUBE / VISTA)
  - PDw       : Proton-density weighted
  - T2star    : T2*-weighted (GRE multi-echo, long-TE gradient echo)
  - SWI       : Susceptibility-Weighted Imaging (SWI, SWAN, phase/mIP maps)
  - DWI       : Diffusion-weighted imaging (includes DTI)
  - MRA       : MR Angiography (TOF, phase-contrast, CE-MRA)
  - ASL       : Arterial Spin Labeling (perfusion)
  - STIR      : Short-TI Inversion Recovery (fat-suppressed IR)
  - UNKNOWN   : Could not classify

Additional flags produced per row:
  - is_derived            : True if image is a reformat / post-processed derivation
  - is_subtraction        : True if image is a subtraction map
  - is_contrast_enhanced  : True if acquired after gadolinium injection
  - dwi_sub_type          : DWI sub-type (TRACE, ADC, FA, eADC, B0, TENSOR,
                            DIRECTIONAL, or empty)
  - sequence_family       : Human-readable family (MPRAGE, SPACE-FLAIR, TFE, etc.)

Classification hierarchy (evaluated top-to-bottom, first match wins):
  0. DICOM diffusion-encoding metadata (b-value, gradient dirs)  — definitive
  1. Vendor-specific pulse-sequence identifiers (most reliable when present)
  2. DICOM ScanningSequence + SequenceVariant + numeric parameters
  3. Protocol / series description keyword matching
  4. Pure numeric-parameter thresholds  (fallback)
  5. Philips enhanced-DICOM private tags (fallback)

Example Usage:
    python src/mr_rate_preprocessing/mri_preprocessing/series_classification.py \\
        data/interim/batch00/batch00_raw_metadata.csv

    Outputs (derived from input path automatically):
        data/interim/batch00/batch00_raw_metadata_classified.csv
        data/interim/batch00/batch00_raw_metadata_classification_summary.csv

References:
  - DICOM PS3.3 C.8.3 MR Image Module
  - DICOM PS3.3 C.8.13.5 MR Diffusion Macro
  - mriquestions.com (MPRAGE, SPACE/CUBE/VISTA, FLAIR, DWI)
  - Defined-term values for ScanningSequence: SE, IR, GR, EP, RM
  - Defined-term values for SequenceVariant:  SK, MTC, SS, TRSS, SP, MP, OSP, NONE
  - BIDS specification for MRI modality suffixes (T1w, T2w, FLAIR, dwi, etc.)
  - Parameter ranges from: Radiopaedia, e-MRI/IMAIOS, mrimaster.com, PMC literature
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================================
# 1.  HELPERS
# ============================================================================
def _safe_float(val):
    """Return float or NaN."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _safe_str(val):
    """Return stripped lowercase string, or empty string for NaN."""
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def _parse_scan_seq(raw: str) -> set:
    """Parse DICOM ScanningSequence (potentially multi-valued) into a set.

    DICOM multi-valued strings use backslash as separator, e.g. 'SE\\IR'.
    After lowercasing this becomes 'se\\ir' or 'se\\\\ir' depending on
    how the CSV was serialised.  We normalise and split into {'se', 'ir'}.
    """
    if not raw:
        return set()
    normalised = raw.replace("\\\\", "\\")
    return {tok.strip() for tok in normalised.split("\\") if tok.strip()}


# ============================================================================
# 2.  KEYWORD MATCHERS  (compiled once)
# ============================================================================

# --- FLAIR ---
_RE_FLAIR = re.compile(
    r"flair|dark[\-_\s]?fluid|brain[\-_\s]?view(?:[\-_\s]?flair)?",
    re.IGNORECASE,
)

# --- T1-weighted ---
# FIX: t1_se -> t1[\-_\s]se to match "t1 se" (space-separated) as well
_RE_T1 = re.compile(
    r"(?<![a-zA-Z])t1[\-_\s]?w|mprage|t1w_|_t1w|"
    r"(?<![a-zA-Z])t1[\-_\s]?(?:mprage|tfe|ffe|vibe|petra|flash|gre|bravo|spgr|fspgr)|"
    r"st1w|vst1w|t1_mprage|t1_petra|t1_vibe|t1_space|"
    r"t1_tse|t1_qtse|t1[\-_\s]se(?![a-zA-Z])|t1_tir",
    re.IGNORECASE,
)

# --- T2-weighted (non-FLAIR, non-T2*) ---
_RE_T2 = re.compile(
    r"(?<![a-zA-Z])t2[\-_\s]?w|t2_tse|t2_fse|t2_se|"
    r"t2[\-_\s]?space(?!.*(?:flair|dark[\-_]?fluid))|"
    r"t2[\-_\s]?cube|t2[\-_\s]?vista|t2[\-_\s]?drive|"
    r"t2_blade|t2_propeller|"
    r"t2_haste|t2_ciss|t2_qtse|st2w",
    re.IGNORECASE,
)

# --- DWI / Diffusion ---
_RE_DWI = re.compile(
    r"(?<![a-zA-Z])dwi(?:bs)?(?![a-zA-Z])|(?<![a-zA-Z])dti(?![a-zA-Z])|"
    r"diffus|diff_|_diff(?![a-zA-Z])|"
    r"(?<![a-zA-Z])trace(?![a-zA-Z])|(?<![a-zA-Z])[de]?adc(?![a-zA-Z])|"
    r"(?<![a-zA-Z])b[_\-]?\d{2,4}(?![a-zA-Z])|"
    r"(?<![a-zA-Z])b[\-_]?val|"
    r"(?<![a-zA-Z])tensor|(?<![a-zA-Z])fa[\-_\s]?map|"
    r"ep2d[\-_]?diff|resolve[\-_]?diff|"
    r"ep[\-_]?b\d|"
    r"dwise|diff[\-_]?se|ssepi|mddw",
    re.IGNORECASE,
)

# --- DWI sub-type in ImageType / SeriesDescription ---
_RE_DWI_ADC    = re.compile(r"(?<![a-zA-Z])adc(?![a-zA-Z])|apparent[\-_\s]?diff[\-_\s]?coeff", re.IGNORECASE)
_RE_DWI_FA     = re.compile(r"(?<![a-zA-Z])fa[\-_\s]?map|fractional[\-_\s]?anisotropy", re.IGNORECASE)
_RE_DWI_TRACE  = re.compile(r"(?<![a-zA-Z])trace(?:w)?(?![a-zA-Z])|(?<![a-zA-Z])isotropic(?![a-zA-Z])", re.IGNORECASE)
_RE_DWI_EADC   = re.compile(r"(?<![a-zA-Z])e[\-_]?adc(?![a-zA-Z])|exponential[\-_\s]?adc", re.IGNORECASE)
_RE_DWI_TENSOR = re.compile(r"(?<![a-zA-Z])tensor(?![a-zA-Z])", re.IGNORECASE)

# --- SWI (Susceptibility-Weighted Imaging — specific) ---
#     NOTE: "merge" removed as standalone match — too many false positives
#     on unrelated protocols.  It is only matched when adjacent to SWI context.
_RE_SWI = re.compile(
    r"(?<![a-zA-Z])swi(?![a-zA-Z])|susceptib|"
    r"(?<![a-zA-Z])swan(?![a-zA-Z])|"
    r"(?:swi|pha)[\-_\s]?merge|merge[\-_\s]?(?:swi|pha)|"
    r"v?ven[\-_\s]?bold|venobold|haemosiderin|"
    r"(?<![a-zA-Z])swip(?![a-zA-Z])|"
    r"(?<![a-zA-Z])v[\-_\s]?swi(?:p)?(?![a-zA-Z])|"
    r"(?<![a-zA-Z])min[\-_\s]?ip[\-_\s]?(?:swi|pha)|"
    r"(?<![a-zA-Z])pha[\-_\s]?(?:filt|image|map)",
    re.IGNORECASE,
)

# --- T2*-weighted (gradient-echo T2*-decay contrast, non-SWI) ---
_RE_T2STAR = re.compile(
    r"(?<![a-zA-Z])t2[\-_\s]?\*|t2[\-_\s]?star|"
    r"(?<![a-zA-Z])medic(?![a-zA-Z])|"
    r"(?<![a-zA-Z])me2d(?![a-zA-Z])",
    re.IGNORECASE,
)

# --- Combined SWI or T2* (used where either should trigger) ---
_RE_SWI_OR_T2STAR = re.compile(
    r"(?<![a-zA-Z])swi(?![a-zA-Z])|susceptib|"
    r"(?<![a-zA-Z])t2[\-_\s]?\*|t2[\-_\s]?star|"
    r"(?<![a-zA-Z])swan(?![a-zA-Z])|"
    r"(?<![a-zA-Z])medic(?![a-zA-Z])|"
    r"venobold|haemosiderin",
    re.IGNORECASE,
)

# --- MRA ---
_RE_MRA = re.compile(
    r"(?<![a-z])mra(?![a-z])|(?<![a-z])angio(?:graphy)?(?![a-z])|"
    r"(?<![a-z])tof[\-_\s]?(?:mra)?|time[\-_\s]?of[\-_\s]?flight|"
    r"phase[\-_\s]?contrast[\-_\s]?(?:mra|angio)|"
    r"(?<![a-z])ce[\-_\s]?mra|"
    r"(?<![a-z])qmra(?![a-z])|"
    r"fl_tof|fl2d_tof|fl3d_tof|"
    r"flow[\-_]?pc|carotid",
    re.IGNORECASE,
)

# --- ASL (Arterial Spin Labeling) ---
_RE_ASL = re.compile(
    r"(?<![a-zA-Z])asl(?![a-zA-Z])|"
    r"arterial[\-_\s]?spin[\-_\s]?label|"
    r"(?<![a-zA-Z])pcasl(?![a-zA-Z])|(?<![a-zA-Z])pasl(?![a-zA-Z])|(?<![a-zA-Z])casl(?![a-zA-Z])|"
    r"(?<![a-zA-Z])3d[\-_\s]?asl|"
    r"pseudo[\-_\s]?continuous[\-_\s]?asl|"
    r"(?<![a-zA-Z])cbf[\-_\s]?map|"
    r"perfus[\-_\s]?(?:ion)?[\-_\s]?(?:weight|map|image)",
    re.IGNORECASE,
)

# --- Contrast-enhanced ---
_RE_CE = re.compile(
    r"\+c(?:\s|$|[^a-z])|"
    r"post[\-_\s]?(?:con|gad)|"
    r"kontras|gadol|"
    r"(?<![a-z])ce[\-_]|"
    r"(?<![a-z])ce(?![a-z])",
    re.IGNORECASE,
)

# --- Subtraction ---
_RE_SUB = re.compile(r"subtraction|sub_|ssub|_sub(?![a-zA-Z])", re.IGNORECASE)

# --- STIR (keyword-level — complements TIER2 numeric IR+TI detection) ---
_RE_STIR = re.compile(
    r"(?<![a-zA-Z])stir(?![a-zA-Z])",
    re.IGNORECASE,
)

# --- Siemens SequenceName patterns ---
_RE_SEQ_SPCIR   = re.compile(r"spcir",  re.IGNORECASE)
_RE_SEQ_TFL3D   = re.compile(r"tfl3d",  re.IGNORECASE)
_RE_SEQ_PETRA   = re.compile(r"petra",   re.IGNORECASE)
_RE_SEQ_FL3D    = re.compile(r"fl3d",    re.IGNORECASE)
_RE_SEQ_EP_B    = re.compile(r"ep[\-_]?b\d|ep2d[\-_]?diff", re.IGNORECASE)
_RE_SEQ_TOF     = re.compile(r"fl\d?d?[\-_]?tof",      re.IGNORECASE)
_RE_SEQ_TSE     = re.compile(r"tse\d?d|spc(?!ir)",      re.IGNORECASE)


# ============================================================================
# 3.  CORE CLASSIFIER
# ============================================================================
def classify_row(row: pd.Series) -> dict:
    """Classify a single MRI series row and return a dict of derived fields."""

    # --- Extract numeric values ------------------------------------------------
    te  = _safe_float(row.get("TE_ms"))
    tr  = _safe_float(row.get("TR_ms"))
    ti  = _safe_float(row.get("TI_ms"))
    fa  = _safe_float(row.get("FlipAngle"))
    etl = _safe_float(row.get("EchoTrainLength"))

    # Diffusion b-value: try multiple column names used in the wild
    bval = _safe_float(row.get("Diffusionb-value"))
    if np.isnan(bval):
        bval = _safe_float(row.get("[DiffusionB-Factor]"))
    if np.isnan(bval):
        bval = _safe_float(row.get("DiffusionBValue"))
    if np.isnan(bval):
        bval = _safe_float(row.get("bval"))

    # --- Extract string values ------------------------------------------------
    scan_seq_raw = _safe_str(row.get("ScanningSequence"))
    scan_seq_set = _parse_scan_seq(scan_seq_raw)
    seq_var      = _safe_str(row.get("SequenceVariant"))
    seq_name     = _safe_str(row.get("SequenceName"))
    pulse_name   = _safe_str(row.get("PulseSequenceName"))
    protocol     = _safe_str(row.get("ProtocolName"))
    series_desc  = _safe_str(row.get("SeriesDescription"))
    image_type   = _safe_str(row.get("ImageType"))
    image_type_text = _safe_str(row.get("ImageTypeText"))
    scan_opts    = _safe_str(row.get("ScanOptions"))
    acq_type     = _safe_str(row.get("MRAcquisitionType"))
    manufacturer = _safe_str(row.get("Manufacturer"))

    # Diffusion directionality
    diff_dir = _safe_str(row.get("Privatetagdata_0_DiffusionDirectionality"))
    if not diff_dir:
        diff_dir = _safe_str(row.get("DiffusionDirectionality"))

    # Diffusion gradient orientation
    diff_grad = _safe_str(row.get("DiffusionGradientOrientation"))

    # Philips enhanced-DICOM private tags
    priv_contrast = _safe_str(row.get("Privatetagdata_0_AcquisitionContrast"))
    priv_ir       = _safe_str(row.get("Privatetagdata_0_InversionRecovery"))

    # EPIFactor (Philips)
    epi_factor = _safe_float(row.get("[EPIFactor]"))
    if np.isnan(epi_factor):
        epi_factor = _safe_float(row.get("EPIFactor"))

    combined_desc = f"{protocol} {series_desc}"
    all_text = f"{protocol} {series_desc} {seq_name} {pulse_name} {image_type} {image_type_text}"

    # --- Flags -----------------------------------------------------------------
    is_derived     = "derived" in image_type
    is_subtraction = bool(_RE_SUB.search(image_type)) or bool(_RE_SUB.search(series_desc))
    is_ce          = bool(_RE_CE.search(combined_desc))
    is_localizer   = bool(re.search(r"localiz|scout|survey|surview", combined_desc, re.IGNORECASE))

    # --- DWI sub-type detection ------------------------------------------------
    dwi_sub = ""
    if _RE_DWI_ADC.search(all_text):
        dwi_sub = "ADC"
    elif _RE_DWI_EADC.search(all_text):
        dwi_sub = "eADC"
    elif _RE_DWI_FA.search(all_text):
        dwi_sub = "FA"
    elif _RE_DWI_TRACE.search(all_text):
        dwi_sub = "TRACE"
    elif _RE_DWI_TENSOR.search(all_text):
        dwi_sub = "TENSOR"
    if not dwi_sub:
        if "adc" in image_type or "adc" in image_type_text:
            dwi_sub = "ADC"
        elif "fa" in image_type_text and "diffusion" in image_type_text:
            dwi_sub = "FA"
        elif "tracew" in image_type or "trace" in image_type_text:
            dwi_sub = "TRACE"
        elif "diffusion" in image_type and is_derived:
            dwi_sub = "ADC"

    # =========================================================================
    #  TIER 0 — DICOM Diffusion-encoding metadata (definitive when present)
    # =========================================================================
    has_diffusion_encoding = False

    if not np.isnan(bval) and bval > 50:
        has_diffusion_encoding = True

    if diff_dir and diff_dir not in ("", "none", "nan"):
        has_diffusion_encoding = True

    if diff_grad:
        try:
            grad_vals = [float(x) for x in diff_grad.replace("\\\\", "\\").split("\\") if x.strip()]
            if any(abs(g) > 0.001 for g in grad_vals):
                has_diffusion_encoding = True
        except (ValueError, AttributeError):
            pass

    if (not np.isnan(epi_factor) and epi_factor > 1) and _RE_DWI.search(all_text):
        has_diffusion_encoding = True

    if has_diffusion_encoding:
        if dwi_sub in ("ADC", "eADC", "FA"):
            family = f"DWI derived map ({dwi_sub})"
        elif dwi_sub == "TRACE":
            family = "DWI trace/isotropic"
        elif diff_dir in ("directional", "bmatrix"):
            family = "DTI (multi-direction diffusion)"
            if not dwi_sub:
                dwi_sub = "DIRECTIONAL"
        else:
            family = "DWI (diffusion-weighted)"
        modality = "DWI"
        rule     = f"TIER0: Diffusion metadata (b={bval}, dir={diff_dir})"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if not np.isnan(bval) and bval == 0:
        if _RE_DWI.search(all_text) or "ep" in scan_seq_set:
            modality = "DWI"
            family   = "DWI b=0 reference"
            dwi_sub  = "B0"
            rule     = "TIER0: b-value=0 + DWI context"
            return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # =========================================================================
    #  TIER 1 — Vendor-specific pulse-sequence identifiers
    # =========================================================================

    # Siemens SequenceName: DWI EPI
    if _RE_SEQ_EP_B.search(seq_name):
        modality = "DWI"
        family   = "Siemens ep2d_diff (SE-EPI diffusion)"
        if not dwi_sub and "adc" in combined_desc:
            dwi_sub = "ADC"
        elif not dwi_sub and "trace" in combined_desc:
            dwi_sub = "TRACE"
        rule = "TIER1: SequenceName matches ep_b/ep2d_diff"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_SEQ_TOF.search(seq_name):
        modality = "MRA"
        family   = "Siemens TOF (fl_tof / fl2d_tof)"
        rule     = "TIER1: SequenceName matches fl*tof"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # Philips PulseSequenceName
    if pulse_name == "tir":
        modality = "T2-FLAIR"
        family   = "Philips-TIR (3D Brain VIEW FLAIR)"
        rule     = "TIER1: PulseSequenceName=TIR"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if pulse_name in ("t1tfe",):
        modality = "T1w"
        family   = "Philips-T1TFE (3D MPRAGE-equivalent)"
        rule     = "TIER1: PulseSequenceName=T1TFE"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if pulse_name in ("t1ffe",):
        if _RE_SWI.search(combined_desc):
            modality = "SWI"
            family   = "Philips-T1FFE venous BOLD / SWI"
            rule     = "TIER1: PulseSequenceName=T1FFE + SWI keyword"
        elif not np.isnan(te) and te > 15:
            modality = "T2star"
            family   = "Philips-T1FFE long-TE (T2*-weighted)"
            rule     = "TIER1: PulseSequenceName=T1FFE + TE>15"
        else:
            modality = "T1w"
            family   = "Philips-T1FFE (T1 Fast Field Echo)"
            rule     = "TIER1: PulseSequenceName=T1FFE"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if pulse_name in ("dwise", "dwi_se", "dwisse"):
        modality = "DWI"
        family   = "Philips-DwiSE (SE-EPI diffusion)"
        rule     = "TIER1: PulseSequenceName=DwiSE"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if pulse_name in ("t2tse", "t2se"):
        modality = "T2w"
        family   = "Philips-T2TSE (Turbo Spin Echo)"
        rule     = "TIER1: PulseSequenceName=T2TSE/T2SE"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if pulse_name in ("swip",):
        modality = "SWI"
        family   = "Philips-SWIp (Susceptibility-Weighted Imaging)"
        rule     = "TIER1: PulseSequenceName=SWIp"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if pulse_name in ("3d-asl", "pcasl", "3dasl"):
        modality = "ASL"
        family   = "Philips-3D-ASL"
        rule     = "TIER1: PulseSequenceName=3D-ASL/PCASL"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if pulse_name in ("tof", "pca"):
        modality = "MRA"
        family   = "Philips-TOF/PCA (MR Angiography)"
        rule     = "TIER1: PulseSequenceName=TOF/PCA"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # Philips B-FFE (balanced SSFP / FIESTA / CISS-equivalent)
    # Mixed T2/T1 contrast — NOT T1-weighted.  Used for inner ear, CSF, cine.
    if pulse_name in ("b-ffe", "bffe"):
        modality = "UNKNOWN"
        family   = "Philips-bSSFP (Balanced FFE / FIESTA / CISS)"
        rule     = "TIER1: PulseSequenceName=B-FFE (balanced SSFP, mixed contrast)"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # Philips plain FFE (generic gradient echo) — contrast depends on parameters
    if pulse_name in ("ffe",):
        if _RE_SWI.search(combined_desc):
            modality = "SWI"
            family   = "Philips-FFE SWI"
            rule     = "TIER1: PulseSequenceName=FFE + SWI keyword"
        elif _RE_T2.search(combined_desc) or _RE_T2STAR.search(combined_desc):
            if not np.isnan(te) and te > 15:
                modality = "T2star"
                family   = "Philips-FFE T2*-weighted"
                rule     = "TIER1: PulseSequenceName=FFE + T2/T2* keyword + TE>15"
            else:
                modality = "T2star"
                family   = "Philips-FFE (T2W_FFE — GRE T2*-weighted)"
                rule     = "TIER1: PulseSequenceName=FFE + T2 keyword"
        elif not np.isnan(te) and te > 15:
            modality = "T2star"
            family   = "Philips-FFE long-TE (T2*-weighted)"
            rule     = "TIER1: PulseSequenceName=FFE + TE>15"
        elif _RE_T1.search(combined_desc):
            modality = "T1w"
            family   = "Philips-FFE T1-weighted"
            rule     = "TIER1: PulseSequenceName=FFE + T1 keyword"
        elif not np.isnan(te) and te < 7 and not np.isnan(fa) and fa > 15:
            modality = "T1w"
            family   = "Philips-FFE T1w (short TE, high FA)"
            rule     = "TIER1: PulseSequenceName=FFE + TE<7 + FA>15"
        else:
            modality = "UNKNOWN"
            family   = "Philips-FFE (ambiguous contrast)"
            rule     = "TIER1: PulseSequenceName=FFE, no clear contrast"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # Siemens SequenceName: structural
    if _RE_SEQ_SPCIR.search(seq_name):
        modality = "T2-FLAIR"
        family   = "Siemens-SPACE-IR (3D FLAIR)"
        rule     = "TIER1: SequenceName contains 'spcir'"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_SEQ_TFL3D.search(seq_name):
        modality = "T1w"
        family   = "Siemens-MPRAGE (turboFLASH 3D)"
        rule     = "TIER1: SequenceName contains 'tfl3d'"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_SEQ_PETRA.search(seq_name):
        modality = "T1w"
        family   = "Siemens-PETRA (pointwise encoding, ultrashort TE)"
        rule     = "TIER1: SequenceName contains 'Petra'"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_SEQ_FL3D.search(seq_name) and not _RE_SEQ_TFL3D.search(seq_name):
        if _RE_SWI.search(combined_desc):
            modality = "SWI"
            family   = "Siemens-SWI (FLASH 3D + SWI keyword)"
            rule     = "TIER1: SequenceName 'fl3d' + SWI keyword"
        elif not np.isnan(te) and te > 15:
            modality = "T2star"
            family   = "Siemens-T2* (FLASH 3D long TE)"
            rule     = "TIER1: SequenceName 'fl3d' + TE>15"
        elif _RE_T2STAR.search(combined_desc):
            modality = "T2star"
            family   = "Siemens-T2* (FLASH 3D + T2* keyword)"
            rule     = "TIER1: SequenceName 'fl3d' + T2* keyword"
        else:
            modality = "T1w"
            family   = "Siemens-FLASH3D (VIBE / GRE)"
            rule     = "TIER1: SequenceName contains 'fl3d' (VIBE/FLASH)"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # Siemens TSE/SPACE (non-IR)
    if _RE_SEQ_TSE.search(seq_name):
        if _RE_FLAIR.search(combined_desc):
            modality = "T2-FLAIR"
            family   = "Siemens-TSE/SPACE + FLAIR keyword"
            rule     = "TIER1: SequenceName TSE/SPACE + FLAIR keyword"
        elif not np.isnan(te) and te > 60:
            modality = "T2w"
            family   = "Siemens-TSE/SPACE T2-weighted"
            rule     = "TIER1: SequenceName TSE/SPACE + TE>60"
        elif not np.isnan(te) and te < 30 and not np.isnan(tr) and tr < 1000:
            modality = "T1w"
            family   = "Siemens-TSE T1-weighted"
            rule     = "TIER1: SequenceName TSE + short TE/TR"
        elif _RE_T2.search(combined_desc):
            modality = "T2w"
            family   = "Siemens-TSE/SPACE + T2 keyword"
            rule     = "TIER1: SequenceName TSE/SPACE + T2 keyword"
        else:
            modality = "T2w"
            family   = "Siemens-TSE/SPACE (default T2)"
            rule     = "TIER1: SequenceName TSE/SPACE (default)"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # GE-specific sequences
    if "bravo" in seq_name or "bravo" in combined_desc:
        modality = "T1w"
        family   = "GE-BRAVO (3D T1w)"
        rule     = "TIER1: BRAVO keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)
    if "cube" in seq_name and _RE_FLAIR.search(combined_desc):
        modality = "T2-FLAIR"
        family   = "GE-CUBE FLAIR"
        rule     = "TIER1: CUBE + FLAIR keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)
    if "swan" in seq_name or "swan" in combined_desc:
        modality = "SWI"
        family   = "GE-SWAN (Susceptibility-Weighted Imaging)"
        rule     = "TIER1: SWAN keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # =========================================================================
    #  TIER 2 — DICOM ScanningSequence + SequenceVariant + numeric parameters
    #
    #  FIX: All checks now use the parsed set (scan_seq_set) so multi-valued
    #       fields like "SE\IR" correctly match both SE and IR branches.
    #       Order: EP → IR → SE → GR  (most specific first)
    # =========================================================================

    # EP (Echo Planar) — before SE/GR because EP may co-occur with SE
    if "ep" in scan_seq_set:
        if _RE_ASL.search(all_text):
            modality = "ASL"
            family   = "EPI-based ASL"
            rule     = "TIER2: ScanSeq=EP + ASL keyword"
        elif _RE_DWI.search(all_text):
            modality = "DWI"
            family   = "EPI-based DWI"
            rule     = "TIER2: ScanSeq=EP + DWI keyword"
        elif "se" in scan_seq_set:
            # SE+EP without diffusion keywords — could be DWI or ASL.
            # Use TI presence + absence of diffusion encoding as ASL hint.
            if not np.isnan(ti) and ti > 500 and not has_diffusion_encoding:
                modality = "ASL"
                family   = "SE-EPI + TI (likely ASL)"
                rule     = "TIER2: ScanSeq=SE+EP + TI>500 + no diffusion encoding"
            else:
                modality = "DWI"
                family   = "SE-EPI (likely DWI)"
                rule     = "TIER2: ScanSeq=SE+EP"
        else:
            # Pure EP with no keywords.  Instead of blindly defaulting to
            # DWI, check for ASL indicators first.
            if not np.isnan(ti) and ti > 500 and not has_diffusion_encoding:
                modality = "ASL"
                family   = "EPI + TI (likely ASL)"
                rule     = "TIER2: ScanSeq=EP + TI>500 + no diffusion encoding"
            elif has_diffusion_encoding:
                modality = "DWI"
                family   = "EPI-based DWI (diffusion metadata)"
                rule     = "TIER2: ScanSeq=EP + diffusion encoding present"
            else:
                modality = "UNKNOWN"
                family   = "EPI-based (ambiguous — no DWI or ASL evidence)"
                rule     = "TIER2: ScanSeq=EP, no keyword/metadata match"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # IR (Inversion Recovery)
    if "ir" in scan_seq_set:
        if (not np.isnan(te) and te > 100) and (not np.isnan(tr) and tr > 4000):
            modality = "T2-FLAIR"
            family   = "IR-based T2-FLAIR"
            rule     = "TIER2: ScanSeq has IR + TE>100 + TR>4000"
        elif (not np.isnan(ti) and ti > 1400):
            modality = "T2-FLAIR"
            family   = "IR-based T2-FLAIR (by TI)"
            rule     = "TIER2: ScanSeq has IR + TI>1400"
        elif (not np.isnan(ti) and 100 < ti <= 600) and (not np.isnan(te) and te < 50):
            # --- FIX: STIR is its own modality, not T2w ---
            modality = "STIR"
            family   = "STIR (Short-TI Inversion Recovery)"
            rule     = "TIER2: ScanSeq has IR + 100<TI<=600 + TE<50"
        elif (not np.isnan(ti) and 400 < ti <= 1200) and (not np.isnan(te) and te < 30):
            modality = "T1w"
            family   = "IR-based T1 (T1-FLAIR / MP-RAGE-like)"
            rule     = "TIER2: ScanSeq has IR + 400<TI<=1200 + TE<30"
        elif _RE_FLAIR.search(combined_desc):
            modality = "T2-FLAIR"
            family   = "IR-based + FLAIR keyword"
            rule     = "TIER2: ScanSeq has IR + FLAIR keyword"
        else:
            modality = "T2-FLAIR"
            family   = "IR-based (default FLAIR)"
            rule     = "TIER2: ScanSeq has IR (default)"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # SE (Spin Echo) — only if IR not present (IR+SE handled above)
    if "se" in scan_seq_set:
        if (not np.isnan(te) and te > 200) and (not np.isnan(tr) and tr > 4000):
            modality = "T2-FLAIR"
            family   = "SE-based 3D FLAIR (Siemens SPACE dark-fluid)"
            rule     = "TIER2: ScanSeq=SE + TE>200 + TR>4000"
        elif (not np.isnan(te) and 60 <= te <= 200) and (not np.isnan(tr) and tr > 2000):
            modality = "T2w"
            family   = "SE-based T2w"
            rule     = "TIER2: ScanSeq=SE + 60<=TE<=200 + TR>2000"
        elif (not np.isnan(te) and te < 30) and (not np.isnan(tr) and tr > 2000):
            modality = "PDw"
            family   = "SE-based PD-weighted"
            rule     = "TIER2: ScanSeq=SE + TE<30 + TR>2000"
        elif (not np.isnan(te) and te < 30) and (not np.isnan(tr) and tr < 1000):
            modality = "T1w"
            family   = "SE-based T1w"
            rule     = "TIER2: ScanSeq=SE + TE<30 + TR<1000"
        else:
            modality = "UNKNOWN"
            family   = "SE (unclassified)"
            rule     = "TIER2: ScanSeq=SE no param match"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # GR (Gradient Recalled Echo)
    if "gr" in scan_seq_set:
        if "mp" in seq_var:
            modality = "T1w"
            family   = "GR+MP (MPRAGE / T1-TFE)"
            rule     = "TIER2: ScanSeq=GR + SeqVar contains MP"
            return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

        if _RE_SWI.search(combined_desc):
            modality = "SWI"
            family   = "GR-based SWI (keyword)"
            rule     = "TIER2: ScanSeq=GR + SWI keyword"
            return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

        if _RE_T2STAR.search(combined_desc):
            modality = "T2star"
            family   = "GR-based T2* (keyword)"
            rule     = "TIER2: ScanSeq=GR + T2* keyword"
            return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

        if _RE_MRA.search(combined_desc):
            modality = "MRA"
            family   = "GR-based MRA/TOF"
            rule     = "TIER2: ScanSeq=GR + MRA keyword"
            return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

        if not np.isnan(te) and te < 10:
            if _RE_FLAIR.search(combined_desc):
                modality = "T2-FLAIR"
                family   = "GR-based (FLAIR keyword in description)"
                rule     = "TIER2: ScanSeq=GR + TE<10 + FLAIR keyword"
            else:
                modality = "T1w"
                family   = "GR-based T1w (spoiled GRE)"
                rule     = "TIER2: ScanSeq=GR + TE<10"
            return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

        if not np.isnan(te) and te > 15:
            modality = "T2star"
            family   = "GR-based T2*-weighted"
            rule     = "TIER2: ScanSeq=GR + TE>15"
            return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # =========================================================================
    #  TIER 3 — Protocol / Series Description keyword matching
    # =========================================================================
    if _RE_DWI.search(all_text):
        modality = "DWI"
        family   = "Keyword-matched DWI"
        rule     = "TIER3: DWI keyword in protocol/description"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_ASL.search(all_text):
        modality = "ASL"
        family   = "Keyword-matched ASL"
        rule     = "TIER3: ASL keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_MRA.search(all_text):
        modality = "MRA"
        family   = "Keyword-matched MRA"
        rule     = "TIER3: MRA keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_SWI.search(all_text):
        modality = "SWI"
        family   = "Keyword-matched SWI"
        rule     = "TIER3: SWI keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_T2STAR.search(all_text):
        modality = "T2star"
        family   = "Keyword-matched T2*"
        rule     = "TIER3: T2* keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_FLAIR.search(combined_desc):
        modality = "T2-FLAIR"
        family   = "Keyword-matched FLAIR"
        rule     = "TIER3: FLAIR keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_T1.search(combined_desc):
        modality = "T1w"
        family   = "Keyword-matched T1w"
        rule     = "TIER3: T1 keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_T2.search(combined_desc):
        modality = "T2w"
        family   = "Keyword-matched T2w"
        rule     = "TIER3: T2 keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if _RE_STIR.search(combined_desc):
        modality = "STIR"
        family   = "Keyword-matched STIR"
        rule     = "TIER3: STIR keyword"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # =========================================================================
    #  TIER 4 — Pure numeric-parameter fallback
    #
    #  FIX: TE threshold raised from <10 to <20 ms to catch short-TE SE T1
    #       sequences (e.g. "Ax T1 SE memp" with TE ~14 ms).
    #  FIX: TR range now covers the full 0–3000 ms range (was missing 50–500)
    #       to catch classic SE T1 protocols with TR ~400 ms.
    # =========================================================================
    if (not np.isnan(tr) and tr > 4000) and (not np.isnan(te) and te > 100):
        modality = "T2-FLAIR"
        family   = "Numeric FLAIR (TR>4000, TE>100)"
        rule     = "TIER4: TR>4000 + TE>100"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if not np.isnan(te) and te < 20:
        if not np.isnan(tr) and tr < 3000:
            modality = "T1w"
            family   = "Numeric T1w (short TE, appropriate TR)"
            rule     = "TIER4: TE<20 + TR<3000"
            return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    if (not np.isnan(tr) and tr > 2000) and (not np.isnan(te) and 60 <= te <= 200):
        modality = "T2w"
        family   = "Numeric T2w (long TR, medium TE)"
        rule     = "TIER4: TR>2000 + 60<=TE<=200"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # =========================================================================
    #  TIER 5 — Philips enhanced-DICOM private tags (last resort)
    # =========================================================================
    if priv_contrast == "t2" and priv_ir == "yes":
        modality = "T2-FLAIR"
        family   = "Private-tag AcquisitionContrast=T2 + IR=YES"
        rule     = "TIER5: Philips private tag T2+IR"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)
    if priv_contrast == "t1":
        modality = "T1w"
        family   = "Private-tag AcquisitionContrast=T1"
        rule     = "TIER5: Philips private tag T1"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)
    if priv_contrast == "t2":
        modality = "T2w"
        family   = "Private-tag AcquisitionContrast=T2"
        rule     = "TIER5: Philips private tag T2 (no IR)"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)
    if priv_contrast == "diffusion":
        modality = "DWI"
        family   = "Private-tag AcquisitionContrast=Diffusion"
        rule     = "TIER5: Philips private tag Diffusion"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)
    if priv_contrast == "proton_density":
        modality = "PDw"
        family   = "Private-tag AcquisitionContrast=PD"
        rule     = "TIER5: Philips private tag PD"
        return _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)

    # =========================================================================
    #  Nothing matched
    # =========================================================================
    return _result("UNKNOWN", "Unclassified", "NO RULE MATCHED",
                   is_derived, is_subtraction, is_ce, dwi_sub, is_localizer)


def _result(modality, family, rule, is_derived, is_subtraction, is_ce, dwi_sub="", is_localizer=False):
    return {
        "classified_modality":    modality,
        "sequence_family":        family,
        "classification_rule":    rule,
        "is_derived":             is_derived,
        "is_subtraction":         is_subtraction,
        "is_contrast_enhanced":   is_ce,
        "dwi_sub_type":           dwi_sub,
        "is_localizer":           is_localizer,
    }


# ============================================================================
# 4.  RUN ON DATAFRAME
# ============================================================================
def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply classifier to every row and return a new dataframe with added columns."""
    results = df.apply(classify_row, axis=1, result_type="expand")
    out = pd.concat([df, results], axis=1)
    return out


# ============================================================================
# 5.  REPORTING
# ============================================================================
def print_report(df: pd.DataFrame):
    """Print a human-readable summary of classification results."""

    total = len(df)
    print("=" * 72)
    print(f"  MRI MODALITY CLASSIFICATION REPORT  ({total:,} series)")
    print("=" * 72)

    print("\n1. CLASSIFIED MODALITY DISTRIBUTION")
    print("-" * 44)
    dist = df["classified_modality"].value_counts()
    for mod, cnt in dist.items():
        pct = cnt / total * 100
        print(f"   {mod:<14s}  {cnt:>5,d}  ({pct:5.1f}%)")

    print("\n2. CLASSIFICATION RULE TIER USAGE")
    print("-" * 44)
    df["_tier"] = df["classification_rule"].str.extract(r"(TIER\d)")[0]
    tier_dist = df["_tier"].value_counts().sort_index()
    for tier, cnt in tier_dist.items():
        pct = cnt / total * 100
        print(f"   {tier}  {cnt:>5,d}  ({pct:5.1f}%)")

    print("\n3. SEQUENCE FAMILY BREAKDOWN")
    print("-" * 62)
    fam_dist = df["sequence_family"].value_counts()
    for fam, cnt in fam_dist.items():
        pct = cnt / total * 100
        print(f"   {fam:<50s}  {cnt:>5,d}  ({pct:5.1f}%)")

    # DWI sub-types
    dwi_mask = df["classified_modality"] == "DWI"
    if dwi_mask.any():
        print("\n3b. DWI SUB-TYPE BREAKDOWN")
        print("-" * 44)
        dwi_sub = df.loc[dwi_mask, "dwi_sub_type"].replace("", "(standard)").value_counts()
        for sub, cnt in dwi_sub.items():
            print(f"   {sub:<30s}  {cnt:>5,d}")

    print("\n4. FLAGS")
    print("-" * 44)
    print(f"   Derived images:            {df['is_derived'].sum():>5,d}")
    print(f"   Subtraction images:         {df['is_subtraction'].sum():>5,d}")
    print(f"   Contrast-enhanced:          {df['is_contrast_enhanced'].sum():>5,d}")
    print(f"   Localizer/survey:           {df['is_localizer'].sum():>5,d}")

    if "predicted_mr_contrast_weighting" in df.columns:
        print("\n5. COMPARISON WITH EXISTING 'predicted_mr_contrast_weighting'")
        print("-" * 58)

        label_map = {"T1": "T1w", "T2": "T2w", "FLAIR": "T2-FLAIR", "MRA": "MRA",
                     "DWI": "DWI", "DTI": "DWI",
                     "ASL": "ASL", "SWI": "SWI", "T2*": "T2star", "PD": "PDw"}
        df["_old_mapped"] = df["predicted_mr_contrast_weighting"].map(label_map).fillna("OTHER")

        agree = (df["classified_modality"] == df["_old_mapped"]).sum()
        disagree = total - agree
        print(f"   Agreement:    {agree:>5,d}  ({agree / total * 100:5.1f}%)")
        print(f"   Disagreement: {disagree:>5,d}  ({disagree / total * 100:5.1f}%)")

        if disagree > 0:
            print("\n   Disagreement detail (old -> new):")
            mismatches = df[df["classified_modality"] != df["_old_mapped"]]
            cross = pd.crosstab(
                mismatches["_old_mapped"],
                mismatches["classified_modality"],
                margins=False,
            )
            print(cross.to_string().replace("\n", "\n   "))

        df.drop(columns=["_old_mapped", "_tier"], inplace=True, errors="ignore")
    else:
        df.drop(columns=["_tier"], inplace=True, errors="ignore")

    print("\n6. SAMPLE ROWS PER MODALITY")
    print("-" * 72)
    show_cols = [
        "classified_modality", "sequence_family", "classification_rule",
        "ProtocolName", "ScanningSequence", "SequenceVariant",
        "TR_ms", "TE_ms", "TI_ms", "FlipAngle",
        "is_derived", "is_contrast_enhanced", "dwi_sub_type",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    for mod in sorted(df["classified_modality"].unique()):
        subset = df[df["classified_modality"] == mod]
        print(f"\n   -- {mod} (n={len(subset)}) --")
        sample = subset[show_cols].drop_duplicates().head(3)
        for _, srow in sample.iterrows():
            parts = [f"{c}={srow[c]}" for c in show_cols if c != "classified_modality"]
            print(f"      " + " | ".join(parts))

    print("\n" + "=" * 72)
    print("  END OF REPORT")
    print("=" * 72)


# ============================================================================
# 6.  MAIN
# ============================================================================
def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "nvidia_1000_mri_raw_metadata.csv"
    )
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  -> {len(df):,} rows, {len(df.columns)} columns\n")

    df = classify_dataframe(df)

    print_report(df)

    output_path = Path(str(input_path).replace(".csv", "_classified.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nClassified CSV saved -> {output_path}")

    summary_cols = [
        "AccessionNumber", "SeriesNumber", "ProtocolName", "SeriesDescription",
        "Manufacturer", "ScanningSequence", "SequenceVariant",
        "PulseSequenceName", "SequenceName",
        "TR_ms", "TE_ms", "TI_ms", "FlipAngle", "EchoTrainLength",
        "FieldStrength_T", "MRAcquisitionType",
        "classified_modality", "sequence_family", "classification_rule",
        "is_derived", "is_subtraction", "is_contrast_enhanced", "is_localizer", "dwi_sub_type",
        "predicted_mr_contrast_weighting",
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    summary_path = Path(str(input_path).replace(".csv", "_classification_summary.csv"))
    df[summary_cols].to_csv(summary_path, index=False)
    print(f"Summary CSV saved   -> {summary_path}")


if __name__ == "__main__":
    main()