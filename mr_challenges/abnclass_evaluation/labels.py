"""
Label columns for the MR-RATE abnormality-classification track.

NeuroVFM taxonomy, restricted to the pathologies with >=10 positive cases in the
held-out batch28 cohort (32 of 74). Labels below that threshold were
dropped from scoring: with single-digit positives a macro average is dominated by
sampling noise rather than model quality.

These keys must match both the ground-truth CSV columns and the keys
participants emit in their predictions JSON.
"""

LABEL_COLS = [
    "cavernous_malformation_cavernoma",
    "acute_ischemic_stroke",
    "chronic_ischemic_stroke",
    "small_vessel_ischemic_disease",
    "lacunar_stroke",
    "intracranial_hemorrhage",
    "intraparenchymal_hemorrhage",
    "multiple_sclerosis",
    "arachnoid_cyst",
    "pineal_cyst",
    "brain_tumor",
    "head_neck_tumor",
    "spine_tumor",
    "spinal_degenerative_changes",
    "glioma",
    "brain_metastasis",
    "meningioma",
    "intraventricular_tumor",
    "pituitary_tumor",
    "rathkes_cleft_cyst",
    "chiari_malformation",
    "cerebral_atrophy",
    "brain_mass_effect",
    "encephalomalacia_gliosis",
    "cerebral_edema",
    "cavum_septum_pellucidum",
    "mega_cisterna_magna",
    "ventriculomegaly",
    "hydrocephalus_ex_vacuo",
    "craniotomy_craniectomy",
    "tumor_resection_cavity",
    "postsurgical_changes",
]
