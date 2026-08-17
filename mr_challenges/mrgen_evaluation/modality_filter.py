"""
Modalite filtresi (Admin kararı, 2026)
=======================================

MR-RATE ground truth dosyaları şu formatta:
    {study_uid}_{sequence_name}.nii.gz
örn:  UWCMTFCZ47_t1w-raw-axi.nii.gz
      UWCMTFCZ47_flair-raw-sag.nii.gz

`sequence_name` = "{modality}-raw-{plane}" şeklinde, modality her zaman
ilk parça (t1w / t2w / flair / swi / mra / dwi / adc / ...).

KARAR: Değerlendirme SADECE T1w/T2w/FLAIR/SWI ile sınırlandırılacak.
Ground truth verisinin kendisi DEĞİŞMİYOR — sadece bu modül, hangi
entry'lerin skorlamaya (ve "missing output = lowest score" cezasına)
dahil edileceğini filtreliyor.

prompts.json içindeki "input_image_name" alanı ground truth dosya adının
uzantısız hali ile birebir aynı (örn. "UWCMTFCZ47_t1w-raw-axi").
"""

from __future__ import annotations

# Değerlendirmeye dahil edilen modaliteler (admin kararı).
# Küçük harfe normalize edilmiş şekilde tutuluyor.
ALLOWED_MODALITIES = frozenset({"t1w", "t2w", "flair", "swi"})


def extract_modality(input_image_name: str) -> str:
    """
    'UWCMTFCZ47_t1w-raw-axi' -> 't1w'
    'UWCMTFCZ47_flair-raw-sag' -> 'flair'

    Format: {study_uid}_{modality}-raw-{plane}
    study_uid içinde '_' geçmediği için sağdan değil, ilk '_' sonrasını
    alıp '-' ile bölüyoruz.
    """
    name = input_image_name.strip()
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        name = name[: -len(".nii")]

    if "_" not in name:
        raise ValueError(f"Beklenmeyen dosya adı formatı (alt çizgi yok): {input_image_name!r}")

    # study_uid'den sonraki ilk parça sequence_name'dir.
    # study_uid'ler gözlemlenen örneklerde alfanumerik ve '_' içermiyor,
    # bu yüzden ilk '_' güvenle ayraç olarak kullanılabilir.
    _study_uid, sequence_name = name.split("_", 1)

    if "-" not in sequence_name:
        raise ValueError(
            f"Beklenmeyen sequence_name formatı (tire yok): {sequence_name!r} "
            f"(kaynak: {input_image_name!r})"
        )

    modality = sequence_name.split("-", 1)[0].lower()
    return modality


def is_scored_modality(input_image_name: str) -> bool:
    """Bu entry değerlendirmeye dahil edilmeli mi?"""
    try:
        modality = extract_modality(input_image_name)
    except ValueError:
        # Formatı çözemediğimiz bir isim varsa, güvenli taraf: skorlama
        # dışında bırak ama sessizce yutma — çağıran taraf loglasın.
        return False
    return modality in ALLOWED_MODALITIES


def filter_scored_entries(prompt_entries: list[dict]) -> list[dict]:
    """
    prompts.json'dan yüklenen entry listesini alır, sadece skorlanacak
    modalitelere ait olanları döndürür.

    Her entry şu formatta bekleniyor:
        {"input_image_name": "...", "report": "..."}
    """
    kept = []
    for entry in prompt_entries:
        name = entry.get("input_image_name")
        if name is None:
            raise KeyError(f"Entry'de 'input_image_name' yok: {entry}")
        if is_scored_modality(name):
            kept.append(entry)
    return kept


if __name__ == "__main__":
    # Küçük bir doğrulama — gerçek örneklerle (bir önceki mesajdaki çıktıdan)
    test_cases = {
        "327DO56MXY_t1w-raw-axi": "t1w",
        "327DO56MXY_t1w-raw-obl": "t1w",
        "3BLIV6OVJC_t2w-raw-axi": "t2w",
        "327DO56MXY_flair-raw-sag": "flair",
        "4S5APY6VMJ_swi-raw-axi": "swi",
        "SOMEID123_mra-raw-axi": "mra",
        "SOMEID123_dwi-raw-axi": "dwi",
        "SOMEID123_adc-raw-axi": "adc",
    }
    for name, expected in test_cases.items():
        got = extract_modality(name)
        status = "OK" if got == expected else "HATA"
        scored = is_scored_modality(name)
        print(f"[{status}] {name!r} -> modality={got!r} (beklenen={expected!r}), scored={scored}")

    assert extract_modality("327DO56MXY_t1w-raw-axi") == "t1w"
    assert is_scored_modality("327DO56MXY_t1w-raw-axi") is True
    assert is_scored_modality("SOMEID123_mra-raw-axi") is False
    assert is_scored_modality("SOMEID123_dwi-raw-axi") is False
    assert is_scored_modality("SOMEID123_adc-raw-axi") is False

    entries = [
        {"input_image_name": "A_t1w-raw-axi", "report": "..."},
        {"input_image_name": "A_mra-raw-axi", "report": "..."},
        {"input_image_name": "B_flair-raw-sag", "report": "..."},
    ]
    kept = filter_scored_entries(entries)
    assert len(kept) == 2, kept
    print("\nTüm testler geçti.")
