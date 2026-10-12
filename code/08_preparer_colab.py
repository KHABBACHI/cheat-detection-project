"""
Packs the files needed for the GPU timing on Google Colab into projet_triche_colab.zip.

Usage: python code/08_preparer_colab.py
"""
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
VIDEO = "zach1"
CONFIGS = ["c5_L16", "c1_L8"]

fichiers = [
    RACINE / "videos" / f"{VIDEO}.avi",
    RACINE / "weights" / "best.pt",
    RACINE / "code" / "05_entrainement.py",
    RACINE / "code" / "07_chronometrage.py",
]
for c in CONFIGS:
    d = RACINE / "artefacts" / "modeles" / c / "transformer"
    fichiers += [d / "graine0.pt", d / "scaler_dev.npz", d / "choix.json"]

manquants = [f for f in fichiers if not f.exists()]
if manquants:
    raise SystemExit("[!] Fichiers manquants :\n  " + "\n  ".join(map(str, manquants)))

sortie = RACINE / "projet_triche_colab.zip"
with zipfile.ZipFile(sortie, "w", zipfile.ZIP_DEFLATED) as z:
    for f in fichiers:
        arc = Path("projet_triche") / f.relative_to(RACINE)
        z.write(f, arc.as_posix())
        print(f"  + {arc.as_posix()}  ({f.stat().st_size / 1e6:.1f} Mo)")
print(f"\nCree : {sortie}  ({sortie.stat().st_size / 1e6:.0f} Mo)")
print("Envoyez ce fichier a la racine de votre Google Drive (Mon Drive).")
