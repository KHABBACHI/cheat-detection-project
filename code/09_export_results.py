"""
Copies the lightweight artefacts needed to audit the study (split, windows, predictions,
selected settings, evaluation and timing outputs) into the repository folder results/.

Usage: python code/09_export_results.py --projet C:\\projet_triche --depot <path/to/this/repository>
"""
import argparse
import shutil
from pathlib import Path

MOTIFS = [
    "split.json",
    "crops_index.csv",
    "fenetres/*.csv",
    "modeles/*/*/cv.csv",
    "modeles/*/*/choix.json",
    "modeles/*/*/oof_dev.csv",
    "modeles/*/*/predictions_test.csv",
    "modeles/*/*/config.json",
    "modeles/c5_L16/transformer/graine*.pt",
    "modeles/c5_L16/transformer/scaler_dev.npz",
    "evaluation/*",
    "chronometrage/*",
    "extraction/config_extraction.txt",
    "extraction/temps_yolo.csv",
    "embeddings/config_dinov2.txt",
    "embeddings/temps_dinov2.csv",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projet", required=True)
    ap.add_argument("--depot", required=True)
    a = ap.parse_args()
    src = Path(a.projet) / "artefacts"
    dst = Path(a.depot) / "results"
    n, taille = 0, 0
    for motif in MOTIFS:
        for f in sorted(src.glob(motif)):
            if f.is_file():
                cible = dst / f.relative_to(src)
                cible.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, cible)
                n += 1
                taille += f.stat().st_size
    print(f"{n} files copied to {dst} ({taille / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
