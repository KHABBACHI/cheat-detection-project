"""
Builds temporal windows inside each subject, after the split, for all window configurations.

Usage: python code/04_fenetres.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_EMB = RACINE / "artefacts" / "embeddings"
DOSSIER_FEN = RACINE / "artefacts" / "fenetres"
SPLIT = RACINE / "artefacts" / "split.json"

FPS_VIDEO = 25
PAS_EXTRACTION = 5
TROU_TOLERE = 2
CONFIGS = [(5, 8), (5, 16), (5, 32), (5, 48), (1, 8), (1, 16)]


def segments_continus(frames, pas):
    debut = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] - frames[i - 1] > TROU_TOLERE * pas:
            yield debut, i
            debut = i


def fenetres_etudiant(etu, partition, frames, labels, cadence, L):
    pas = FPS_VIDEO // cadence
    garde = np.where(frames % pas == 0)[0]
    f, y = frames[garde], labels[garde]
    stride = max(1, L // 2)
    sortie = []
    for a, b in segments_continus(f, pas):
        for s in range(a, b - L + 1, stride):
            idx = garde[s:s + L]
            ly = y[s:s + L]
            frac = float(ly.mean())
            sortie.append({
                "etudiant": etu,
                "partition": partition,
                "frame_debut": int(f[s]),
                "frame_fin": int(f[s + L - 1]),
                "duree_s": round((f[s + L - 1] - f[s]) / FPS_VIDEO + 1 / cadence, 2),
                "part_triche": round(frac, 4),
                "label_majorite": int(frac >= 0.5),
                "label_derniere": int(ly[-1]),
                "indices": " ".join(map(str, idx)),
            })
    return sortie


def main():
    if not SPLIT.exists():
        sys.exit("[!] split.json introuvable : lancez 02_split.py")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    partition = {e: "test" for e in split["test_final"]}
    for i, fold in enumerate(split["folds_validation_croisee"]):
        for e in fold:
            assert e not in partition, f"{e} est dans deux partitions !"
            partition[e] = f"fold{i}"

    donnees = {}
    for e in sorted(partition):
        p = DOSSIER_EMB / f"{e}.npz"
        if not p.exists():
            sys.exit(f"[!] Embeddings manquants pour {e} : relancez 03_embeddings.py")
        z = np.load(p)
        ordre = np.argsort(z["frame"])
        assert (ordre == np.arange(len(ordre))).all(), f"{e}: frames non triees"
        donnees[e] = (z["frame"].astype(np.int64), z["label"].astype(np.int64))

    DOSSIER_FEN.mkdir(parents=True, exist_ok=True)
    resume = []
    print("=" * 78)
    print(f"{'config':<10}{'partition':<10}{'fenetres':>9}{'triche':>8}{'taux':>7}"
          f"{'duree med (s)':>15}{'fenetres mixtes':>17}")
    for cadence, L in CONFIGS:
        toutes = []
        for e, (frames, labels) in donnees.items():
            toutes += fenetres_etudiant(e, partition[e], frames, labels, cadence, L)
        nom = f"c{cadence}_L{L}"
        with open(DOSSIER_FEN / f"{nom}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id"] + list(toutes[0].keys()))
            w.writeheader()
            for i, r in enumerate(toutes):
                w.writerow({"id": f"{nom}_{i:06d}", **r})

        par_part = defaultdict(list)
        for r in toutes:
            par_part[r["partition"]].append(r)
        for part in sorted(par_part):
            rs = par_part[part]
            n_tri = sum(r["label_majorite"] for r in rs)
            mixtes = sum(0 < r["part_triche"] < 1 for r in rs)
            duree = float(np.median([r["duree_s"] for r in rs]))
            resume.append({"config": nom, "cadence_img_s": cadence, "L": L, "partition": part,
                           "fenetres": len(rs), "triche": n_tri,
                           "taux_triche": round(n_tri / len(rs), 3),
                           "duree_mediane_s": duree, "fenetres_mixtes": mixtes})
            print(f"{nom:<10}{part:<10}{len(rs):>9}{n_tri:>8}{n_tri / len(rs):>7.2f}"
                  f"{duree:>15.1f}{mixtes:>17}")
        print("-" * 78)

    with open(DOSSIER_FEN / "resume.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resume[0].keys()))
        w.writeheader()
        w.writerows(resume)

    verif = Counter((r["config"], r["partition"]) for r in resume)
    assert all(v == 1 for v in verif.values())
    print("Verification : chaque etudiant est dans une seule partition, "
          "chaque fenetre ne contient qu'un seul etudiant. OK")
    print("Fichiers ecrits dans", DOSSIER_FEN)
    print("=" * 78)


if __name__ == "__main__":
    main()
