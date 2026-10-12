"""
Indexes the crops and splits the data BY SUBJECT before any window is built:
test subjects given on the command line, remaining subjects in 4 cross-validation folds.

Usage: python code/02_split.py --test Muath1 XiYin1 zach1
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_EXTR = RACINE / "artefacts" / "extraction"
DOSSIER_ART = RACINE / "artefacts"
GRAINE = 42
NB_FOLDS = 4


def label_depuis_classe(nom):
    n = (nom or "").strip().lower()
    if not n:
        return None
    if n.startswith(("no", "not", "non")):
        return 0
    return 1 if ("cheat" in n or "triche" in n) else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", nargs="+", required=True,
                   help="etudiants du test final (non utilises pour entrainer YOLOv5)")
    a = p.parse_args()

    csvs = sorted(DOSSIER_EXTR.glob("*.csv"))
    csvs = [c for c in csvs if c.stem != "temps_yolo"]
    etudiants = [c.stem for c in csvs]
    if not csvs:
        sys.exit(f"[!] Aucun CSV dans {DOSSIER_EXTR} : lancez d'abord 01_extraction_crops.py")

    inconnus = [e for e in a.test if e not in etudiants]
    if inconnus:
        sys.exit(f"[!] Etudiants inconnus : {inconnus}\n    Noms disponibles : {etudiants}")
    if len(etudiants) < 15:
        print(f"[!] Attention : seulement {len(etudiants)} videos extraites sur 15.")

    test = sorted(a.test)
    dev = sorted(e for e in etudiants if e not in test)
    rng = random.Random(GRAINE)
    melange = dev[:]
    rng.shuffle(melange)
    folds = [sorted(melange[i::NB_FOLDS]) for i in range(NB_FOLDS)]
    partition = {e: "test" for e in test}
    for i, f in enumerate(folds):
        for e in f:
            partition[e] = f"fold{i}"

    lignes = []
    stats = defaultdict(lambda: {"images": 0, "crops": 0, "triche": 0, "normal": 0})
    for c in csvs:
        with open(c, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                e = r["etudiant"]
                y = label_depuis_classe(r["classe_yolo"])
                s = stats[e]
                s["images"] += 1
                if y is not None:
                    s["crops"] += 1
                    s["triche" if y else "normal"] += 1
                lignes.append({
                    "etudiant": e,
                    "partition": partition[e],
                    "frame": int(r["frame"]),
                    "temps_s": r["temps_s"],
                    "classe_yolo": r["classe_yolo"],
                    "conf_yolo": r["conf"],
                    "label": "" if y is None else y,
                    "source_label": "yolo_pseudo",
                    "fichier": r["fichier"],
                })
    lignes.sort(key=lambda l: (l["etudiant"], l["frame"]))

    with open(DOSSIER_ART / "crops_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)

    for s in stats.values():
        s["taux_triche"] = round(s["triche"] / max(1, s["crops"]), 3)

    split = {
        "cree_le": datetime.now().isoformat(timespec="seconds"),
        "graine": GRAINE,
        "unite_de_split": "etudiant (= video webcam OEP = session)",
        "split_fait_avant_les_fenetres": True,
        "fps_echantillonnage_crops": 5,
        "source_des_etiquettes": "yolo_pseudo - A REVOIR (correction 2)",
        "test_final": test,
        "test_final_exclu_entrainement_yolov5": True,
        "dev": dev,
        "folds_validation_croisee": folds,
        "regles": [
            "Le TEST n'est utilise qu'une seule fois, a la fin, apres avoir fige "
            "architecture, hyperparametres, checkpoint et seuil.",
            "Tous ces choix se font par validation croisee sur 'dev' uniquement.",
            "StandardScaler et autres normalisations : ajustes sur les etudiants "
            "d'entrainement de chaque fold uniquement.",
            "Les fenetres sont construites a l'interieur de chaque etudiant, apres le split.",
        ],
        "stats_par_etudiant": dict(stats),
    }
    (DOSSIER_ART / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False),
                                            encoding="utf-8")

    print("=" * 72)
    print(f"{'etudiant':<16}{'images':>8}{'crops':>8}{'triche':>8}{'normal':>8}{'taux':>7}   partition")
    for e in sorted(stats, key=lambda e: (partition[e], e)):
        s = stats[e]
        print(f"{e:<16}{s['images']:>8}{s['crops']:>8}{s['triche']:>8}{s['normal']:>8}"
              f"{s['taux_triche']:>7}   {partition[e]}")
    tot = lambda part: sum(stats[e]["crops"] for e in stats if partition[e] == part)
    print(f"\nTEST final : {test}  ({tot('test')} crops)")
    for i, f in enumerate(folds):
        print(f"fold{i}      : {f}  ({tot(f'fold{i}')} crops)")
    print(f"\nEcrit : {DOSSIER_ART / 'crops_index.csv'}  et  {DOSSIER_ART / 'split.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
