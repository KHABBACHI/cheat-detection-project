"""
Computes frozen DINOv2 (ViT-B/14) CLS embeddings for every crop. No standardisation here.

Usage: python code/03_embeddings.py [--modele dinov2_vitb14] [--etudiant NAME]
"""
import argparse
import csv
import platform
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

RACINE = Path(__file__).resolve().parent.parent
INDEX = RACINE / "artefacts" / "crops_index.csv"
DOSSIER_EMB = RACINE / "artefacts" / "embeddings"
TAILLE = 224

PRETRAITEMENT = transforms.Compose([
    transforms.Resize((TAILLE, TAILLE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def lire_index():
    par_etu = defaultdict(list)
    with open(INDEX, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fichier"]:
                par_etu[r["etudiant"]].append((int(r["frame"]), int(r["label"]), r["fichier"]))
    for e in par_etu:
        par_etu[e].sort()
    return par_etu


@torch.no_grad()
def traiter(modele, etu, lignes, batch):
    sortie = DOSSIER_EMB / f"{etu}.npz"
    if sortie.exists():
        print(f"[=] {etu} deja fait, on saute.")
        return
    embs, temps = [], []
    t0 = time.time()
    for i in range(0, len(lignes), batch):
        lot = lignes[i:i + batch]
        x = torch.stack([PRETRAITEMENT(Image.open(RACINE / f).convert("RGB")) for _, _, f in lot])
        t = time.perf_counter()
        z = modele(x)
        temps.append((time.perf_counter() - t) / len(lot))
        embs.append(z.numpy().astype(np.float16))
        fait = i + len(lot)
        if (i // batch) % 20 == 0:
            vit = fait / (time.time() - t0)
            print(f"    {etu}: {fait}/{len(lignes)} | {vit:.1f} img/s | "
                  f"reste ~{(len(lignes) - fait) / vit / 60:.0f} min")
    emb = np.concatenate(embs)
    tmp = sortie.with_name(sortie.stem + "_tmp.npz")
    np.savez(tmp, emb=emb,
             frame=np.array([l[0] for l in lignes], dtype=np.int32),
             label=np.array([l[1] for l in lignes], dtype=np.int8))
    tmp.replace(sortie)
    ms = 1000 * float(np.mean(temps))
    print(f"[ok] {etu}: {emb.shape[0]} crops -> vecteurs de dimension {emb.shape[1]} | "
          f"DINOv2 {ms:.0f} ms/image (par lot de {batch})")
    with open(DOSSIER_EMB / "temps_dinov2.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([etu, emb.shape[0], round(ms, 2), batch])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modele", default="dinov2_vitb14",
                   help="dinov2_vitb14 (768 dim, defaut = article) ou dinov2_vits14 (384 dim)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--etudiant", default=None)
    a = p.parse_args()

    if not INDEX.exists():
        sys.exit(f"[!] {INDEX} introuvable : lancez d'abord 02_split.py")
    DOSSIER_EMB.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    print("Chargement de", a.modele, "...")
    modele = torch.hub.load("facebookresearch/dinov2", a.modele, verbose=False)
    modele.eval()
    for prm in modele.parameters():
        prm.requires_grad = False

    (DOSSIER_EMB / "config_dinov2.txt").write_text(
        f"modele={a.modele}\ngele=oui (aucun fine-tuning)\nsortie=token CLS\n"
        f"pretraitement=Resize({TAILLE}x{TAILLE}) + normalisation ImageNet\n"
        f"normalisation_StandardScaler=non (faite par fold, sur le train uniquement)\n"
        f"batch={a.batch}\npython={platform.python_version()}\ntorch={torch.__version__}\n"
        f"cpu={platform.processor()}\nthreads={torch.get_num_threads()}\n"
        f"cuda={torch.cuda.is_available()}\n", encoding="utf-8")

    par_etu = lire_index()
    etudiants = [a.etudiant] if a.etudiant else sorted(par_etu)
    for e in etudiants:
        if e not in par_etu:
            sys.exit(f"[!] Etudiant inconnu : {e}. Disponibles : {sorted(par_etu)}")
        print(f"[>] {e}: {len(par_etu[e])} crops")
        traiter(modele, e, par_etu[e], a.batch)
    print("\nTermine. Embeddings dans", DOSSIER_EMB)


if __name__ == "__main__":
    main()
