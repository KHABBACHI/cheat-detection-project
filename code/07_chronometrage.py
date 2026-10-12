"""
End-to-end, frame-by-frame timing of the full pipeline (decoding, YOLOv5, DINOv2,
Transformer, decision, alert) on CPU or GPU.

Usage: python code/07_chronometrage.py --yolov5 <path/to/yolov5> --config c5_L16
"""
import argparse
import csv
import importlib
import json
import os
import platform
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import cv2
import pathlib

if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
ENT = importlib.import_module("05_entrainement")

DOSSIER_VIDEOS = RACINE / "videos"
POIDS_YOLO = RACINE / "weights" / "best.pt"
DOSSIER_MOD = RACINE / "artefacts" / "modeles"
DOSSIER_SORTIE = RACINE / "artefacts" / "chronometrage"
APPAREIL = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ECHAUFFEMENT = 20

PRETRAITEMENT = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def synchro():
    if APPAREIL.type == "cuda":
        torch.cuda.synchronize()


def materiel():
    info = {"os": platform.platform(), "processeur": platform.processor(),
            "coeurs_logiques": os.cpu_count(), "threads_torch": torch.get_num_threads(),
            "python": platform.python_version(), "torch": torch.__version__,
            "opencv": cv2.__version__, "appareil": str(APPAREIL),
            "gpu": torch.cuda.get_device_name(0) if APPAREIL.type == "cuda" else "aucun"}
    try:
        import psutil
        info["ram_go"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["ram_go"] = "psutil absent"
    return info


def charger_modeles(dossier_yolov5, config):
    yolo = torch.hub.load(str(dossier_yolov5), "custom", path=str(POIDS_YOLO), source="local", verbose=False)
    yolo.conf = 0.25
    yolo.to(APPAREIL).eval()

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False).to(APPAREIL).eval()

    d = DOSSIER_MOD / config / "transformer"
    L = int(config.split("_L")[1])
    hp = {**ENT.HP_BASE, **ENT.VARIANTES["transformer"]}
    trans = ENT.TransformerTemporel(768, L, hp)
    trans.load_state_dict(torch.load(d / "graine0.pt", map_location="cpu"))
    trans.to(APPAREIL).eval()
    sc = np.load(d / "scaler_dev.npz")
    mu = torch.tensor(sc["mu"], dtype=torch.float32, device=APPAREIL)
    sd = torch.tensor(sc["sd"], dtype=torch.float32, device=APPAREIL)
    seuil = json.loads((d / "choix.json").read_text(encoding="utf-8"))["seuil"]
    return yolo, dino, trans, mu, sd, seuil, L


def stats(v):
    v = np.array(v) * 1000
    return {"moyenne_ms": round(float(v.mean()), 1), "ecart_type_ms": round(float(v.std()), 1),
            "mediane_ms": round(float(np.median(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolov5", required=True)
    ap.add_argument("--config", default="c1_L8")
    ap.add_argument("--video", default="zach1")
    ap.add_argument("--secondes", type=float, default=180)
    a = ap.parse_args()

    cadence = int(a.config[1:].split("_L")[0])
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    print("Appareil :", APPAREIL, "|", materiel()["gpu"])
    print("Chargement des modeles ...")
    yolo, dino, trans, mu, sd, seuil, L = charger_modeles(a.yolov5, a.config)
    stride = max(1, L // 2)

    cap = cv2.VideoCapture(str(DOSSIER_VIDEOS / f"{a.video}.avi"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pas = int(round(fps / cadence))
    n_max = int(a.secondes * cadence)
    print(f"Video {a.video} | {cadence} img/s analysees (1 frame sur {pas}) | fenetre {L} crops "
          f"| Transformer toutes les {stride} images | {n_max} images a traiter")

    tampon = deque(maxlen=L)
    depuis_derniere = 0
    lignes, alertes = [], 0
    i_frame, n = -1, 0
    t_debut_mesure = None
    while n < n_max:
        t0 = time.perf_counter()
        ok = True
        for _ in range(pas - 1 if n else 0):
            ok = cap.grab(); i_frame += 1
            if not ok:
                break
        ok, img = cap.read() if ok else (False, None)
        i_frame += 1
        if not ok:
            break
        t1 = time.perf_counter()

        det = yolo(img[..., ::-1], size=640).xyxy[0]
        synchro(); t2 = time.perf_counter()

        t_pre = t_dino = t_trans = t_alerte = 0.0
        boite = None
        if det is not None and len(det):
            x1, y1, x2, y2 = [int(v) for v in det[det[:, 4].argmax(), :4].tolist()]
            if x2 > max(0, x1) and y2 > max(0, y1):
                boite = (x1, y1, x2, y2)
        if boite:
            x1, y1, x2, y2 = boite
            crop = img[max(0, y1):y2, max(0, x1):x2][..., ::-1]
            x = PRETRAITEMENT(Image.fromarray(np.ascontiguousarray(crop))).unsqueeze(0).to(APPAREIL)
            synchro(); t3 = time.perf_counter()
            z = dino(x)[0]
            synchro(); t4 = time.perf_counter()
            t_pre, t_dino = t3 - t2, t4 - t3
            tampon.append(z)
            depuis_derniere += 1
            if len(tampon) == L and depuis_derniere >= stride:
                depuis_derniere = 0
                seq = ((torch.stack(list(tampon)) - mu) / sd).unsqueeze(0)
                p = torch.sigmoid(trans(seq)).item()
                synchro(); t5 = time.perf_counter()
                alerte = p >= seuil
                alertes += int(alerte)
                t6 = time.perf_counter()
                t_trans, t_alerte = t5 - t4, t6 - t5
        t_fin = time.perf_counter()
        n += 1
        if n == ECHAUFFEMENT + 1:
            t_debut_mesure = t0
        lignes.append({"image": n, "frame": i_frame, "lecture": t1 - t0, "yolo": t2 - t1,
                       "pretraitement": t_pre, "dinov2": t_dino, "transformer": t_trans,
                       "alerte": t_alerte, "total": t_fin - t0})
        if n % 50 == 0:
            moy = np.mean([l["total"] for l in lignes[-50:]]) * 1000
            print(f"  {n}/{n_max} images | {moy:.0f} ms/image (50 dernieres)")
    cap.release()
    t_fin_mesure = time.perf_counter()

    mes = lignes[ECHAUFFEMENT:]
    if not mes:
        sys.exit("[!] Pas assez d'images traitees.")
    duree_mesure = t_fin_mesure - t_debut_mesure
    debit = len(mes) / duree_mesure
    budget_ms = 1000 / cadence
    avec_trans = [l for l in mes if l["transformer"] > 0]
    res = {
        "config": a.config, "cadence_img_s": cadence, "fenetre_crops": L,
        "duree_fenetre_s": L / cadence, "pas_transformer_images": stride,
        "nouvelle_fenetre_toutes_les_s": stride / cadence,
        "video": a.video, "images_mesurees": len(mes), "images_echauffement_exclues": ECHAUFFEMENT,
        "execution": "sequentielle, une image a la fois (batch 1), sans recouvrement entre etapes",
        "budget_par_image_ms": budget_ms,
        "latence_par_etape": {k: stats([l[k] for l in mes]) for k in
                              ("lecture", "yolo", "pretraitement", "dinov2", "total")},
        "latence_transformer_plus_alerte_par_fenetre": stats(
            [l["transformer"] + l["alerte"] for l in avec_trans]) if avec_trans else None,
        "debit_soutenu_img_s": round(debit, 2),
        "facteur_temps_reel": round(debit / cadence, 2),
        "temps_reel": bool(debit >= cadence),
        "part_images_au_dessus_du_budget": round(float(np.mean([l["total"] * 1000 > budget_ms for l in mes])), 3),
        "alertes_emises": alertes, "materiel": materiel(),
    }
    nom = f"{'gpu' if APPAREIL.type == 'cuda' else 'cpu'}_{a.config}"
    (DOSSIER_SORTIE / f"{nom}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(DOSSIER_SORTIE / f"{nom}_par_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)

    print("\n" + "=" * 74)
    print(f"CHRONOMETRAGE {nom}  |  {res['materiel']['processeur']}  |  GPU : {res['materiel']['gpu']}")
    print(f"{'etape':<16}{'moyenne':>10}{'mediane':>10}{'p95':>10}   (ms par image)")
    for k, v in res["latence_par_etape"].items():
        print(f"{k:<16}{v['moyenne_ms']:>10}{v['mediane_ms']:>10}{v['p95_ms']:>10}")
    if res["latence_transformer_plus_alerte_par_fenetre"]:
        v = res["latence_transformer_plus_alerte_par_fenetre"]
        print(f"{'transf.+alerte':<16}{v['moyenne_ms']:>10}{v['mediane_ms']:>10}{v['p95_ms']:>10}"
              f"   (par fenetre, toutes les {stride} images)")
    print(f"\nBudget par image : {budget_ms:.0f} ms ({cadence} img/s)")
    print(f"Debit soutenu    : {debit:.2f} img/s  ->  facteur temps reel {res['facteur_temps_reel']}")
    print(f"Images au-dessus du budget : {100 * res['part_images_au_dessus_du_budget']:.1f} %")
    print(f"TEMPS REEL : {'OUI' if res['temps_reel'] else 'NON'}")
    print(f"Fichiers : {DOSSIER_SORTIE / (nom + '.json')}")
    print("=" * 74)


if __name__ == "__main__":
    main()
