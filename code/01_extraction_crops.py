"""
Runs the trained YOLOv5 detector on the webcam videos (5 frames/s) and saves one crop per frame,
named <subject>_f<frame>.jpg, with the detector class and box in a CSV per video.

Usage: python code/01_extraction_crops.py --yolov5 <path/to/yolov5> [--fps 5] [--video NAME]
"""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_VIDEOS = RACINE / "videos"
POIDS = RACINE / "weights" / "best.pt"
DOSSIER_CROPS = RACINE / "crops_v2"
DOSSIER_ART = RACINE / "artefacts" / "extraction"
TAILLE_IMG = 640
SEUIL_CONF = 0.25
MARGE = 0.0


def charger_yolo(dossier_yolov5):
    import torch
    modele = torch.hub.load(str(dossier_yolov5), "custom", path=str(POIDS),
                            source="local", verbose=False)
    modele.conf = SEUIL_CONF
    modele.eval()
    return modele


def detecter(modele, image_bgr):
    res = modele(image_bgr[..., ::-1], size=TAILLE_IMG)
    det = res.xyxy[0]
    if det is None or len(det) == 0:
        return None
    meilleure = det[det[:, 4].argmax()].tolist()
    x1, y1, x2, y2, conf, cls = meilleure
    return int(x1), int(y1), int(x2), int(y2), float(conf), modele.names[int(cls)]


def traiter_video(modele, chemin_video, fps_cible):
    etu = chemin_video.stem
    csv_sortie = DOSSIER_ART / f"{etu}.csv"
    if csv_sortie.exists():
        print(f"[=] {etu} deja traite, on saute.")
        return
    dossier = DOSSIER_CROPS / etu
    dossier.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(chemin_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pas = max(1, round(fps / fps_cible))
    print(f"[>] {etu}: {total} frames a {fps:.0f} FPS, 1 frame sur {pas} "
          f"(~{total // pas} images a analyser)")

    lignes, temps_yolo = [], []
    t0 = time.time()
    i = -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % pas:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        t = time.perf_counter()
        d = detecter(modele, img)
        temps_yolo.append(time.perf_counter() - t)

        ligne = {"etudiant": etu, "frame": i, "temps_s": round(i / fps, 3),
                 "classe_yolo": "", "conf": "", "x1": "", "y1": "", "x2": "", "y2": "",
                 "fichier": ""}
        if d is not None:
            x1, y1, x2, y2, conf, nom = d
            h, w = img.shape[:2]
            mx, my = int((x2 - x1) * MARGE), int((y2 - y1) * MARGE)
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(w, x2 + mx), min(h, y2 + my)
            nom_fichier = f"{etu}_f{i:06d}.jpg"
            cv2.imwrite(str(dossier / nom_fichier), img[y1:y2, x1:x2])
            ligne.update(classe_yolo=nom, conf=round(conf, 4), x1=x1, y1=y1, x2=x2, y2=y2,
                         fichier=str(Path("crops_v2") / etu / nom_fichier))
        lignes.append(ligne)

        n = len(lignes)
        if n % 500 == 0:
            ecoule = time.time() - t0
            reste = (total // pas - n) * ecoule / n
            print(f"    {n} images | {n / ecoule:.1f} img/s | reste ~{reste / 60:.0f} min")
    cap.release()

    tmp = csv_sortie.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)
    tmp.replace(csv_sortie)

    sans = sum(1 for l in lignes if not l["fichier"])
    moy = 1000 * sum(temps_yolo) / max(1, len(temps_yolo))
    print(f"[ok] {etu}: {len(lignes)} images, {len(lignes) - sans} crops, "
          f"{sans} sans detection | YOLOv5 {moy:.0f} ms/image")
    with open(DOSSIER_ART / "temps_yolo.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([etu, len(temps_yolo), round(moy, 2)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolov5", required=True, help="dossier du depot yolov5 (contient detect.py)")
    p.add_argument("--fps", type=float, default=5)
    p.add_argument("--video", default=None)
    a = p.parse_args()

    dossier_yolo = Path(a.yolov5)
    if not (dossier_yolo / "hubconf.py").exists():
        sys.exit(f"[!] {dossier_yolo} ne contient pas hubconf.py : ce n'est pas le dossier yolov5.")
    if not POIDS.exists():
        sys.exit(f"[!] Poids introuvables : {POIDS}")

    DOSSIER_ART.mkdir(parents=True, exist_ok=True)
    DOSSIER_CROPS.mkdir(exist_ok=True)

    import platform
    import torch
    (DOSSIER_ART / "config_extraction.txt").write_text(
        f"poids={POIDS.name}\nimg_size={TAILLE_IMG}\nconf={SEUIL_CONF}\nmarge={MARGE}\n"
        f"fps_echantillonnage={a.fps}\nregle=1 detection par image (la plus confiante)\n"
        f"python={platform.python_version()}\ntorch={torch.__version__}\n"
        f"cpu={platform.processor()}\ncuda={torch.cuda.is_available()}\n", encoding="utf-8")

    modele = charger_yolo(dossier_yolo)
    print("Classes YOLOv5 :", modele.names)

    videos = sorted(DOSSIER_VIDEOS.glob("*.avi"))
    if a.video:
        videos = [v for v in videos if v.stem == a.video]
    for v in videos:
        traiter_video(modele, v, a.fps)
    print("\nTermine. CSV dans", DOSSIER_ART, "| crops dans", DOSSIER_CROPS)


if __name__ == "__main__":
    main()
