"""
Grouped cross-validation on the development subjects (choice of epochs and threshold),
retraining on all development subjects and prediction on the test subjects, for each variant and seed.

Usage: python code/05_entrainement.py [--config c5_L16] [--modeles ...] [--graines 0 1 2 3 4]
"""
import argparse
import csv
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

APPAREIL = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_EMB = RACINE / "artefacts" / "embeddings"
DOSSIER_FEN = RACINE / "artefacts" / "fenetres"
SPLIT = RACINE / "artefacts" / "split.json"
DOSSIER_MOD = RACINE / "artefacts" / "modeles"

HP_BASE = dict(d_model=256, n_tetes=4, n_couches=2, ff=512, dropout=0.3,
               lr=3e-4, weight_decay=1e-2, batch=64, epoques_max=15, clip=1.0,
               scheduler=True, choix_epoque=True)

VARIANTES = {
    "transformer": dict(archi="transformer"),
    "transformer_sans_dropout": dict(archi="transformer", dropout=0.0),
    "transformer_sans_regul": dict(archi="transformer", dropout=0.0, weight_decay=0.0,
                                   scheduler=False, clip=None, choix_epoque=False),
    "moyenne": dict(archi="moyenne"),
    "lstm": dict(archi="lstm"),
    "gru": dict(archi="gru"),
}


class TransformerTemporel(nn.Module):
    def __init__(self, d_in, L, hp):
        super().__init__()
        d = hp["d_model"]
        self.proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.normal_(self.pos, std=0.02)
        couche = nn.TransformerEncoderLayer(d, hp["n_tetes"], hp["ff"], hp["dropout"],
                                            batch_first=True)
        self.enc = nn.TransformerEncoder(couche, hp["n_couches"])
        self.drop = nn.Dropout(hp["dropout"])
        self.tete = nn.Linear(d, 1)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.tete(self.drop(h.mean(1))).squeeze(1)


class Moyenne(nn.Module):
    def __init__(self, d_in, L, hp):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, hp["d_model"]), nn.ReLU(),
                                 nn.Dropout(hp["dropout"]), nn.Linear(hp["d_model"], 1))

    def forward(self, x):
        return self.mlp(x.mean(1)).squeeze(1)


class Recurrent(nn.Module):
    def __init__(self, d_in, L, hp, cellule):
        super().__init__()
        d = hp["d_model"]
        self.proj = nn.Linear(d_in, d)
        self.rnn = cellule(d, d, batch_first=True)
        self.drop = nn.Dropout(hp["dropout"])
        self.tete = nn.Linear(d, 1)

    def forward(self, x):
        sortie, _ = self.rnn(self.proj(x))
        return self.tete(self.drop(sortie[:, -1])).squeeze(1)


def construire(archi, d_in, L, hp):
    if archi == "transformer":
        return TransformerTemporel(d_in, L, hp)
    if archi == "moyenne":
        return Moyenne(d_in, L, hp)
    return Recurrent(d_in, L, hp, nn.LSTM if archi == "lstm" else nn.GRU)


def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metriques(y, p, seuil):
    yh = (p >= seuil).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    rec = tp / max(1, tp + fn); spe = tn / max(1, tn + fp); prec = tp / max(1, tp + fp)
    return dict(acc=(tp + tn) / len(y), bal_acc=(rec + spe) / 2, precision=prec, rappel=rec,
                f1=2 * prec * rec / max(1e-9, prec + rec), auc=auc(y, p), bce=bce(y, p),
                TP=tp, TN=tn, FP=fp, FN=fn)


def meilleur_seuil(y, p):
    grille = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [metriques(y, p, s)["bal_acc"] for s in grille]
    return float(grille[int(np.argmax(scores))])


def charger(config):
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    emb = {}
    for p in sorted(DOSSIER_EMB.glob("*.npz")):
        if not p.stem.endswith("_tmp"):
            emb[p.stem] = np.load(p)["emb"]
    fen = []
    with open(DOSSIER_FEN / f"{config}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fen.append(dict(id=r["id"], etu=r["etudiant"], part=r["partition"],
                            y=int(r["label_majorite"]),
                            idx=np.array(r["indices"].split(), dtype=np.int64)))
    return split, emb, fen


def scaler(emb, etudiants):
    tout = np.concatenate([emb[e].astype(np.float32) for e in etudiants])
    return tout.mean(0), tout.std(0) + 1e-6


def tenseurs(fenetres, emb, mu, sd):
    X = np.stack([((emb[w["etu"]][w["idx"]].astype(np.float32) - mu) / sd) for w in fenetres])
    return X.astype(np.float16), np.array([w["y"] for w in fenetres], dtype=np.float32)


@torch.no_grad()
def predire(modele, X):
    modele.eval()
    out = [torch.sigmoid(modele(torch.from_numpy(X[i:i + 512].astype(np.float32)).to(APPAREIL))).cpu().numpy()
           for i in range(0, len(X), 512)]
    return np.concatenate(out)


def entrainer(hp, Xtr, ytr, graine, epoques, Xval=None):
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    modele = construire(hp["archi"], Xtr.shape[2], Xtr.shape[1], hp).to(APPAREIL)
    opt = torch.optim.AdamW(modele.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epoques_max"])
             if hp["scheduler"] else None)
    pos = float(ytr.mean())
    critere = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=APPAREIL))
    historique = []
    for _ in range(epoques):
        modele.train()
        ordre = rng.permutation(len(ytr))
        for i in range(0, len(ordre), hp["batch"]):
            b = ordre[i:i + hp["batch"]]
            opt.zero_grad()
            perte = critere(modele(torch.from_numpy(Xtr[b].astype(np.float32)).to(APPAREIL)),
                            torch.from_numpy(ytr[b]).to(APPAREIL))
            perte.backward()
            if hp["clip"]:
                nn.utils.clip_grad_norm_(modele.parameters(), hp["clip"])
            opt.step()
        if sched:
            sched.step()
        if Xval is not None:
            historique.append(predire(modele, Xval))
    return modele, historique


def executer_variante(m, hp, a, split, emb, fen, sortie_cfg):
    sortie = sortie_cfg / m
    if (sortie / "predictions_test.csv").exists():
        print(f"[=] {m} deja termine, on saute.")
        return
    sortie.mkdir(parents=True, exist_ok=True)
    folds = split["folds_validation_croisee"]
    dev = [e for f in folds for e in f]
    E = hp["epoques_max"]
    t_debut = time.time()

    fen_dev = [w for w in fen if w["etu"] in dev]
    y_dev = np.array([w["y"] for w in fen_dev], dtype=np.float32)
    pos_dev = {w["id"]: i for i, w in enumerate(fen_dev)}
    oof = {s: np.full((E, len(fen_dev)), np.nan, dtype=np.float32) for s in a.graines}
    print(f"\n=== {m} | validation croisee ({len(fen_dev)} fenetres dev, {E} epoques max)")
    for k, fold_val in enumerate(folds):
        train_etu = [e for e in dev if e not in fold_val]
        mu, sd = scaler(emb, train_etu)
        f_tr = [w for w in fen_dev if w["etu"] in train_etu]
        f_va = [w for w in fen_dev if w["etu"] in fold_val]
        Xtr, ytr = tenseurs(f_tr, emb, mu, sd)
        Xva, _ = tenseurs(f_va, emb, mu, sd)
        cols = np.array([pos_dev[w["id"]] for w in f_va])
        for s in a.graines:
            t = time.time()
            _, hist = entrainer(hp, Xtr, ytr, s, E, Xva)
            for ep, p in enumerate(hist):
                oof[s][ep, cols] = p
            print(f"  fold{k} {m:<26} graine {s}  ({time.time() - t:.0f} s)")
        del Xtr, Xva

    bce_ep = np.mean([[bce(y_dev, oof[s][ep]) for ep in range(E)] for s in a.graines], 0)
    e_star = int(np.argmin(bce_ep)) + 1 if hp["choix_epoque"] else E
    p_pool = np.concatenate([oof[s][e_star - 1] for s in a.graines])
    seuil = meilleur_seuil(np.tile(y_dev, len(a.graines)), p_pool)
    lignes_cv = []
    for s in a.graines:
        mt = metriques(y_dev, oof[s][e_star - 1], seuil)
        lignes_cv.append(dict(variante=m, graine=s, epoques=e_star, seuil=seuil,
                              **{c: round(v, 4) if isinstance(v, float) else v for c, v in mt.items()}))
    (sortie / "choix.json").write_text(json.dumps({
        "regle": ("E* = epoque minimisant la BCE hors-fold moyenne (folds x graines) ; "
                  "seuil = maximise l'exactitude equilibree hors-fold a E*. "
                  "Variante sans_regul : E* = epoques_max (pas de choix d'epoque)."),
        "epoques": e_star, "seuil": seuil,
        "bce_oof_par_epoque": [round(float(v), 5) for v in bce_ep]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    with open(sortie / "cv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes_cv[0].keys()))
        w.writeheader()
        w.writerows(lignes_cv)
    with open(sortie / "oof_dev.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "etudiant", "label"] + [f"g{s}" for s in a.graines])
        for i, fw in enumerate(fen_dev):
            w.writerow([fw["id"], fw["etu"], int(fw["y"])] +
                       [round(float(oof[s][e_star - 1, i]), 5) for s in a.graines])

    mu, sd = scaler(emb, dev)
    Xdev, ydev = tenseurs(fen_dev, emb, mu, sd)
    fen_test = [w for w in fen if w["part"] == "test"]
    Xte, _ = tenseurs(fen_test, emb, mu, sd)
    np.savez(sortie / "scaler_dev.npz", mu=mu, sd=sd)
    preds = {}
    for s in a.graines:
        modele, _ = entrainer(hp, Xdev, ydev, s, e_star)
        torch.save(modele.state_dict(), sortie / f"graine{s}.pt")
        preds[s] = predire(modele, Xte)
        print(f"  final {m:<26} graine {s}  ({e_star} epoques)")
    (sortie / "config.json").write_text(json.dumps({
        "variante": m, "config_fenetres": a.config, "graines": a.graines, "hyperparametres": hp,
        "etiquette_fenetre": "majorite des crops (>= 50 % triche)",
        "source_etiquettes": "yolo_pseudo",
        "scaler": "StandardScaler ajuste sur les etudiants d'entrainement de chaque fold ; "
                  "modele final : sur les 12 etudiants dev",
        "dinov2": "gele", "python": platform.python_version(), "torch": torch.__version__,
        "cpu": platform.processor(), "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if APPAREIL.type == "cuda" else "aucun",
        "duree_min": round((time.time() - t_debut) / 60, 1)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    tmp = sortie / "predictions_test.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "etudiant", "label"] + [f"g{s}" for s in a.graines])
        for i, fw in enumerate(fen_test):
            w.writerow([fw["id"], fw["etu"], int(fw["y"])] + [round(float(preds[s][i]), 5) for s in a.graines])
    tmp.replace(sortie / "predictions_test.csv")
    print(f"[ok] {m} termine en {(time.time() - t_debut) / 60:.0f} min")


def resume(sortie_cfg):
    print("\n" + "=" * 86)
    print(f"RESULTATS HORS-FOLD sur les 12 etudiants dev  ({sortie_cfg.name}, moyenne +- ecart-type sur les graines)")
    print(f"{'variante':<26}{'E*':>4}{'seuil':>7}{'bal_acc':>16}{'F1':>16}{'AUC':>16}")
    for d in sorted(p for p in sortie_cfg.iterdir() if (p / "cv.csv").exists()):
        with open(d / "cv.csv", encoding="utf-8") as f:
            r = list(csv.DictReader(f))
        g = lambda c: f"{np.mean([float(x[c]) for x in r]):.3f} +- {np.std([float(x[c]) for x in r]):.3f}"
        print(f"{d.name:<26}{r[0]['epoques']:>4}{float(r[0]['seuil']):>7.2f}{g('bal_acc'):>16}{g('f1'):>16}{g('auc'):>16}")
    print("Les scores du TEST ne sont pas affiches ici : ils seront calcules a l'etape F.")
    print("=" * 86)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c5_L16")
    ap.add_argument("--modeles", nargs="+", default=list(VARIANTES))
    ap.add_argument("--graines", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epoques_max", type=int, default=HP_BASE["epoques_max"])
    a = ap.parse_args()
    for m in a.modeles:
        if m not in VARIANTES:
            sys.exit(f"[!] Variante inconnue : {m}. Choix : {list(VARIANTES)}")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    print("Appareil :", APPAREIL, torch.cuda.get_device_name(0) if APPAREIL.type == "cuda" else "")
    sortie_cfg = DOSSIER_MOD / a.config
    sortie_cfg.mkdir(parents=True, exist_ok=True)
    split, emb, fen = charger(a.config)
    for m in a.modeles:
        hp = {**HP_BASE, "epoques_max": a.epoques_max, **VARIANTES[m]}
        executer_variante(m, hp, a, split, emb, fen, sortie_cfg)
    resume(sortie_cfg)


if __name__ == "__main__":
    main()
