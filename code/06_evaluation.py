"""
Single evaluation on the test subjects: metrics, block-bootstrap CIs, McNemar tests,
calibration, threshold sensitivity and per-subject false alarms.

Usage: python code/06_evaluation.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_MOD = RACINE / "artefacts" / "modeles"
DOSSIER_FEN = RACINE / "artefacts" / "fenetres"
DOSSIER_EVAL = RACINE / "artefacts" / "evaluation"
CONFIG_PRINCIPALE = "c5_L16"
REFERENCE = "transformer"
N_BOOT = 2000
TAILLE_BLOC = 20
GRAINE_BOOT = 12345
BLEU, ORANGE, GRIS, ENCRE = "#2a78d6", "#eb6834", "#9a9a94", "#3d3d3a"


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
                specificite=spe, f1=2 * prec * rec / max(1e-9, prec + rec), auc=auc(y, p),
                TP=tp, TN=tn, FP=fp, FN=fn)


def ece(y, p, n=10):
    bords = np.linspace(0, 1, n + 1)
    total = 0.0
    for a, b in zip(bords[:-1], bords[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def blocs(etudiants):
    par_etu = {}
    for i, e in enumerate(etudiants):
        par_etu.setdefault(e, []).append(i)
    return {e: [np.array(ix[k:k + TAILLE_BLOC]) for k in range(0, len(ix), TAILLE_BLOC)]
            for e, ix in par_etu.items()}


def tirages(bl, rng):
    out = []
    for e, liste in bl.items():
        choix = rng.integers(0, len(liste), len(liste))
        out += [liste[c] for c in choix]
    return np.concatenate(out)


def moy_graines(y, P, seuil, cle):
    return float(np.mean([metriques(y, P[:, g], seuil)[cle] for g in range(P.shape[1])]))


def ic_bootstrap(y, P, seuil, bl, cles=("bal_acc", "f1", "auc", "acc")):
    rng = np.random.default_rng(GRAINE_BOOT)
    vals = {c: [] for c in cles}
    for _ in range(N_BOOT):
        ix = tirages(bl, rng)
        ms = [metriques(y[ix], P[ix, g], seuil) for g in range(P.shape[1])]
        for c in cles:
            vals[c].append(np.mean([m[c] for m in ms]))
    return {c: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for c, v in vals.items()}


def ic_difference(y, Pa, s_a, Pb, s_b, bl):
    rng = np.random.default_rng(GRAINE_BOOT)
    d = []
    for _ in range(N_BOOT):
        ix = tirages(bl, rng)
        d.append(moy_graines(y[ix], Pa[ix], s_a, "bal_acc") - moy_graines(y[ix], Pb[ix], s_b, "bal_acc"))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mcnemar(y, p_a, s_a, p_b, s_b):
    ca = (p_a >= s_a).astype(int) == y
    cb = (p_b >= s_b).astype(int) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pval = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, float(pval)


def lire_variante(dossier):
    with open(dossier / "predictions_test.csv", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    graines = [c for c in r[0] if c.startswith("g")]
    ids = [x["id"] for x in r]
    etu = [x["etudiant"] for x in r]
    y = np.array([int(x["label"]) for x in r])
    P = np.array([[float(x[g]) for g in graines] for x in r])
    choix = json.loads((dossier / "choix.json").read_text(encoding="utf-8"))
    return dict(ids=ids, etu=etu, y=y, P=P, seuil=float(choix["seuil"]),
                epoques=int(choix["epoques"]), graines=graines)


def infos_config(config):
    cad, L = config[1:].split("_L")
    cad, L = int(cad), int(L)
    with open(DOSSIER_FEN / f"{config}.csv", encoding="utf-8") as f:
        durees = [float(r["duree_s"]) for r in csv.DictReader(f)]
    return cad, L, float(np.median(durees)), (L // 2) / cad


def figure_fiabilite(y, p, config, chemin):
    bords = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for a, b in zip(bords[:-1], bords[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    ax.plot([0, 1], [0, 1], color=GRIS, lw=1, ls="--", label="perfect calibration")
    ax.plot(xs, ys, color=BLEU, lw=2, marker="o", ms=6, mec="white", mew=1.5, label="Transformer (mean of 5 seeds)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of cheating", color=ENCRE)
    ax.set_ylabel("Observed frequency of cheating", color=ENCRE)
    ax.set_title(f"Reliability diagram - test subjects ({config})", fontsize=10, color=ENCRE)
    ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(chemin); plt.close(fig)


def figure_seuil(y, p, seuil_dev, config, chemin):
    grille = np.linspace(0.05, 0.95, 91)
    bal = [metriques(y, p, s)["bal_acc"] for s in grille]
    fpr = [1 - metriques(y, p, s)["specificite"] for s in grille]
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)
    ax.plot(grille, bal, color=BLEU, lw=2, label="Balanced accuracy")
    ax.plot(grille, fpr, color=ORANGE, lw=2, label="False-positive rate")
    ax.axvline(seuil_dev, color=GRIS, lw=1, ls="--")
    ax.text(seuil_dev + 0.01, 0.62, f"threshold selected\non dev = {seuil_dev:.2f}",
            fontsize=7, color=ENCRE, va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Decision threshold", color=ENCRE); ax.set_ylabel("Value", color=ENCRE)
    ax.set_title(f"Threshold sensitivity - test subjects ({config})", fontsize=10, color=ENCRE)
    ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout(); fig.savefig(chemin); plt.close(fig)


def main():
    if not DOSSIER_MOD.exists():
        sys.exit("[!] artefacts\\modeles introuvable : lancez d'abord l'etape E.")
    DOSSIER_EVAL.mkdir(parents=True, exist_ok=True)
    lignes, comparaisons, par_etu = [], [], []

    configs = sorted(d.name for d in DOSSIER_MOD.iterdir() if d.is_dir())
    for config in configs:
        cad, L, duree, pas_s = infos_config(config)
        variantes = {d.name: lire_variante(d) for d in sorted((DOSSIER_MOD / config).iterdir())
                     if (d / "predictions_test.csv").exists()}
        if not variantes:
            continue
        for nom, v in variantes.items():
            y, P, s = v["y"], v["P"], v["seuil"]
            p_moy = P.mean(1)
            bl = blocs(v["etu"])
            par_graine = [metriques(y, P[:, g], s) for g in range(P.shape[1])]
            ic = ic_bootstrap(y, P, s, bl)
            heures = len(y) * pas_s / 3600
            ligne = dict(config=config, cadence_img_s=cad, L=L, duree_fenetre_s=round(duree, 1),
                         variante=nom, n_fenetres=len(y), n_triche=int(y.sum()),
                         epoques=v["epoques"], seuil_dev=s, n_graines=P.shape[1])
            for c in ("acc", "bal_acc", "precision", "rappel", "specificite", "f1", "auc"):
                vals = [g[c] for g in par_graine]
                ligne[f"{c}_moy"] = round(float(np.mean(vals)), 4)
                ligne[f"{c}_et"] = round(float(np.std(vals)), 4)
            for c in ("acc", "bal_acc", "f1", "auc"):
                ligne[f"{c}_ic95"] = f"[{ic[c][0]:.3f} ; {ic[c][1]:.3f}]"
            for c in ("TP", "TN", "FP", "FN"):
                ligne[c] = round(float(np.mean([g[c] for g in par_graine])), 1)
            for c in ("TP", "TN", "FP", "FN"):
                ligne[f"{c}_par_graine"] = " ".join(str(g[c]) for g in par_graine)
            ligne.update(ece=round(float(np.mean([ece(y, P[:, g]) for g in range(P.shape[1])])), 4),
                         brier=round(float(np.mean([brier(y, P[:, g]) for g in range(P.shape[1])])), 4),
                         fausses_alertes_par_heure=round(ligne["FP"] / heures, 1))
            lignes.append(ligne)

            if nom == REFERENCE:
                for e in sorted(set(v["etu"])):
                    ix = np.array([i for i, x in enumerate(v["etu"]) if x == e])
                    mg = lambda c: moy_graines(y[ix], P[ix], s, c)
                    fp = mg("FP")
                    par_etu.append(dict(config=config, etudiant=e, n_fenetres=len(ix),
                                        n_triche=int(y[ix].sum()), rappel=round(mg("rappel"), 4),
                                        specificite=round(mg("specificite"), 4),
                                        bal_acc=round(mg("bal_acc"), 4), FP=round(fp, 1),
                                        taux_fausses_alertes=round(1 - mg("specificite"), 4),
                                        fausses_alertes_par_heure=round(fp / (len(ix) * pas_s / 3600), 1)))
                figure_fiabilite(y, p_moy, config, DOSSIER_EVAL / f"fiabilite_{config}.png")
                figure_seuil(y, p_moy, s, config, DOSSIER_EVAL / f"seuil_{config}.png")

        if REFERENCE in variantes:
            ref = variantes[REFERENCE]
            y, Pr, sr = ref["y"], ref["P"], ref["seuil"]
            bl = blocs(ref["etu"])
            for nom, v in variantes.items():
                if nom == REFERENCE:
                    continue
                assert v["ids"] == ref["ids"], "fenetres differentes : comparaison impossible"
                Po, so = v["P"], v["seuil"]
                d = moy_graines(y, Pr, sr, "bal_acc") - moy_graines(y, Po, so, "bal_acc")
                lo, hi = ic_difference(y, Pr, sr, Po, so, bl)
                pvals = [mcnemar(y, Pr[:, g], sr, Po[:, g], so)[2]
                         for g in range(min(Pr.shape[1], Po.shape[1]))]
                comparaisons.append(dict(config=config, comparaison=f"{REFERENCE} vs {nom}",
                                         diff_bal_acc=round(d, 4), ic95_diff=f"[{lo:.3f} ; {hi:.3f}]",
                                         mcnemar_p_median=round(float(np.median(pvals)), 4),
                                         mcnemar_graines_p_inf_005=f"{sum(p < 0.05 for p in pvals)}/{len(pvals)}",
                                         ic_exclut_zero="oui" if (lo > 0 or hi < 0) else "non"))

    for nom_f, rows in (("resultats_test.csv", lignes), ("comparaisons_test.csv", comparaisons),
                        ("par_etudiant_test.csv", par_etu)):
        if rows:
            with open(DOSSIER_EVAL / nom_f, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
                w.writeheader()
                w.writerows(rows)

    print("=" * 100)
    print("RESULTATS SUR LE TEST (3 etudiants jamais vus) - seuil choisi sur le dev")
    print("bal_acc / F1 / AUC : moyenne +- ecart-type sur les graines ; [IC 95 %] de cette moyenne,")
    print("bootstrap par blocs ; TP/TN/FP/FN : moyenne sur les graines (valeurs exactes dans le CSV)")
    print("=" * 100)
    for config in configs:
        rs = [l for l in lignes if l["config"] == config]
        if not rs:
            continue
        print(f"\n--- {config}  ({rs[0]['cadence_img_s']} img/s, {rs[0]['L']} crops = "
              f"{rs[0]['duree_fenetre_s']} s, {rs[0]['n_fenetres']} fenetres de test)")
        print(f"{'variante':<26}{'bal_acc':>16}{'IC95':>18}{'F1':>16}{'AUC':>16}"
              f"{'TP':>6}{'TN':>6}{'FP':>6}{'FN':>6}{'ECE':>7}{'Brier':>7}")
        for l in rs:
            print(f"{l['variante']:<26}{l['bal_acc_moy']:>9.3f}+-{l['bal_acc_et']:.3f}{l['bal_acc_ic95']:>18}"
                  f"{l['f1_moy']:>9.3f}+-{l['f1_et']:.3f}{l['auc_moy']:>9.3f}+-{l['auc_et']:.3f}"
                  f"{l['TP']:>7.0f}{l['TN']:>6.0f}{l['FP']:>6.0f}{l['FN']:>6.0f}{l['ece']:>7.3f}{l['brier']:>7.3f}")
    if comparaisons:
        print(f"\n--- COMPARAISONS APPARIEES ({REFERENCE} vs ...) : difference d'exactitude equilibree")
        print(f"{'config':<9}{'comparaison':<38}{'diff':>8}{'IC95 diff':>20}{'IC exclut 0':>12}"
              f"{'p McNemar med.':>16}{'graines p<0,05':>16}")
        for c in comparaisons:
            print(f"{c['config']:<9}{c['comparaison']:<38}{c['diff_bal_acc']:>+8.3f}{c['ic95_diff']:>20}"
                  f"{c['ic_exclut_zero']:>12}{c['mcnemar_p_median']:>16.4f}{c['mcnemar_graines_p_inf_005']:>16}")
    if par_etu:
        print(f"\n--- {REFERENCE} PAR ETUDIANT DE TEST")
        print(f"{'config':<9}{'etudiant':<12}{'fenetres':>9}{'rappel':>8}{'specif.':>9}"
              f"{'taux FA':>9}{'FA/heure':>10}")
        for r in par_etu:
            print(f"{r['config']:<9}{r['etudiant']:<12}{r['n_fenetres']:>9}{r['rappel']:>8.3f}"
                  f"{r['specificite']:>9.3f}{r['taux_fausses_alertes']:>9.3f}{r['fausses_alertes_par_heure']:>10.1f}")
    print(f"\nFichiers ecrits dans {DOSSIER_EVAL}")
    print("=" * 100)


if __name__ == "__main__":
    main()
