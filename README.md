# Real-Time Exam Cheating Detection — Subject-Independent Evaluation

Code and evaluation artefacts for the paper:

> I. Khabbachi, G. Mdaghri-Alaoui, A. Zouhair, *Real-Time Exam Cheating Detection Using a Multi-Agent
> Spatio-Temporal Architecture: A Subject-Independent Evaluation*, International Journal of Intelligent
> Engineering and Systems (IJIES), under review.

The pipeline localises the examinee with YOLOv5, encodes each crop with a frozen DINOv2 ViT-B/14, classifies
short temporal windows with a Transformer encoder, and applies a decision threshold selected on development
data only. The evaluation splits the 15 OEP webcam subjects **before** any window is built, keeps three
subjects aside for a single final test, and makes every modelling choice by grouped cross-validation on the
other twelve.

## Repository layout

```
code/
  01_extraction_crops.py   YOLOv5 crops at 5 frames/s, one CSV per video (frame, time, class, box)
  02_split.py              subject-level split (3 test subjects + 4 folds), split.json
  03_embeddings.py         frozen DINOv2 ViT-B/14 CLS embeddings per crop
  04_fenetres.py           windows built inside each subject, after the split (all configurations)
  05_entrainement.py       grouped CV, choice of epochs and threshold, final models, test predictions
  run_training.bat         runs every training configuration in sequence (Windows)
  06_evaluation.py         test metrics, block-bootstrap CIs, McNemar, calibration, per-subject errors
  07_chronometrage.py      frame-by-frame end-to-end timing on CPU or GPU
  08_preparer_colab.py     packs the files needed for the GPU timing on Google Colab
  09_export_results.py     copies the audit artefacts into results/
colab/
  chronometrage_gpu.ipynb  notebook used for the GPU timing (Tesla T4)
results/                   split, windows, predictions, selected settings, evaluation and timing outputs
```

## Data

The OEP (Online Exam Proctoring) videos are distributed by their original authors
(Y. Atoum et al., *IEEE Trans. Multimedia*, 19(7), 2017) and are **not** included here. Only the webcam
stream of 15 subjects is used. Expected local layout:

```
<project>/
  code/            scripts of this repository
  videos/          <subject>.avi  (e.g. alhashe31.avi, zach1.avi)
  weights/         best.pt        (YOLOv5 detector trained on the LabelImg annotations)
  crops_v2/        created by step 1
  artefacts/       created by steps 1-7
```

**Labels.** The YOLOv5 detector was trained on manually annotated frames (LabelImg, 2 frames/s,
19,749 images). The temporal models use, for every analysed frame, the class predicted by this detector, and
each window takes the majority label of its frames. Results therefore measure agreement with
detector-derived labels, as discussed in the paper.

## Installation

```
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
```

DINOv2 is downloaded automatically through `torch.hub` on first use.

## Reproducing the study

Run from the project folder:

```
python code/01_extraction_crops.py --yolov5 <path/to/yolov5>
python code/02_split.py --test Muath1 XiYin1 zach1
python code/03_embeddings.py
python code/04_fenetres.py
code\run_training.bat                       (or the python commands it contains)
python code/06_evaluation.py
python code/07_chronometrage.py --yolov5 <path/to/yolov5> --config c5_L16
python code/07_chronometrage.py --yolov5 <path/to/yolov5> --config c1_L8
python code/09_export_results.py --projet <project> --depot <this repository>
```

Steps 1, 3 and 5 can be interrupted and restarted: completed subjects or variants are skipped.

## Protocol summary

| Item | Setting |
|---|---|
| Split unit | subject (= one webcam session) |
| Test subjects | Muath1, XiYin1, zach1 (not used to train YOLOv5 or any temporal model) |
| Development folds | 4 folds of 3 subjects, seed 42 |
| Sampling / primary window | 5 frames/s, 16 crops (3.2 s), stride 8 |
| Standardisation | z-score fitted on the training subjects of each fold only |
| Epochs and threshold | chosen on out-of-fold predictions (no early stopping, no checkpoint selection on test) |
| Seeds | 0, 1, 2, 3, 4 |
| Confidence intervals | subject-stratified block bootstrap, blocks of 20 windows, 2,000 replicates |

## Results folder

| Path | Content |
|---|---|
| `results/split.json` | test subjects, development folds, seed, rules |
| `results/crops_index.csv` | every analysed frame: subject, partition, frame, time, detector class and confidence |
| `results/fenetres/*.csv` | every window: id, subject, partition, first/last frame, time span, labels, crop indices |
| `results/modeles/<config>/<variant>/` | `cv.csv`, `choix.json` (epochs, threshold, rule), `oof_dev.csv`, `predictions_test.csv`, `config.json` |
| `results/evaluation/` | test metrics, paired comparisons, per-subject results, reliability and threshold figures |
| `results/chronometrage/` | per-stage timing, sustained throughput and hardware description (CPU and GPU) |

## Citation

Please cite the paper above if you use this code. The full reference will be added after publication.
