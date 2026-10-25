# ViruTwin: Human Virulence Factor Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

&gt; **Dual-Path Fusion with Iterative Low-Rank Refinement for Human Virulence Factor Prediction**  
&gt; Submitted to *Briefings in Bioinformatics*

---

## Overview

ViruTwin predicts whether a protein sequence is a human virulence factor (VF) using:

- **ESM-2** (1280-d) + **ProtT5** (1024-d) protein language model embeddings
- **Dual-Path Fusion (DPF)** for cross-modal feature interaction
- **VF-Iter** for iterative low-rank refinement
- **SuperEnsemble** with out-of-fold AUROC optimization

---

## Model Performance (Test Set, n=1,200)
&gt; See `docs/verification_correct.csv` for complete results.

---

## Quick Start

```bash
pip install -r requirements.txt

---

## Installation & Usage (Inference)

```bash
pip install -r requirements.txt

export VT_MODELS_PY=$PWD/model_defs.py
export OMP_NUM_THREADS=8

# single sequence (protein or mRNA, auto-detected)
python predict.py --seq "MKKILCAVAVGISASVQATK" --model superensemble

# batch FASTA
python predict.py --input seqs.fasta --model superensemble --out result.json
```

The firtt run downloads ESM-2 (~2.5 GB) and ProtT5-XL (~2.4 GB) from HuggingFace
automatically. In China, run `export HF_ENDPOINT=https://hf-mirror.com` first.

## Available Models

| key | weights | held-out test AUROC |
|---|---|---|
| superensemble | equal-weight average of M0-M4 | 0.9199 (AUPRC 0.9210) |
| mlp | cat__MLP.joblib | 0.9216 (best single) |
| vfiter_cat / vfiter_esm2 / vfiter_p5 | VFIter.pth | 0.9011-0.9052 |
| dpf (M0), dpf_m1..m4 | M0-M4.pth | 0.9047-0.9110 |

## Web API

```bash
python app.py   # serves on port 6006
```

- `GET /health` - model loading status
- `POST /predict` - JSON body `{"sequences": ">id SEQ...", "model": "superensemble"}`

## Weight Files

- `model_weights/` - individual `.pth` / `.joblib` / scaler `.npz` files, loaded automatically.
- `ViruTwin-weights.z01`~`.z04` - archived full split volumes; merge with any zip tool.

## Citation

If you use ViruTwin, please cite the ViruTwin manuscript, plus ESM-2
(Lin et al., Science 2023), ProtT5 (Elnaggar et al., IEEE TPAMI 2022)
and VFDB (Zhou et al., Nucleic Acids Research 2025).
