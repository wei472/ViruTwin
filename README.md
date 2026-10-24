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

| Architecture | Feature | F1 | AUROC | AUPRC |
|---|---|:---:|:---:|:---:|
| MLP (shallow) | Concatenated | **0.8456** | **0.9216** | **0.9261** |
| DPF wo/SelfAttn | ESM-2 + ProtT5 | 0.8238 | 0.9110 | 0.9142 |
| DPF Full | ESM-2 + ProtT5 | 0.8166 | 0.9049 | 0.9043 |
| VFIterTuned | Concatenated | 0.8176 | 0.9011 | 0.9005 |
| **SuperEnsemble** | Combined | — | **0.9199** | **0.9210** |

&gt; See `docs/verification_correct.csv` for complete results.

---

## Quick Start

```bash
pip install -r requirements.txt
