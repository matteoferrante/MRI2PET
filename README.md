# MRI2PET

**Generation of synthetic TSPO PET maps from structural MRI images**

<p align="center">
  <img src="https://img.shields.io/badge/python->=3.9-blue?style=flat" />
  <img src="https://img.shields.io/badge/PyTorch->=2.1-lightgrey?style=flat" />
  <img src="https://img.shields.io/badge/MONAI->=1.4.0-orange?style=flat" />
  <img src="https://img.shields.io/badge/Status-Research--Prototype-yellow" />
</p>

> Deep‑learning framework to translate conventional **T1‑weighted MRI** volumes into **synthetic TSPO PET** images, enabling low‑cost, radiation‑free neuro‑inflammation mapping.

---

## Table of contents

1. [Background](#background)
2. [Repository layout](#repository-layout)
3. [Quick start](#quick-start)
4. [Training](#training)
5. [Inference](#inference)
6. [Results](#results)
7. [Citation](#citation)
8. [License](#license)

---

## Background

Neuro‑inflammation is usually quantified with TSPO PET tracers such as **\[11C]PBR28**, but PET is expensive and exposes subjects to ionising radiation.
This work demonstrates that a **3‑D U‑Net** can learn to reconstruct PET uptake patterns directly from MRI, using 204 simultaneous PET/MR scans from:

* Knee osteoarthritis (KOA) patients — *15 × 1 scan, 15 × 2 scans, 14 × 3 scans*
* Chronic back‑pain patients — *40 × 2 scans, 3 × 3 scans*
* Healthy controls — *28 × 1 scan*

### Key numbers

| Metric                  | Mean ± SD               |
| ----------------------- | ----------------------- |
| **Voxel‑wise MSE**      | 0.0033 ± 0.0010         |
| **CNR (median)**        | 0.0640 ± 0.2500         |
| **# train / val folds** | 5‑fold cross‑validation |

Synthetic PET volumes preserved fine‑grained spatial patterns and remained accurate after spatial normalisation, laying a foundation for non‑invasive, low‑cost neuro‑inflammation imaging.


**Abstract**
Introduction: Neuroinflammation, a pathophysiological process involved in numerous disorders, is
typically imaged using [ 11 C]PBR28 (or TSPO) PET. However, this technique is limited by high costs
and ionizing radiation, restricting its widespread clinical use. MRI, a more accessible alternative, is
commonly used for structural or functional imaging, but when used using traditional approaches has
limited sensitivity to specific molecular processes. This study aims to develop a deep learning model
to generate TSPO PET images from structural MRI data collected in human subjects.
Methods: A total of 204 scans, from participants with knee osteoarthritis (n = 15 scanned once, 15
scanned twice, 14 scanned three times), back pain (n = 40 scanned twice, 3 scanned three times), and
healthy controls (n=28, scanned once), underwent simultaneous 3T MRI and [ 11 C]PBR28 TSPO PET
scans. A 3D U-Net model was trained on 80% of these PET-MRI pairs and validated using 5-fold
cross-validation. The model’s accuracy in reconstructed PET from MRI only was assessed using
various intensity and noise metrics.
Results: The model achieved a low voxel-wise mean squared error (0.0033 ± 0.0010) across all folds
and a median contrast-to-noise ratio of 0.0640 ± 0.2500 when comparing true to reconstructed PET
images. The synthesized PET images accurately replicated the spatial patterns observed in the
original PET data. Additionally, the reconstruction accuracy was maintained even after spatial
normalization.

## Repository layout

```
MRI2PET/
├─ notebooks/            # Jupyter notebooks used for figures & analysis
│   ├─ Analysis.ipynb
│   ├─ PET2MRI_v1.ipynb
│   └─ ...
├─ network.py            # Custom MONAI regression backbone (depth‑wise option)
├─ unet3D.py             # 3‑D U‑Net with optional attention & depth‑wise convs
├─ pet_classifiers.ipynb # Down‑stream ML metrics
└─ .gitignore            # Tracks only *.py, *.ipynb & folder structure
```

*(File names kept as in the original commit; feel free to reorganise.)*

## Quick start

> **Prerequisites**
> Python ≥ 3.9, CUDA‑enabled PyTorch ≥ 2.1, [MONAI](https://monai.io) ≥ 1.4, plus standard scientific stack.

```bash
# 1. Clone
$ git clone https://github.com/matteoferrante/MRI2PET.git
$ cd MRI2PET

# 2. Create environment (conda example)
$ conda env create -f environment.yml   # or follow requirements.txt once provided
$ conda activate mri2pet

# 3. Organise data
# ├─ data/
# │   ├─ subject_001/
# │   │   ├─ T1.nii.gz
# │   │   └─ PET.nii.gz
# │   └─ ...
# (Raw data are not distributed – request via corresponding author.)

# 4. Run a demo notebook
$ jupyter lab notebooks/Analysis.ipynb
```


## Citation

If you use this codebase, please cite the accompanying manuscript:

```bibtex
@article{Ferrante2025MRI2PET,
  title     = {Generation of synthetic TSPO PET maps from structural MRI images},
  author    = {Ferrante, Matteo and colleagues},
  journal   = {bioRxiv},
  year      = {2025},
  url       = {https://github.com/matteoferrante/MRI2PET}
}
```

## License

This repository is released for **academic research**.
Unless a `LICENSE` file is added, all rights remain with the authors. For commercial enquiries please contact the corresponding author.
