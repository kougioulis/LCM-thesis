# Large Causal Models for Temporal Causal Discovery

<p align="center">
  <em>Scalable • Robust • Multi-domain • Pre-trained</em>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-black?logo=pytorch)
![NumPy](https://img.shields.io/badge/-NumPy-013243?\&logo=NumPy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-20232A?\&logoColor=61DAFB)
![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat\&logo=pandas)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/gkorgkolis/TCS/blob/main/LICENSE)
[![e-Locus](https://img.shields.io/badge/e--Locus-archive-9D2235?style=flat)](https://elocus.lib.uoc.gr/dlib/1/d/9/metadata-dlib-1764761882-792089-25440.tkl)
</div>

Reproducibility experiments of MSc the Thesis *"Large Causal Models for Temporal Causal Discovery"* at the University of Crete (complete LaTeX source of the thesis text is available at: https://github.com/kougioulis/thesis).

---


<div align="center">

| Classical Paradigm                       | Large Causal Models              |
| -----------------------------------------| -------------------------------- |
| One model per dataset                    | One model, many datasets         |
| No pretraining                           | Massive multi-domain pretraining |
| Brittle to domain shift                  | Robust & transferable            |
| Slow inference for larger inputs         | Fast inference                   |

</div>

---

## Abstract

Causal discovery for both cross-sectional and temporal data has traditionally
followed a dataset-specific paradigm, where a new model is fitted for each individual dataset. Such an approach underutilizes the potential of multi-dataset and large-scale pretraining, especially given recent advances in foundation models. The concept of Large Causal Models (LCMs) envisions a class of pre-trained neural architectures specifically designed for temporal causal discovery. Existing approaches remain largely proofs of concept, typically constrained to small input sizes (e.g., five variables), with performance degrading rapidly to random guessing as the number of variables or model parameters increases. Moreover, current methods rely heavily on synthetic data, generated under arbitrary assumptions, which substantially limits their ability to generalize to realistic or out-of-distribution samples. This work addresses these challenges through novel methods for training on mixtures of synthetic and realistic data collections, enabling both higher input dimensionality and deeper architectures without loss of performance. Extensive experiments demonstrate that LCMs achieve competitive or superior performance compared to classical causal discovery algorithms, while maintaining robustness across diverse domains, especially on non-synthetic data cases. Our findings also highlight promising directions towards integrating interventional samples and domain knowledge, further advancing the development of foundation models for causal discovery.

---

## Contributions

### Large Causal Models
* Introduced Large Causal Models (LCMs); a family of scalable, pre-trained neural architectures for temporal causal discovery.

### Data Generation

* Developed a high-fidelity synthetic data pipeline from ground-truth temporal SCMs.
* Developed and utilized [**Temporal Causal-based Simulation (TCS)**](https://github.com/gkorgkolis/TCS): a generative methodology for creating *realistic* causal models and data from real multivariate time series.

### Training at Scale

* Generated hundreds of thousands of (data, graph) training pairs.
* Demonstrated that mixtures of synthetic and realistic training data significantly improve generalization and zero-shot performance.
* Identified optimal synthetic/realistic mixing ratios, that align with findings of works on time-series forecasting foundation models.

* Proposed regularizing term to suppress low-support edges and aid model performance.
* Experimentally showed that using observed statistics during training and inference improves model performance.

### Comparison with Existing Approaches

* Benchmarked against established methods in temporal causal discovery and showcased competitive or superior performance across synthetic, semi-synthetic and realistic datasets
* Demonstrated robustness under domain shift and zero-shot performance.

### Efficiency

* Achieved significantly faster inference than classical temporal causal discovery methods, thus opening the path to real-time applications.

---

## Setup & Getting Started 

### Conda Environment 🐍

We provide a conda environment for reproducibility purposes only. One can create a virtual conda environment using

- `conda env create -f environment.yaml`
- `conda activate LCM` 

### Using pip

Alternatively, you can just install the dependencies from the `requirements.txt` file using pip, either on your base environment or into an existing conda environment by

- `pip install -r requirements.txt`

---

## Notebooks

`experimental_results.ipynb` contains the experimental results of Section 6.5.

`illustrative_example.ipynb` contains an example of loading a pre-trained LCM, preprocessing a simple synthetic input time-series data and performing causal discovery. It illustrates both the discovered lagged causal graph, as well as the confidence weights of the lagged adjacency tensor and the $AUC$ of the model.

`ablation_experiments.ipynb` contains ablation experiments (Section 6.4.1) and zero-shot experiments on assessing the optimal mixture of realistic and synthetic training data (Section 6.4.2).

| Notebook                     | Description                                | Thesis Section |
| ---------------------------- | ------------------------------------------ | -------------- |
| `experimental_results.ipynb` | Main experimental benchmarks               | §6.5           |
| `illustrative_example.ipynb` | Loading a pretrained LCM & performing CD   | Appendix D     |
| `ablation_experiments.ipynb` | Ablations & optimal training data mixture  | §6.4.1, 6.4.2  |

CSV results used in the thesis are available under `code/data/results/`.

---


## ✨ Pretrained Models

Due to GitHub size limitations, pretrained checkpoints are hosted externally on Google Drive.

| Model     | Parameters | Link                                                                                              |
| --------- | ---------- | ------------------------------------------------------------------------------------------------- |
| LCM-2.5M  | 2.5M       | [Download](https://drive.google.com/file/d/1vjzKMpr7M_feQuFgZGSb5VWRJ_7psV0v/view?usp=sharing)    |
| LCM-9.4M  | 9.4M       | [Download](https://drive.google.com/file/d/1UCjLG4Hs6MKSJF_5G3MJmTlhGQVp3qx7/view?usp=drive_link) |
| LCM-12.2M | 12.2M      | [Download](https://drive.google.com/file/d/1bLKASu085xBJ0oqWbhNObcDie94eZFzR/view?usp=sharing)    |
| LCM-24M   | 24M        | [Download](https://drive.google.com/file/d/10gARotO3pK-bYnvpESNBlIV94SqxLxmw/view?usp=drive_link) |

---

## Test Sets

We additionally provide the test sets for the experimental evaluations, available via Google Drive links. The $f$MRI collections are available in the  `data` folder. 

### Synthetic

- S_Joint
  [https://drive.google.com/drive/folders/1RB7umIQH2H3F-kIUWVvVJzJfgv12Sxy8](https://drive.google.com/drive/folders/1RB7umIQH2H3F-kIUWVvVJzJfgv12Sxy8)
- Synth_230K
  [https://drive.google.com/drive/folders/1iqwnrMHx8sXWJRd6iysrKg13b-PCwwJs](https://drive.google.com/drive/folders/1iqwnrMHx8sXWJRd6iysrKg13b-PCwwJs)


### Semi-Synthetic

- fMRI-5
  [https://github.com/kougioulis/LCM-thesis/tree/main/data/fMRI_5](https://github.com/kougioulis/LCM-thesis/tree/main/data/fMRI_5)
- fMRI
  [https://github.com/kougioulis/LCM-thesis/tree/main/data/fMRI](https://github.com/kougioulis/LCM-thesis/tree/main/data/fMRI)
- Kuramoto-5
  [https://drive.google.com/drive/folders/1Jh9e7o4c60MDkHykX4tJvjwfWZ-khC8f](https://drive.google.com/drive/folders/1Jh9e7o4c60MDkHykX4tJvjwfWZ-khC8f)
- Kuramoto-10
  [https://drive.google.com/drive/folders/1MT3u0xvk2Wg9C0QRJ78FF5VMFCFZeKhc](https://drive.google.com/drive/folders/1MT3u0xvk2Wg9C0QRJ78FF5VMFCFZeKhc)

### Simulated (Realistic)

- Sim_45K
  [https://drive.google.com/drive/folders/1VRi2q4VH7bgxv56lCLOZlUr12sVAyYka](https://drive.google.com/drive/folders/1VRi2q4VH7bgxv56lCLOZlUr12sVAyYka)
- AirQualityMS
  [https://drive.google.com/drive/folders/15Ix7n-zIRKtJBZUTyfvtkI9bzKtl4M1O](https://drive.google.com/drive/folders/15Ix7n-zIRKtJBZUTyfvtkI9bzKtl4M1O)

### Mixture Collection

- Synth_230K_Sim_45K
  [https://drive.google.com/drive/folders/1k0cXzh8PgNX5eY3nSpb6vBYPCiYQFRm9](https://drive.google.com/drive/folders/1k0cXzh8PgNX5eY3nSpb6vBYPCiYQFRm9)

### Additional

- CDML (Lawrence et al., 2020)
  [https://drive.google.com/drive/folders/1EOIg5J3u_HAHBXP-S7Kgl_cOsG2KjYNn](https://drive.google.com/drive/folders/1EOIg5J3u_HAHBXP-S7Kgl_cOsG2KjYNn) (not present in the main text, added for completeness.)

---

## Citation

If you find this work useful, please cite:

```bibtex
@mastersthesis{kougioulis2025large,
  title   = {Large Causal Models for Temporal Causal Discovery},
  author  = {Kougioulis, Nikolaos},
  year    = {2025},
  month   = nov,
  address = {Heraklion, Greece},
  url     = {https://elocus.lib.uoc.gr/dlib/1/d/9/metadata-dlib-1764761882-792089-25440.tkl},
  note    = {Available at the University of Crete e-repository},
  school  = {Department of Computer Science, University of Crete},
  type    = {Master's Thesis}
}
```