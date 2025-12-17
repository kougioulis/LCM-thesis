# Large Causal Models for Temporal Causal Discovery

<div style="white-space: nowrap;">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
  <img src="https://img.shields.io/badge/PyTorch-black?logo=PyTorch">
  <img src="https://img.shields.io/badge/-Numpy-013243?&logo=NumPy">
  <img src="https://img.shields.io/badge/Scikit--learn-20232A?&logoColor=61DAFB">
  <img src="https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas">
  <a href="https://github.com/gkorgkolis/TCS/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  </a>
  <a href="https://elocus.lib.uoc.gr/dlib/1/d/9/metadata-dlib-1764761882-792089-25440.tkl">
    <img src="https://img.shields.io/badge/e--Locus-archive-9D2235?style=flat">
  </a>
</div>


Reproducibility and source for experiments of the Thesis "Large Causal Models for Temporal Causal Discovery" at the University of Crete.

Complete LaTeX source of the thesis text is available at: https://github.com/kougioulis/thesis.


## 📌 Overview

Causal discovery for both cross-sectional and temporal data has traditionally followed a dataset-specific paradigm, where a new model is fitted for each individual dataset. Such an approach underutilizes the potential of multi-dataset and large-scale pretraining, especially given recent advances in foundation models. The concept of *Large Causal Models (LCMs)* envisions a class of pre-trained neural architectures specifically designed for temporal causal discovery. Existing approaches remain largely proofs of concept, typically constrained to small input sizes (e.g., five variables), with performance degrading rapidly to random guessing as the number of variables or model parameters increases. Moreover, current methods rely heavily on synthetic data, generated under arbitrary assumptions, which substantially limits their ability to generalize to realistic or out-of-distribution samples. This work addresses these challenges through novel methods for training on mixtures of synthetic and realistic data collections, enabling both higher input dimensionality and deeper architectures without loss of performance. Extensive experiments demonstrate that LCMs achieve competitive or superior performance compared to classical causal discovery algorithms, while maintaining robustness across diverse domains, especially on non-synthetic data cases. Our findings also highlight promising directions towards integrating interventional samples and domain knowledge, further advancing the development of foundation models for causal discovery.

## Contributions

Contributions of this dissertation can be summarized as follows:

- Introduced a family of pre-trained neural architectures (**Large Causal Models - LCMs**) capable of performing causal discovery on time-series data, scaling to significantly larger numbers of variables than prior work, without degradation in performance compared to prior work.

- Developed a pipeline for **generating synthetic data of high fidelity** (time-series samples derived from a ground truth temporal SCM).

- Introduced a **novel AutoML-based generative method (Temporal Causal-based Simulation - TCS) for creating realistic (simulated) causal models and coressponding data samples from real, multivariate time-series**.

- Proposed an **adversarial discriminator methodology using Classifier 2-Sample Tests - C2STs (Adversarial Causal Tuning - ACT)** for *optimal causal model selection from TCS** and avoid selection of degenerate, fully-connected graps **(sparsity penalty)**.

- Generated **hundreds of thousands of training pairs** from mixtures of synthetic and simulated datasets, enabling robust multi-domain pretraining of LCMs.

- Showcased that **training of causal foundation models benefits from a mixture of both synthetic and realistic data**, as illustrated in foundation models for time-series forecasting. Identified the optimal ratio for this mixture, which improves generalization to semi-synthetic and realistic benchmarks.

- **Proposed a lagged-correlation-informed regularization technique** that stabilizes training and suppresses low-support edges in predicted causal graphs, motivated by prior work, and showcased that the **use of observed statistics (training aids) during both training and inference of causal foundation models improves performance**.

- **Compared LCMs against well-known causal discovery methods** (e.g., PCMCI, DYNOTEARS) and demonstrated **competitive or superior performance** across synthetic, semi-synthetic and realistic datasets, maintaining robustness under domain shifts and distributions outside the training set; one of the first results for causal foundation models in temporal data.

- Achieved **significantly faster inference time** than existing non-foundation temporal causal discovery algorithms.

- Designed a novel transformer-based model using *patch embeddings, variable embeddings, and multi-head temporal–spatial attention for improved expressivity* and generalization **(preliminary experiments)**.

- Proposed promising approaches for *incorporating interventional data in foundation models for causal discovery* **(preliminary experiments)**.

- Designed promising mechanisms for *representation and training to integrate prior domain knowledge*, as a soft auxiliary task **(preliminary experiments)**.

## 🧪 Setup & Getting Started 

### 🐍 Conda Environment

We provide a conda environment for reproducibility purposes. One can create a virtual conda environment using

- `conda env create -f environment.yaml`
- `conda activate LCM` 

Alternatively, you can just install the dependencies from the `requirements.txt` file, either on your base environment or into an existing conda environment using

`pip install -r requirements.txt`

## 📔 Available Notebooks

Notebooks for reproducible experiments and demo scripts (`running_examples.ipynb`) are available in the `code/notebooks/` folder. Experimental results in `.csv` form presented in the Thesis text are available in `code/data/results/`.

  - `experimental_results.ipynb` contains the experimental results of Section ..

  - `illustrative_example.ipynb` contains an example of loading a pre-trained LCM, preprocessing a minimal input time-series data and performing causal discovery. It illustrates both the discovered lagged causal graph, as well as the confidence weights of the lagged adjacency tensor.

  - `ablation_experiments.ipynb` contains ablation experiments (Section ..) and zero-shot experiments on assessing the optimal mixture of realistic and synthetic training data (Section ..).

## 📁 Structure

## ✨ Pretrained Weights

Pre-trained LCM models presented in the thesis are provided outside of GitHub due to size constraints, in the following Google Drive links:

- [LCM_2.5M](https://drive.google.com/file/d/...)
- [LCM_9.4M](https://drive.google.com/file/d/...)
- [LCM_12.2M](https://drive.google.com/file/d/...)
- [LCM_24M](https://drive.google.com/file/d/...)

## Datasets (Test Sets)

We additionally provide the test sets for our evaluations, available via Google Drive links and to be placed in the data folder.

### Synthetic

    - S_Joint: 
    - Synth_230K: 

### Mixture Collection

    - Synth_230K_Sim_45K:

### Semi-Synthetic

Semi-synthetic test data are provided in the `data` folder.

    - $f$MRI-5: 
    - $f$MRI: 
    - Kuramoto-5: 
    - Kuramoto-10: 

### Simulated (Realistic data)

    - Sim_45K
    - AirQualityMS

## Citing

If this work has proven useful, please consider citing:

```bibtex
@mastersthesis{kougioulis2025large,
  title = {Large Causal Models for Temporal Causal Discovery},
  author = {Kougioulis, Nikolaos},
  year = {2025},
  month = nov,
  address = {Heraklion, Greece},
  url = {https://elocus.lib.uoc.gr/dlib/1/d/9/metadata-dlib-1764761882-792089-25440.tkl},
  note = {Available at the University of Crete e-repository},
  school = {Department of Computer Science, University of Crete},
  type = {Master's Thesis},
}