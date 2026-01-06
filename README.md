# Cooperative Multi-Task Semantic Communication with Implicit Optimal Priors (IoPm)

This repository contains the **official simulation codes** for the published journal paper:

> **Semantic Communication for Cooperative Multi-Tasking Over Rate-Limited Wireless Channels With Implicit Optimal Prior**  
> *(A. Halimi Razlighi, C. Bockelmann, A. Dekorsy)*

> *Published in: IEEE Open Journal of the Communications Society ( Volume: 6)*
> *Date of Publication: 02 October 2025*
> *https://doi.org/10.1109/OJCOMS.2025.3617156*

In this work, we aim to tackle the rate-limit constraint, represented through the Kullback-Leibler (KL) divergence, by employing the density ratio trick alongside the implicit optimal prior method (IoPm). By applying the IoPm to our multi-task processing framework, we propose a hybrid-learning approach that combines deep neural networks with kernelized-parametric machine learning methods, enabling a robust solution for the CMT-SemCom.

The codes implement and evaluate **cooperative multi-task semantic communication (CMT-SemCom)** systems for rate-limited wireless channels under different prior assumptions, including **explicit priors** (e.g., standard Gaussian and log-uniform) and the **proposed Implicit Optimal Prior Method (IoPm)**.

The simulation results reported in the paper are generated using the codes provided in this repository.

---

## Repository Structure

The repository is organized into **four main folders**, corresponding to different datasets and prior modeling approaches:

CMT-SemCom_IoPm/
│
├── CMT_SemCom_CIFAR_EP/ # CIFAR-10 with explicit priors
├── CMT_SemCom_CIFAR_IoPm/ # CIFAR-10 with proposed IoPm
├── CMT_SemCom_MNIST_EP/ # MNIST with explicit priors
└── CMT_SemCom_MNIST_IoPm/ # MNIST with proposed IoPm


Each folder is **self-contained** and can be executed independently.

---

## Folder Descriptions

### 1. `CMT_SemCom_CIFAR_EP`
**CIFAR-10 | Explicit Priors (EP)**

This folder implements cooperative multi-task semantic communication on the **CIFAR-10 dataset** using **widely adopted explicit priors**, namely:

- Standard Gaussian prior  
- Log-uniform prior  

#### Prior Selection

The prior is selected inside the following file:

CUSUModel.py


- To use the **standard Gaussian prior**, uncomment the following line:
```python
# kl = gaussian_kl_divergence(x2_mu, x2_ln_var)


To use the log-uniform prior, comment out the Gaussian KL term and enable the corresponding log-uniform implementation in the same file.

Important: Only one prior should be active at a time.

