# 🚀 Cooperative Multi-Task Semantic Communication with Implicit Optimal Priors (IoPm)

📄 This repository contains the **official simulation codes** for the published journal paper:

> **Semantic Communication for Cooperative Multi-Tasking Over Rate-Limited Wireless Channels With Implicit Optimal Prior**  
> *(A. Halimi Razlighi, C. Bockelmann, A. Dekorsy)*
> 
> *Published in: IEEE Open Journal of the Communications Society ( Volume: 6)*
> *Date of Publication: 02 October 2025*
> 
> *https://doi.org/10.1109/OJCOMS.2025.3617156*

In this work, we aim to tackle the rate-limit constraint, represented through the Kullback-Leibler (KL) divergence, by employing the density ratio trick alongside the implicit optimal prior method (IoPm). By applying the IoPm to our multi-task processing framework, we propose a hybrid-learning approach that combines deep neural networks with kernelized-parametric machine learning methods, enabling a robust solution for the CMT-SemCom.

The codes implement and evaluate **cooperative multi-task semantic communication (CMT-SemCom)** systems for rate-limited wireless channels under different prior assumptions, including **explicit priors** (e.g., standard Gaussian and log-uniform) and the **proposed Implicit Optimal Prior Method (IoPm)**.

The simulation results reported in the paper are generated using the codes provided in this repository.

---

## 📂 Repository Structure

The repository is organized into **four main folders**, corresponding to different datasets and prior modeling approaches:

CMT-SemCom_IoPm/
- `CMT_SemCom_CIFAR_EP/` – CIFAR-10 with explicit priors  
- `CMT_SemCom_CIFAR_IoPm/` – CIFAR-10 with proposed IoPm  
- `CMT_SemCom_MNIST_EP/` – MNIST with explicit priors  
- `CMT_SemCom_MNIST_IoPm/` – MNIST with proposed IoPm  



Each folder is **self-contained** and can be executed independently.

---

## 🧠 Multi-Task Learning Setup

The simulations mainly consider a **multi-task learning setup combining**:

- **Binary classification**
- **Categorical classification**

The framework is designed such that **tasks can be easily modified or extended**.


## 📁 Folder Descriptions

### 1. `CMT_SemCom_CIFAR_EP`
**CIFAR-10 | Explicit Priors (EP)**

This folder implements cooperative multi-task semantic communication on the **CIFAR-10 dataset** using **widely adopted explicit priors**, namely:

- Standard Gaussian prior  
- Log-uniform prior  

#### Prior Selection

The prior is selected inside the following file:

```CUSUModel.py```


- To use the **standard Gaussian prior**, uncomment the following line:
```python
# kl = gaussian_kl_divergence(x2_mu, x2_ln_var)
```

- To use the log-uniform prior, comment out the Gaussian KL term and enable the corresponding log-uniform implementation in the same file.

⚠️ **Important**: Only one prior should be active at a time.

### 2. `CMT_SemCom_CIFAR_IoPm`
**CIFAR-10 | Implicit Optimal Prior Method (IoPm)**

This folder contains the implementation of the proposed IoPm for multi-tasking.

Key characteristics:

- No explicit prior assumption (e.g., Gaussian or log-uniform)
- The optimal prior is learned implicitly from data using **logistic regression (LR)**
- Fully aligned with the proposed method described in the paper

This folder is used to generate the IoPm-based results for the CIFAR-10 experiments.

### 3. `CMT_SemCom_MNIST_EP`
**MNIST | EP**

This folder mirrors the functionality of `CMT_SemCom_CIFAR_EP`, but uses the **MNIST dataset** instead.

- Supports standard Gaussian and log-uniform priors
- Prior selection is performed via `CUSUModel.py`
- Used for baseline comparisons on MNIST

### 4. `CMT_SemCom_MNIST_IoPm`
**MNIST | IoPm**

This folder mirrors `CMT_SemCom_CIFAR_IoPm`, but for the **MNIST dataset**.

- Implements the proposed IoPm approach
- Used to generate MNIST results reported in the paper

## 🧩 Code Organization (Inside Each Folder)
Each folder contains **five Python files**, organized as follows:

- `datasets.py`
  
   Preprocesses the datasets for **multi-label, multi-task learning**.
   The task definitions are modular, allowing readers to easily modify or redefine tasks.

- `CUSUModel.py`
  
  Defines the **encoder–decoder architectures, loss functions, training loop, and evaluation loop**.
  In addition to the cooperative multi-task architecture, this file also includes a **single-task encoder–decoder**, which is used to compare single-task processing against cooperative multi-task learning.

- `utils.py`
  
  Contains required utility functions, including the implementation of the **machine-learning-based logistic regression (LR).
  The noise power of the communication channel can also be configured here.

- `resultPloting.py`
  
  Used to plot and visualize simulation results.

- `main.py`
  
  Specifies the **training configuration**, such as:
  - Number of iterations
  - Learning rate
  - Other simulation parameters


## ▶️ How to Run the Code

Each folder is **independently executable**.

1. Navigate to the desired folder:
```cd CMT_SemCom_CIFAR_EP```
2. Run the main
```python main.py```


## 📊 Reproducing the Paper Results

The results presented in the paper are obtained by:
- Running simulations in the **EP** folders
- Running simulations in the corresponding **IoPm** folders
- Combining and comparing the outputs.
- 

## 📎 Citation
**Please do not forget to cite us when you use the code/paper:**

@ARTICLE{11190017,

  author={Halimi Razlighi, Ahmad and Bockelmann, Carsten and Dekorsy, Armin},
  
  journal={IEEE Open Journal of the Communications Society},
  
  title={Semantic Communication for Cooperative Multi-Tasking Over Rate-Limited Wireless Channels With Implicit Optimal Prior}, 
  
  year={2025},
  
  volume={6},
  
  number={},
  
  pages={8523-8538},
  
  keywords={Multitasking;Semantic communication;Wireless communication;Artificial neural networks;Knowledge graphs;Accuracy;Probabilistic logic;Linear programming;Transmitters;Symbols;Cooperative multi-tasking;deep learning;hybrid learning;information theory;implicit optimal prior;parametric methods;semantic communication},
  
  doi={10.1109/OJCOMS.2025.3617156}}
