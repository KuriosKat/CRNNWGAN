<div align="center">

# CRNNWGAN

### *Data Augmentation for Power Factor Correction Fault Classification: A GANs Approach*

[![IEEE IV 2025](https://img.shields.io/badge/IEEE_IV-2025-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://ieee-iv.org)
[![Conference](https://img.shields.io/badge/Conference-IEEE_Intelligent_Vehicles_Symposium-003087?style=for-the-badge)](https://ieee-iv.org)
[![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

> **Official implementation** of the paper published at **2025 IEEE Intelligent Vehicles Symposium (IEEE IV)**  
> *"Data Augmentation for Power Factor Correction Fault Classification: A GANs Approach"*

</div>

---

## 📄 Paper

> Yi-Hyeong Park, Dong-In Lee, Han-Shin Youn, Chang Mook Kang,  
> **"Data Augmentation for Power Factor Correction Fault Classification: A GANs Approach"**,  
> *2025 IEEE Intelligent Vehicles Symposium (IEEE IV)*, 2025.

**Funding:** This work was supported by KEIT grant funded by MOTIE (Korea) under Grant No. RS-2024-00443216,  
*"Development and PoC of On-Device AI Computing-based AI Fusion Mobility Device."*

---

## 🔍 Abstract

The growing adoption of electric vehicles (EVs) has heightened the need for reliable On-Board Chargers (OBCs). Power Factor Correction (PFC) circuits within OBCs are critical for optimizing energy conversion, but **fault diagnosis remains challenging** due to the difficulty of replicating real-world fault scenarios for data collection.

This work proposes **CRNNWGAN** — a novel fusion of **C-RNN-GAN** and **WGAN-GP** — that augments fault signal data by generating diverse and realistic synthetic fault signals. Experimental results confirm that the augmented dataset significantly outperforms training on real-world data alone.

---

## 🚗 Motivation & Problem Statement

```
Real-world PFC fault data collection challenges:
  ├── ⚠️  Safety risk: deliberately inducing MOSFET faults is dangerous
  ├── 💸  High cost: physical fault replication requires expensive hardware
  ├── 📉  Data imbalance: fault samples (×100) << normal samples (×1,000)
  └── 🔁  Limited diversity: hard to reproduce all fault variations
```

In PFC circuits, **MOSFET failures** — either **open faults** or **short faults** — are the most critical failure modes:

| Fault Type | Impact on Circuit | Consequences |
|------------|-------------------|--------------|
| **Open Fault** | Breaks current path, reduces energy efficiency | Operational interruption, increased power costs |
| **Short Fault** | Causes excessive heat, risk of fire | MOSFET damage, safety hazard, high maintenance cost |

---

## 🏗️ Architecture: CRNNWGAN

CRNNWGAN is a **novel hybrid GAN** that combines:

| Component | Origin | Role |
|-----------|--------|------|
| **C-RNN-GAN** | Mogren, 2016 | LSTM-based generator to capture temporal dependencies in sequential signals |
| **WGAN-GP** | Gulrajani et al., 2017 | Wasserstein distance + gradient penalty for stable training, prevents mode collapse |

### Why this fusion?

```
C-RNN-GAN alone:      Good temporal modeling  ✅   Unstable training      ❌
WGAN-GP alone:        Stable training         ✅   Weaker sequence model  ❌
──────────────────────────────────────────────────────────────────────────
CRNNWGAN (ours):      Good temporal modeling  ✅   Stable training        ✅
```

### Architecture Diagram

```
                    ┌──────────────────────────────────────────────┐
                    │              CRNNWGAN                        │
                    │                                              │
  prev_x ──────────►│                                              │
  next_x ──────────►│   Generator G(θ_G)                          │
  noise z ──────────►│   ┌─────────────────────┐                  │
                    │   │  LSTM layers         │──► x_fake        │
                    │   │  (temporal modeling) │         │         │
                    │   └─────────────────────┘         │         │
                    │                                    ▼         │
  x_real ──────────────────────────────────► Critic C(θ_C)        │
  (x_mid)          │                         ┌──────────────────┐ │
                    │                         │ Wasserstein dist │ │
                    │                         │ + Gradient       │ │
                    │                         │   Penalty (GP)   │ │
                    │                         └────────┬─────────┘ │
                    │                                  │           │
                    │              ┌───────────────────┘           │
                    │              ▼                               │
                    │   Update θ_C (critic loss)                   │
                    │   Update θ_G (generator loss)                │
                    │                                              │
                    └──────────────────────────────────────────────┘
                                         │
                                         ▼
                          Augmented Signal = Real + Fake
```

---

## 📊 Dataset

### Experimental Setup

Data was collected via **PSIM simulation** of a three-phase PFC topology with **6 MOSFETs** (P1–P6), each subjected to open and short fault conditions.

```
Fault Points: P1, P2, P3, P4, P5, P6
Fault Types:  open, short
─────────────────────────────────────────────
Total Fault Classes: 6 × 2 = 12 + 1 (Normal) = 13 classes
```

### Sampling Strategy (Realistic Imbalance)

| Condition | Sampling Rate | Samples | Reason |
|-----------|--------------|---------|--------|
| **Normal** | 32 kHz (Δt = 3.125×10⁻⁵ s) | 1,000 | Easy to collect in real environments |
| **Pn_open** | ~3.2 kHz (Δt = 3.125×10⁻⁴ s) | 100 | Hard to replicate safely |
| **Pn_short** | ~3.2 kHz (Δt = 3.125×10⁻⁴ s) | 100 | Hard to replicate safely |

> This 10:1 imbalance between normal and fault data directly reflects **real industrial constraints**, and is the core problem CRNNWGAN is designed to solve.

### Fault Signal Characteristics

| Condition | Peak Current (Ampere sensor) |
|-----------|------------------------------|
| Normal | ~8.6 × 10⁻⁹ A (low amplitude, wide scatter) |
| P1 open | ~6.64 × 10⁻⁸ A (elevated, distinct from normal) |
| P1 short | ~4.21 × 10⁻⁷ A (highest, most distinct) |

---

## ⚙️ CRNNWGAN Training Procedure

### Algorithm (Pseudocode from Paper)

```python
procedure TRAIN-CRNNWGAN(D, θ_G, θ_C):
    Initialize generator G(θ_G), critic C(θ_C)
    Set hyperparameters: numEpochs, batchSize, critic_iter, λ_GP

    for epoch in range(numEpochs):
        for each mini-batch x in D:

            # ── Step 1: Split sequence ──────────────────────────────
            x_prev, x_mid, x_next = split(x)

            # ── Step 2: Generate fake middle point ──────────────────
            z ~ N(0, I)                         # Sample Gaussian noise
            x_fake = G(x_prev, x_next, z)       # LSTM generator

            # ── Step 3: Update Critic (×critic_iter times) ──────────
            loss_real = C(x_prev, x_mid,  x_next)
            loss_fake = C(x_prev, x_fake, x_next)

            # Wasserstein distance + Gradient Penalty
            gp = gradient_penalty(x_mid, x_fake)
            loss_C = loss_fake - loss_real + λ * gp
            update θ_C ← minimize loss_C

            # ── Step 4: Update Generator ────────────────────────────
            if critic_update_done:
                loss_G = -C(x_prev, x_fake, x_next)
                update θ_G ← minimize loss_G

    return G(θ_G), C(θ_C)
```

### Key Design Choices

**Generator (G)**
- Uses **LSTM layers** to model temporal dependencies in fault signals
- Input: `[prev_x, next_x, noise_z]` — context-aware generation of the middle segment
- Output: `x_fake` — synthetic fault signal segment

**Critic (C)**
- Replaces traditional discriminator with a **Wasserstein critic** (no sigmoid output)
- Measures **Earth Mover's Distance** between real and generated distributions
- Enforces **Lipschitz constraint** via gradient penalty (not weight clipping)

**Gradient Penalty**

$$\mathcal{L}_{GP} = \lambda \cdot \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} C(\hat{x})\|_2 - 1\right)^2\right]$$

where $\hat{x}$ is sampled uniformly along lines between real and fake samples.

---

## 📈 Experimental Results

### Environment

| Spec | Details |
|------|---------|
| CPU | AMD Ryzen 5600G — 6 Cores, 12 Threads, 3.9 GHz |
| RAM | 64 GB DDR4 2400 MHz (16 GB × 4) |
| OS | Windows 10 64-bit |
| Python | 3.8 |

### Result 1 — MMD (Maximum Mean Discrepancy)

MMD measures how closely synthetic data matches the **real data distribution** (lower = better fit, but not always better classification).

$$\text{MMD}^2(X,Y) = \frac{1}{m^2}\sum_{i,j}k(x_i,x_j) + \frac{1}{n^2}\sum_{i,j}k(y_i,y_j) - \frac{2}{mn}\sum_{i,j}k(x_i,y_j)$$

$$k(x,y) = \exp\left(-\gamma\|x-y\|^2\right), \quad \gamma = 1/\text{dim}$$

| Model | Average MMD |
|-------|------------|
| C-RNN-GAN | **0.111** (lowest — closest fit) |
| WGAN-GP | 0.207 |
| **CRNNWGAN (ours)** | 0.226 |

> ⚠️ Note: CRNNWGAN expands 100 → 1,000 samples (×10 augmentation factor). A higher MMD is expected at this scale, yet it achieves **superior classification accuracy** — demonstrating that richer diversity matters more than distributional proximity alone.

### Result 2 — Classification Accuracy (LazyPredict)

Augmented datasets evaluated across multiple classifiers using **LazyPredict**:

| Dataset | Top-5 Avg. Accuracy | Top-10 Avg. Accuracy |
|---------|--------------------|--------------------|
| Original | 0.630 | 0.617 |
| C-RNN-GAN | 0.630 | 0.610 |
| WGAN-GP | **1.000** | 0.960 |
| **CRNNWGAN (ours)** | **1.000** | **0.980** |

**Key takeaway:** CRNNWGAN achieves **+58.7% accuracy gain** (Top-5) over the original dataset, and consistently outperforms all baselines across Top-10 classifiers — demonstrating both power and robustness of the augmentation.

---

## 🛠️ Tech Stack

```
Deep Learning   : PyTorch
  ├── Generator : LSTM (sequence modeling)
  └── Critic    : Linear layers + Gradient Penalty

Signal Simulation : PSIM (PowerSim)
Feature Evaluation: t-SNE (2D embedding) + RBF Kernel (MMD)
Classification    : LazyPredict (multi-classifier benchmark)
```

---

## 📂 Fault Class Reference

| Fault Label | Description |
|-------------|-------------|
| `Normal` | No fault — clean PFC operation |
| `P1_open` ~ `P6_open` | Open-circuit fault at MOSFET P1–P6 |
| `P1_short` ~ `P6_short` | Short-circuit fault at MOSFET P1–P6 |

---

## 🔗 Citation

If you use this code or paper in your research, please cite:

```bibtex
@inproceedings{park2025crnnwgan,
  title     = {Data Augmentation for Power Factor Correction Fault Classification: A GANs Approach},
  author    = {Yi-Hyeong Park and Dong-In Lee and Han-Shin Youn and Chang Mook Kang},
  booktitle = {2025 IEEE Intelligent Vehicles Symposium (IEEE IV)},
  year      = {2025}
}
```

---

## 📬 Contact

| | |
|--|--|
| **First Author** | Yi-Hyeong Park |
| **Affiliation** | Dept. of Electrical Engineering, Hanyang University, Seoul, Korea |
| **Email** | robo20001117@hanyang.ac.kr |
| **GitHub** | [@KuriosKat](https://github.com/KuriosKat) |

---

<div align="center">

*Published at 2025 IEEE Intelligent Vehicles Symposium (IEEE IV)*  
*Hanyang University · Seoul, Republic of Korea*

</div>
