# Radial Müntz–Szász Networks (RMN)

This repository provides a reference implementation of **Radial Müntz–Szász Networks (RMN)**,
a class of neural architectures designed to learn **multidimensional radial singularities**
using **learnable power bases** and a **numerically stable log-primitive formulation**.

RMN generalizes Müntz–Szász Networks from one-dimensional coordinate power laws to
**truly radial representations**, enabling accurate approximation of functions such as

- \( r^\alpha \) for arbitrary (including negative) exponents,
- \( \log r \),
- mixed and multi-source singular fields,

which are not representable by coordinate-separable models.

The method is particularly suited for scientific machine learning, physics-informed learning,
and PDE problems posed on **punctured domains**.

---

## Paper

**Radial Müntz–Szász Networks: Neural Architectures with Learnable Power Bases for Multidimensional Singularities**  
**Gnankan Landry Regis N’guessan**, **Bum Jun Kim**

The preprint version of the paper is available on Arxiv.

---

## Key idea

Classical neural architectures (MLP, SIREN, coordinate-wise power networks) are **coordinate-separable**.
As a result, they cannot represent non-quadratic radial functions of the form \( f(x)=g(\|x\|) \).

RMN resolves this limitation by:
1. Explicitly parameterizing **radial power bases**,
2. Learning the exponents directly from data or PDE constraints,
3. Supporting **negative and logarithmic singularities**,
4. Providing closed-form gradients and Laplacians.

This makes RMN both **interpretable** and **numerically robust** near singularities.

---

## Architecture variants

- **RMN-Direct** – single-center radial expansion  
- **RMN-Angular** – radial powers with angular modulation  
- **RMN-MC** – multi-center radial expansions  

Baselines include MLP, SIREN, RBF, and coordinate-wise Müntz–Szász Networks.

---
## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quickstart

Fit \(\log r\) on a punctured 2D domain:

```bash
python examples/quickstart_fit_logr.py
```

## What is inside

- `rmn/parameterizations.py`: differentiable cumulative-gap exponent parameterization
- `rmn/utils.py`: stable power evaluation, log-primitive, (2D/3D) angular bases
- `rmn/layers.py`: RMN edges (direct, angular, multi-center)
- `rmn/models.py`: user-facing models (`RMNDirect`, `RMNAngular`, `RMNMultiCenter`)
- `rmn/training.py`: minimal function-fitting loop for quick replication


## Citation

Will be updated soon
