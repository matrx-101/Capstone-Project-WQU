# Cardinality-Constrained Index Tracking  
### Convex Optimization, Metaheuristics & Hybrid Initialization

---

## Project Overview

This project studies the **Cardinality-Constrained Index Tracking Problem** from a regression perspective.

The objective is to construct a portfolio of **K assets** that closely replicates a benchmark index while minimizing tracking error, under realistic investment constraints.

The project compares:

- Classical Convex Optimization (Dense QP)
- L1-Regularized Index Tracking
- L2-Regularized (Ridge) Index Tracking
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Hybrid Convex-Initialized Metaheuristics
- Overfitting & Generalization Analysis
- Dimension Scaling Behaviour

The empirical study is conducted on benchmark datasets from the **OR Library (Beasley Index Tracking datasets)**.

---

## Mathematical Formulation

### Portfolio Return

For asset return matrix R ∈ R^(T × N) and weights w ∈ R^N:

r^p = R w

---

## Tracking Error

TE = (1 / T) || R w − r^I ||_2^2

Where:

- r^I = index return vector  
- T = number of time periods  

---

## Constraints

### Budget Constraint

Sum_{i=1}^N w_i = 1  

### Long-Only Constraint

w_i ≥ 0  

### Cardinality Constraint

Sum_{i=1}^N Z_i = K  

Z_i ∈ {0, 1}

---

# Models Implemented

## Convex Baseline Models

### 1. Dense Quadratic Programming (QP)

- Classical least squares formulation  
- Long-only constraint  
- Budget constraint enforced  

Objective:

min_w (1/T) || R w − r^I ||_2^2  

---

### 2. L1-Regularized Model

Soft sparsity via L1 penalty:

min_w (1/T) || R w − r^I ||_2^2 + λ || w ||_1  

- Encourages shrinkage  
- Produces sparse-like solutions  
- Still convex  

---

### 3. L2-Regularized Model (Ridge)

min_w (1/T) || R w − r^I ||_2^2 + λ || w ||_2^2  

- Stabilizes multicollinearity  
- Reduces estimation variance  
- Improves conditioning in high dimensions  

All convex subproblems are solved using the **OSQP** solver.

---

# Discrete Metaheuristic Models

Because the cardinality constraint creates a combinatorial search space of size (N choose K), the problem becomes NP-Hard.

To handle this, we implement two discrete metaheuristic approaches.

---

## Genetic Algorithm (GA)

- Chromosome = subset S = {i1, i2, ..., iK}  
- Fitness function = in-sample tracking error  

For a given subset S:

min_{w_S} (1/T) || R_S w_S − r^I ||_2^2  

Subject to:

Sum w_S = 1  
w_S ≥ 0  

GA Operators:

- Random subset initialization  
- Tournament selection  
- Crossover via union sampling  
- Mutation for diversification  
- Elitism retention  

Lower fitness score → better index replication.

---

## Particle Swarm Optimization (PSO)

- Discrete subset-based PSO  
- No velocity vector (combinatorial version)  
- Particle = subset S  

Fitness:

F(S) = (1/T) || R_S w_S − r^I ||_2^2  

Update rule (probabilistic):

With probability c1 → move toward personal best (pbest)  
With probability c2 → move toward global best (gbest)  
Otherwise → retain current position  

---

# Hybrid Initialization Strategy

Instead of purely random initialization, we construct a probability distribution using dense convex weights.

Dense initializer w^(0) is obtained from:

- Classical QP  
- L1-Regularized QP  
- L2-Regularized QP  

Probability distribution:

p_i = |w_i^(0)| / Sum_j |w_j^(0)|  

With smoothing:

p_i = (1 − ε) p_i + (ε / N)

This produces hybrid variants:

- HGA-QP  
- HGA-L1  
- HGA-L2  
- HPSO-QP  
- HPSO-L1  
- HPSO-L2  

Hybrid initialization improves convergence stability and reduces premature random exploration in high-dimensional problems.

---


## How to Run This Project Locally

Follow the steps below to clone and execute the project on your local machine.

### Clone the Repository

```bash
git clone https://github.com/matrx-101/Capstone-Project-WQU.git
cd Capstone-Project-WQU

python3 -m venv venv
source venv/bin/activate     # Mac/Linux
# venv\Scripts\activate      # Windows
pip install -r requirements.txt
python3 -m run.py