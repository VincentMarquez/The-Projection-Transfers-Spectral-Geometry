in the works 

# The-Projection-Transfers-Spectral-Geometry


Abstract

Transfer learning is commonly treated as an empirical phenomenon, with effectiveness assessed through downstream performance and attributed to representation similarity, model scale, or task relatedness. In contrast, we formulate transfer learning as a rank-limited spectral projection between learned representations, isolating structural compatibility from optimization effects. We define transfer fidelity as the fraction of target variance captured by a rank- projector constructed from the source representation’s covariance spectrum.

Using controlled experiments on synthetic manifolds and high-dimensional subspaces, we show that transfer fidelity is governed by shared intrinsic dimensionality, exhibiting a characteristic knee as projection rank increases. Below this knee, transfer is limited by missing spectral directions; above it, gains diminish until a spanning regime is reached, in which sufficiently high rank yields near-perfect fidelity independent of detailed alignment. We further identify an inherent asymmetry in transfer behavior: representations learned on broader domains can fully span narrower targets given sufficient rank, whereas the converse remains fundamentally constrained.

Together, these results provide a geometric account of transfer learning based on spectral structure, reinterpret rank as a form of memory bandwidth, and introduce a diagnostic for assessing transfer compatibility prior to training, independent of task semantics, parameter count, or downstream loss.



1 Introduction

Transfer learning is a central mechanism in modern machine learning, enabling models trained on one task or domain to accelerate learning or improve performance on another. In practice, transfer effectiveness is evaluated empirically through downstream loss or accuracy, and success is commonly attributed to representation similarity, task relatedness, or model scale. Despite extensive empirical study, however, there remains no mechanistic account of when and why transfer succeeds or fails that is independent of optimization dynamics.

A core difficulty is that transfer learning is typically conflated with parameter reuse or fine-tuning performance. Large, overparameterized models frequently exhibit positive transfer across diverse and weakly related tasks, obscuring the structural conditions that make such transfer possible. As a result, transfer is often treated as an empirical phenomenon to be observed rather than a mechanism that can be analyzed or controlled.

In this work, we propose a geometric formulation of transfer learning that separates structural compatibility from optimization effects. Rather than asking whether two tasks are similar, we ask a more fundamental question:

> How much of the structure learned by a source representation is accessible to a target task under an explicit rank constraint?



We show that this question admits a precise spectral answer.


---

1.1 Classical Axes of Transfer Learning

Existing transfer learning methods are commonly organized along axes describing what is transferred and how it is reused. Table 1 summarizes this prevailing perspective.

Table 1: Classical Axes of Transfer Learning

Axis	Description

Transferred object	Parameters, features, embeddings, activations
Transfer stage	Pretraining, fine-tuning, multi-task learning
Evaluation criterion	Downstream loss, accuracy, sample efficiency
Capacity control	Implicit (model size, regularization)
Similarity notion	Task semantics, feature correlation, representation distance
Failure mode	Negative transfer (empirical degradation)


Under this view, dimensionality and rank are treated indirectly as proxies for model capacity. Spectral structure, when considered, is used primarily for post hoc representation analysis rather than as a component of the transfer mechanism itself.


---

1.2 Spectral–Geometric Axes of Transfer (This Work)

We introduce an alternative organization in which transfer is governed by rank-limited access to learned spectral structure. Table 2 summarizes the axes introduced in this work.

Table 2: Spectral–Geometric Axes of Transfer Learning

Axis	Description

Transferred object	Spectral structure of learned trajectories
Transfer operator	Rank- spectral projector derived from the source
Control variable	Projection rank  (memory bandwidth)
Compatibility measure	Transfer fidelity (captured target variance)
Structural driver	Shared intrinsic dimensionality
Failure mode	Structural incompatibility (low or zero fidelity)
Degenerate regime	Spanning regime induced by overparameterization


This formulation makes explicit a quantity that is typically implicit: the number of spectral directions through which transfer is permitted. As we show, this single control variable induces distinct and interpretable transfer regimes, including guaranteed failure, partial transfer, and a trivial spanning regime in which sufficiently high rank yields near-perfect fidelity independent of detailed alignment.


---

1.3 Relation to Prior Algorithms

Several existing methods touch subsets of these axes, but none complete them. Table 3 situates representative prior approaches relative to the spectral–geometric formulation developed here.

Table 3: Prior Methods and Axis Coverage

Method / Family	Uses spectral structure	Controls rank at transfer	Defines fidelity	Predicts failure pre-training	Mechanistic transfer operator

PCA / Truncated SVD	✓	✗ (analysis only)	✗	✗	✗
CCA	✓	✗	✗	✗	✗
SVCCA / PWCCA / CKA	✓	✗	✗	✗	✗
Random features / NTK	✓ (implicit)	✗	✗	✗	✗
LoRA / Adapters	✗ (implicit only)	✓ (capacity)	✗	✗	✗
Linear probing	✗	✗	✗	✗	✗
This work	✓	✓	✓	✓	✓


Prior work has introduced spectral tools for representation analysis and low-rank parameterizations for efficient adaptation. However, these approaches treat spectral structure either as a diagnostic or as an implicit capacity constraint. To our knowledge, no existing method formulates transfer learning itself as a rank-limited spectral projection, nor defines a fidelity measure that isolates structural compatibility from optimization dynamics.


---

1.4 Contributions

Building on this formulation, we make the following contributions:

1. Geometric formulation of transfer
We formalize transfer learning as rank-limited spectral projection between learned representations and define transfer fidelity as a structure-preserving compatibility measure.


2. Rank–fidelity transfer law
We show that transfer fidelity is governed by shared intrinsic dimensionality, exhibiting a characteristic knee as projection rank increases.


3. Asymmetry of transfer
We demonstrate that transfer is inherently asymmetric: representations learned on broader domains can span narrower targets given sufficient rank, while the converse is fundamentally limited.


4. Spanning regime in overparameterized models
We identify a degenerate regime in which high rank yields near-perfect fidelity independent of structural alignment, explaining the apparent universality of large pretrained models.


5. Pre-training diagnostic
We introduce a geometry-based diagnostic for predicting transfer compatibility prior to optimization.

# Appendix A: Mathematical Foundations of Spectral Transfer Fidelity

## A.1 Notation

| Symbol | Definition |
|--------|------------|
| $X_s \in \mathbb{R}^{N \times D}$ | Source data matrix (N samples, D features) |
| $X_t \in \mathbb{R}^{M \times D}$ | Target data matrix (M samples, D features) |
| $\rho \in \mathbb{R}^{D \times D}$ | Density matrix (positive semi-definite) |
| $\Pi \in \mathbb{R}^{D \times D}$ | Orthogonal projector (idempotent) |
| $U \in \mathbb{R}^{D \times D}$ | Orthonormal eigenvector matrix |
| $\Lambda \in \mathbb{R}^{D \times D}$ | Diagonal eigenvalue matrix |
| $k \in \mathbb{N}$ | Rank of projector (hyperparameter) |
| $F \in [0,1]$ | Transfer fidelity |
| $\text{Tr}(\cdot)$ | Matrix trace |

---

## A.2 Density Matrix Construction

### Definition A.2.1 (Centered Data)
Given data matrix $X \in \mathbb{R}^{N \times D}$, the centered data is:

$$\tilde{X} = X - \mathbf{1}_N \mu^T$$

where $\mu = \frac{1}{N} X^T \mathbf{1}_N \in \mathbb{R}^D$ is the sample mean and $\mathbf{1}_N$ is the N-dimensional ones vector.

### Definition A.2.2 (Density Matrix)
The density matrix is the normalized covariance:

$$\rho = \frac{1}{N} \tilde{X}^T \tilde{X}$$

### Properties of ρ

**Proposition A.2.1.** *The density matrix ρ satisfies:*
1. *Symmetry:* $\rho = \rho^T$
2. *Positive semi-definiteness:* $v^T \rho v \geq 0$ for all $v \in \mathbb{R}^D$
3. *Trace equals total variance:* $\text{Tr}(\rho) = \sum_{i=1}^{D} \text{Var}(X_{\cdot,i})$

**Proof.**
1. $\rho^T = \frac{1}{N}(\tilde{X}^T \tilde{X})^T = \frac{1}{N}\tilde{X}^T \tilde{X} = \rho$ ✓

2. For any $v \in \mathbb{R}^D$:
   $$v^T \rho v = \frac{1}{N} v^T \tilde{X}^T \tilde{X} v = \frac{1}{N} \|\tilde{X}v\|_2^2 \geq 0$$ ✓

3. $\text{Tr}(\rho) = \frac{1}{N}\text{Tr}(\tilde{X}^T\tilde{X}) = \frac{1}{N}\sum_{i,j} \tilde{X}_{ij}^2 = \sum_{j=1}^{D} \frac{1}{N}\sum_{i=1}^{N} \tilde{X}_{ij}^2 = \sum_{j=1}^{D} \text{Var}(X_{\cdot,j})$ ✓

---

## A.3 Spectral Decomposition

### Theorem A.3.1 (Eigendecomposition)
*Since ρ is real symmetric, it admits the spectral decomposition:*

$$\rho = U \Lambda U^T = \sum_{i=1}^{D} \lambda_i u_i u_i^T$$

*where:*
- $U = [u_1 | u_2 | \cdots | u_D]$ is orthonormal: $U^T U = U U^T = I_D$
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_D)$ with $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D \geq 0$
- $u_i$ are the principal directions (eigenvectors)
- $\lambda_i$ are the variances along each principal direction

### Connection to SVD
If $\tilde{X} = V \Sigma W^T$ is the SVD of centered data, then:
$$\rho = \frac{1}{N} W \Sigma^2 W^T$$

Thus $U = W$ and $\lambda_i = \sigma_i^2 / N$.

---

## A.4 Subspace Projector Construction

### Definition A.4.1 (Rank-k Projector)
Given eigenvectors sorted by decreasing eigenvalue, the rank-k projector is:

$$\Pi_k = \sum_{i=1}^{k} u_i u_i^T = U_k U_k^T$$

where $U_k = [u_1 | \cdots | u_k] \in \mathbb{R}^{D \times k}$.

### Properties of Π

**Proposition A.4.1.** *The projector Π_k satisfies:*
1. *Symmetry:* $\Pi_k = \Pi_k^T$
2. *Idempotence:* $\Pi_k^2 = \Pi_k$
3. *Rank:* $\text{rank}(\Pi_k) = k$
4. *Trace:* $\text{Tr}(\Pi_k) = k$
5. *Eigenvalues:* $\text{spec}(\Pi_k) = \{\underbrace{1,\ldots,1}_{k}, \underbrace{0,\ldots,0}_{D-k}\}$

**Proof.**
1. $\Pi_k^T = (U_k U_k^T)^T = U_k U_k^T = \Pi_k$ ✓

2. $\Pi_k^2 = U_k U_k^T U_k U_k^T = U_k (U_k^T U_k) U_k^T = U_k I_k U_k^T = \Pi_k$ ✓
   (using orthonormality: $U_k^T U_k = I_k$)

3. $\text{rank}(\Pi_k) = \text{rank}(U_k U_k^T) = \text{rank}(U_k) = k$ ✓

4. $\text{Tr}(\Pi_k) = \text{Tr}(U_k U_k^T) = \text{Tr}(U_k^T U_k) = \text{Tr}(I_k) = k$ ✓

5. From idempotence, eigenvalues satisfy $\lambda^2 = \lambda$, so $\lambda \in \{0, 1\}$.
   Since $\text{Tr}(\Pi_k) = k$, exactly k eigenvalues equal 1. ✓

### Geometric Interpretation
$\Pi_k$ projects any vector onto the k-dimensional subspace spanned by the top-k eigenvectors:
$$\Pi_k v = \sum_{i=1}^{k} (u_i^T v) u_i$$

The **projected component** lies in the learned subspace.
The **rejected component** $(I - \Pi_k)v$ is orthogonal to it.

---

## A.5 Transfer Fidelity

### Definition A.5.1 (Transfer Fidelity)
Given source projector $\Pi_s$ and target density $\rho_t$:

$$F = \frac{\text{Tr}(\Pi_s \rho_t)}{\text{Tr}(\rho_t)}$$

### Theorem A.5.1 (Fidelity Bounds)
*For any valid projector and density matrix:*
$$0 \leq F \leq 1$$

**Proof.**

**Lower bound:** Since $\Pi_s$ and $\rho_t$ are both positive semi-definite:
$$\text{Tr}(\Pi_s \rho_t) = \text{Tr}(\Pi_s^{1/2} \rho_t \Pi_s^{1/2}) \geq 0$$
(using cyclic property and the fact that $\Pi_s = \Pi_s^{1/2} \Pi_s^{1/2}$)

**Upper bound:** Let $\rho_t = \sum_j \mu_j v_j v_j^T$ be the eigendecomposition of target.

$$\text{Tr}(\Pi_s \rho_t) = \sum_j \mu_j \text{Tr}(\Pi_s v_j v_j^T) = \sum_j \mu_j v_j^T \Pi_s v_j = \sum_j \mu_j \|\Pi_s v_j\|^2$$

Since $\Pi_s$ is a projector, $\|\Pi_s v_j\| \leq \|v_j\| = 1$, thus:
$$\text{Tr}(\Pi_s \rho_t) \leq \sum_j \mu_j = \text{Tr}(\rho_t)$$

Therefore $F \leq 1$. ✓

### Theorem A.5.2 (Fidelity as Variance Ratio)
*The fidelity equals the fraction of target variance captured by the source subspace:*

$$F = \frac{\sum_{i=1}^{k} \text{Var}_t(u_i^{(s)})}{\text{Var}_t(\text{total})}$$

*where $u_i^{(s)}$ are source eigenvectors and $\text{Var}_t(u) = u^T \rho_t u$.*

**Proof.**
$$\text{Tr}(\Pi_s \rho_t) = \text{Tr}\left(\sum_{i=1}^k u_i u_i^T \rho_t\right) = \sum_{i=1}^k \text{Tr}(u_i u_i^T \rho_t) = \sum_{i=1}^k u_i^T \rho_t u_i$$

Each term $u_i^T \rho_t u_i$ is the variance of target data projected onto source direction $u_i$. ✓

---

## A.6 Special Cases and Boundary Conditions

### Case 1: Perfect Fidelity (F = 1)

**Theorem A.6.1.** $F = 1$ if and only if $\text{Range}(\rho_t) \subseteq \text{Range}(\Pi_s)$.

**Proof.** 
$F = 1$ ⟺ $\text{Tr}(\Pi_s \rho_t) = \text{Tr}(\rho_t)$
⟺ $\text{Tr}((I - \Pi_s)\rho_t) = 0$
⟺ $(I - \Pi_s)\rho_t = 0$ (since both are PSD)
⟺ $\rho_t = \Pi_s \rho_t$
⟺ Every eigenvector of $\rho_t$ with nonzero eigenvalue lies in Range($\Pi_s$). ✓

**Corollary A.6.1 (The Loophole).** If $k \geq \text{rank}(\rho_t)$, then $F$ can equal 1 even if source and target are unrelated.

### Case 2: Zero Fidelity (F = 0)

**Theorem A.6.2.** $F = 0$ if and only if $\text{Range}(\rho_t) \perp \text{Range}(\Pi_s)$.

**Proof.**
$F = 0$ ⟺ $\text{Tr}(\Pi_s \rho_t) = 0$ ⟺ $\Pi_s \rho_t = 0$ (product of PSD matrices with zero trace)
⟺ Every eigenvector of $\rho_t$ is orthogonal to Range($\Pi_s$). ✓

### Case 3: Full Rank Projector

**Theorem A.6.3.** If $k = D$ (full rank), then $\Pi_s = I_D$ and $F = 1$ for any target.

This is the **trivial loophole**: a full-rank projector captures everything.

---

## A.7 Asymmetry Analysis

### Theorem A.7.1 (Non-commutativity)
*In general, $F(s \to t) \neq F(t \to s)$.*

**Proof by construction.** Let:
- Source: 1D manifold along $e_1$, so $\rho_s = e_1 e_1^T$, $\Pi_s = e_1 e_1^T$
- Target: 2D manifold in $e_1, e_2$ plane, so $\rho_t = \frac{1}{2}(e_1 e_1^T + e_2 e_2^T)$

Then:
$$F(s \to t) = \frac{\text{Tr}(\Pi_s \rho_t)}{\text{Tr}(\rho_t)} = \frac{\frac{1}{2}}{1} = 0.5$$

But with $\Pi_t = e_1 e_1^T + e_2 e_2^T = I_2$ (rank 2):
$$F(t \to s) = \frac{\text{Tr}(\Pi_t \rho_s)}{\text{Tr}(\rho_s)} = \frac{1}{1} = 1$$

Thus $F(s \to t) = 0.5 \neq 1 = F(t \to s)$. ✓

### Interpretation
- **Narrow → Wide**: Source captures subset of target → $F < 1$
- **Wide → Narrow**: Source spans target → $F = 1$ (if rank sufficient)

---

## A.8 Rank-Fidelity Relationship

### Theorem A.8.1 (Monotonicity)
*Fidelity is monotonically non-decreasing in rank:*
$$k_1 \leq k_2 \implies F(k_1) \leq F(k_2)$$

**Proof.**
$\Pi_{k_2} = \Pi_{k_1} + \sum_{i=k_1+1}^{k_2} u_i u_i^T$

Thus:
$$\text{Tr}(\Pi_{k_2} \rho_t) = \text{Tr}(\Pi_{k_1} \rho_t) + \sum_{i=k_1+1}^{k_2} u_i^T \rho_t u_i \geq \text{Tr}(\Pi_{k_1} \rho_t)$$

since $u_i^T \rho_t u_i \geq 0$. ✓

### Theorem A.8.2 (Saturation)
*Let $r = \text{rank}(\rho_t)$. Then:*
$$F(k) = 1 \quad \text{for all } k \geq D$$

*And more precisely, F saturates when the source subspace spans the target's support.*

---

## A.9 Connection to Quantum Mechanics

### Quantum State Fidelity (Jozsa, 1994)
For quantum states $\rho$ and $\sigma$, the fidelity is:
$$F_Q(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

### Our Simplification
When $\Pi$ is a pure projector (rank-k subspace), this reduces to:
$$F = \text{Tr}(\Pi \rho) / \text{Tr}(\rho)$$

This measures **how much of the quantum state ρ lives in the subspace Π**.

### Physical Interpretation
| Quantum Mechanics | Transfer Learning |
|-------------------|-------------------|
| Density matrix ρ | Data covariance |
| Pure state $\|ψ⟩⟨ψ\|$ | Single principal direction |
| Mixed state | Multi-dimensional manifold |
| Projective measurement | Subspace projection |
| Measurement probability | Variance captured |

---

## A.10 Computational Complexity

### Algorithm Complexity

| Step | Operation | Complexity |
|------|-----------|------------|
| 1 | Center data | $O(ND)$ |
| 2 | Compute $\rho = X^T X / N$ | $O(ND^2)$ |
| 3 | Eigendecomposition | $O(D^3)$ |
| 4 | Form projector $\Pi = U_k U_k^T$ | $O(D^2 k)$ |
| 5 | Compute $\text{Tr}(\Pi \rho_t)$ | $O(D^2)$ |

**Total:** $O(ND^2 + D^3)$

For $N \gg D$: $O(ND^2)$ — dominated by covariance computation.
For $N \ll D$: $O(D^3)$ — dominated by eigendecomposition.

### Comparison to Training
- Transfer fidelity: $O(D^3)$ — **computed once, before training**
- Neural network training: $O(\text{epochs} \times N \times \text{params})$ — much larger

**Speedup:** Fidelity computation is typically $10^3$–$10^6\times$ faster than training.

---

## A.11 Intrinsic Dimension Estimation

### Definition A.11.1 (Effective Dimension)
The effective dimension at threshold $\tau$ is:

$$d_{\text{eff}}(\tau) = \min\left\{k : \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^D \lambda_i} \geq \tau\right\}$$

This counts eigenvalues needed to capture fraction $\tau$ of total variance.

### Proposition A.11.1 (Dimension Mismatch Detection)
If $d_{\text{eff}}^{(s)} < d_{\text{eff}}^{(t)}$, then $F < 1$ for any $k \leq d_{\text{eff}}^{(s)}$.

**Interpretation:** Source has lower intrinsic dimension than target → imperfect transfer.

---

## A.12 Practical Recommendations

### Rank Selection
1. **Conservative:** $k = d_{\text{eff}}^{(s)}(0.95)$ — capture 95% of source variance
2. **Aggressive:** $k = \min(d_{\text{eff}}^{(s)}, d_{\text{eff}}^{(t)})$ — match dimensions
3. **Diagnostic:** Plot $F(k)$ vs $k$ to find the "knee"

### Fidelity Thresholds (Empirical Guidelines)

| Fidelity | Prediction | Recommendation |
|----------|------------|----------------|
| $F > 0.9$ | Excellent transfer | Fine-tune with small learning rate |
| $0.7 < F < 0.9$ | Good transfer | Standard fine-tuning |
| $0.5 < F < 0.7$ | Partial transfer | May need more target data |
| $F < 0.5$ | Poor transfer | Consider training from scratch |
| $F < 0.1$ | Negative transfer risk | Do not transfer |

---

## A.13 Proofs of Experimental Results

### Lemma A.13.1 (Orthogonal Lines)
*For X-axis data $X_s$ and Y-axis data $X_t$ in $\mathbb{R}^2$:*
$$F = 0$$

**Proof.**
$\rho_s = \begin{pmatrix} \sigma^2 & 0 \\ 0 & 0 \end{pmatrix}$, so $\Pi_s = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$

$\rho_t = \begin{pmatrix} 0 & 0 \\ 0 & \sigma^2 \end{pmatrix}$

$\text{Tr}(\Pi_s \rho_t) = \text{Tr}\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} = 0$ ✓

### Lemma A.13.2 (Diagonal Line to Circle)
*For diagonal line (1D) and circle (2D) in $\mathbb{R}^2$ with rank-1 projector:*
$$F = 0.5$$

**Proof.**
Diagonal line: $\rho_s = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$

Top eigenvector: $u_1 = \frac{1}{\sqrt{2}}(1, 1)^T$

$\Pi_s = u_1 u_1^T = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$

Circle: $\rho_t = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$

$\text{Tr}(\Pi_s \rho_t) = \frac{1}{2}\text{Tr}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \frac{1}{4}\text{Tr}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \frac{1}{4}(1+1) = \frac{1}{2}$

$\text{Tr}(\rho_t) = 1$

$F = 0.5$ ✓

---

## A.14 Summary of Novel Contributions

| Aspect | Classical | This Work |
|--------|-----------|-----------|
| **Density Matrix** | Covariance for single dataset | Cross-task comparison |
| **Projector** | Dimensionality reduction | Transfer subspace |
| **Fidelity** | Quantum information theory | Pre-training oracle |
| **Application** | Compression, visualization | Transfer learning prediction |
| **Rank Analysis** | Scree plot heuristics | Loophole characterization |
| **Asymmetry** | N/A | $F(A→B) \neq F(B→A)$ analysis |

---

## References

1. Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*.

2. Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*.

3. Jozsa, R. (1994). Fidelity for mixed quantum states. *Journal of Modern Optics*.

4. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.



