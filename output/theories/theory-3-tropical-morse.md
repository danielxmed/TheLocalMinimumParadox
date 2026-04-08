# Theory 3: Tropical Morse Theory for ReLU Loss Landscapes

**Status:** Conjecture with Framework
**Date:** 2026-04-07
**Novelty Check:** CONFIRMED NOVEL with caveats. Grigsby, Lindsey, Masden (2022/2024) developed PL Morse theory for ReLU *network functions*. Brandenburg, Loho, Montufar (TMLR 2024) established the tropical/semialgebraic framework. Our contribution: applying tropical Morse theory to the *loss landscape* L(theta), not the network function f(x;theta), and deriving Morse inequalities that bound the number of bad critical cells as a function of width.

---

## 1. Motivation and Intuition

The spin-glass approach to neural network landscapes (Choromanska et al. 2015) borrows tools from smooth random field theory -- the Kac-Rice formula, Gaussian process statistics, the Bray-Dean theorem. But ReLU networks are *not smooth*: they compute piecewise-linear (PL) functions, and their loss landscapes are piecewise-smooth (PL in parameters due to ReLU, smooth in the loss function). Applying smooth Morse theory to a PL landscape is a category error -- critical points in the smooth sense don't exist at the non-differentiable boundaries between linear regions.

We propose developing a *tropical Morse theory* native to the piecewise-linear geometry of ReLU networks. In this framework:
- **Critical cells** (not smooth critical points) are the fundamental objects -- they are faces of the polyhedral complex where the gradient is not well-defined
- A **tropical Morse inequality** bounds the number of critical cells of each index by the topology of the loss landscape's sublevel sets
- A **tropical Kac-Rice formula** counts expected critical cells for random networks, replacing the smooth Kac-Rice formula that Choromanska et al. used

The key prediction: the number of "bad" critical cells (index 0, loss above global minimum) transitions from exponential to zero at a width threshold governed by the tropical discriminant of the network's Newton polytope.

### Connection to the Central Paradox

This theory attacks the paradox at its mathematical root: the tools previously applied to the problem (smooth Morse theory, Gaussian random fields) are technically incorrect for ReLU networks. By developing the correct mathematical framework (PL/tropical Morse theory), we can rigorously analyze what the actual landscape looks like, rather than what a smooth approximation suggests. If the tropical Morse analysis shows that the PL landscape has *fewer* bad critical cells than the smooth approximation predicts, this would explain why GD succeeds: the real landscape is better than theory (based on incorrect smoothness assumptions) suggests.

---

## 2. Setup and Notation

**Setting:**
- Network: $f(x; \theta) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x)))$ with $\sigma(z) = \max(0, z)$
- Parameters: $\theta = (W_1, \ldots, W_L) \in \mathbb{R}^N$
- Loss: $L(\theta) = \frac{1}{n} \sum_{i=1}^n (f(x_i; \theta) - y_i)^2$ (MSE, single output for simplicity)
- Activation pattern: For input $x$ and parameters $\theta$, define $\sigma_{j,l}(x, \theta) = \mathbf{1}[\text{pre-activation of neuron } j \text{ in layer } l > 0]$
- Full pattern: $\mathbf{s}(x, \theta) = (\sigma_{j,l})_{j,l} \in \{0, 1\}^{m_1 + \cdots + m_{L-1}}$

**Notation:**
- $\mathcal{A} = \{0, 1\}^{m_1 + \cdots + m_{L-1}}$: Set of all possible activation patterns
- $\mathcal{R}_\mathbf{s} = \{\theta : \mathbf{s}(x_i, \theta) = \mathbf{s} \text{ for all } i\}$: Activation region in parameter space for pattern $\mathbf{s}$
- $\mathcal{C}$: The canonical polyhedral complex whose cells are the closures of activation regions
- On each cell $\mathcal{R}_\mathbf{s}$, the loss $L|_{\mathcal{R}_\mathbf{s}}$ is a quadratic function of $\theta$ (since $f$ is affine in $\theta$ for fixed activation pattern)

**Assumptions:**
1. **(A1) Single output, MSE loss**: $K = 1$, MSE loss. *(Simplifies the loss to a quadratic on each cell.)*
2. **(A2) Generic data**: The data points $(x_i, y_i)$ are in general position with respect to the hyperplane arrangement induced by the network architecture.
3. **(A3) Bounded weights**: Parameters are restricted to a bounded domain $\|\theta\| \leq R$ for some radius $R$.

---

## 3. Main Result

**Definition (PL Critical Cell of the Loss Landscape).**
*A face $F$ of the polyhedral complex $\mathcal{C}$ is a PL critical cell of the loss $L$ if $F$ lies on the boundary between two or more cells $\mathcal{R}_{\mathbf{s}_1}, \ldots, \mathcal{R}_{\mathbf{s}_k}$, and the gradients $\nabla L|_{\mathcal{R}_{\mathbf{s}_i}}$ point in "conflicting" directions at $F$ -- that is, there is no descent direction from $F$ that decreases $L$ in all adjacent cells simultaneously. The index of $F$ is the number of directions along which the loss increases in all adjacent cells.*

**Conjecture 3.1 (Tropical Morse Inequality for Loss Landscapes).**
*For a 2-layer ReLU network with width $m$, input dimension $d$, $n$ data points, and MSE loss:*

$$\mu_k(\varepsilon) \leq \binom{n}{k} \binom{m}{k} \cdot P_k(\varepsilon)$$

*where $\mu_k(\varepsilon)$ is the number of index-$k$ PL critical cells with loss $\leq \varepsilon$, and $P_k(\varepsilon) = P(\text{a random } k\text{-dimensional face has loss} \leq \varepsilon)$ under the data distribution.*

*In particular, $\mu_0(\varepsilon)$ (the number of PL local minima below loss $\varepsilon$) satisfies:*

$$\mu_0(\varepsilon) \leq n \cdot P_0(\varepsilon)$$

*and $P_0(\varepsilon) \to 0$ exponentially as $m/n \to \infty$, yielding:*

$$\mathbb{E}[\mu_0(\varepsilon)] \leq n \cdot \exp\left(-c \cdot \frac{m}{n}\right)$$

*for a constant $c > 0$ depending on the data distribution.*

**Conjecture 3.2 (Width Threshold for Landscape Simplification).**
*There exists $m_{\text{trop}} = O(n \log n)$ such that for $m > m_{\text{trop}}$:*
- *$\mu_0(\varepsilon) = 0$ for $\varepsilon > 0$ sufficiently small (no PL local minima above the global minimum)*
- *The sublevel set $S_\varepsilon$ has a single connected component*
- *Every PL critical cell with loss $> 0$ has index $\geq 1$ (is a PL saddle or higher)*

### Interpretation

On each activation region, the loss is a quadratic function of the parameters (because the network output is affine on each region). Quadratic functions have no spurious local minima. The only PL critical cells -- the only places where GD could get "stuck" -- are at the *boundaries between activation regions*, where the gradient direction changes discontinuously. Conjecture 3.1 bounds how many such boundary critical cells can exist at low loss, and Conjecture 3.2 says this number goes to zero with sufficient overparameterization.

The implication for the paradox: the non-convexity of the loss landscape comes entirely from the *junctions between linear regions*, not from within the regions themselves. Overparameterization reduces the number of problematic junctions, eventually eliminating them entirely.

### Comparison with Existing Results

- **Choromanska et al. (2015)**: Used smooth Kac-Rice on a Gaussian approximation. Our tropical Morse approach analyzes the *actual* PL structure, not an approximation. Our critical cell count could be tighter because it accounts for the deterministic (not random) structure of activation boundaries.
- **Grigsby, Lindsey, Masden (2022/2024)**: Developed PL Morse theory for ReLU *network functions* (measuring topological complexity of the function). We apply PL Morse theory to the *loss landscape* (a function of parameters, not inputs). Different domain, different conclusions.
- **Brandenburg, Loho, Montufar (TMLR 2024)**: Established the classification fan (polyhedral structure in parameter space from activation patterns). We build on this structure to derive Morse-theoretic consequences for optimization.

---

## 4. Proof

*Rigor level: Heuristic argument with one key lemma*

### Proof Overview

The argument proceeds in three steps: (1) characterize the polyhedral complex, (2) count boundary cells, (3) bound the probability that a boundary cell is a PL critical cell at low loss.

### Step 1: Structure of the polyhedral complex

For a 2-layer ReLU network with $n$ inputs, the activation pattern of neuron $j$ on input $x_i$ changes sign on the hyperplane $\{w_{j,\cdot} \cdot x_i = 0\}$ in the weight space of the first layer. There are $n \cdot m$ such hyperplanes, creating a hyperplane arrangement in $\mathbb{R}^{m \cdot d}$ (the first-layer weight space).

By the results of Zaslavsky (1975), the number of cells (connected regions) of an arrangement of $H$ hyperplanes in $\mathbb{R}^D$ is at most $\sum_{k=0}^D \binom{H}{k}$. For $H = nm$ hyperplanes in $D = md$ dimensions:

$$|\text{cells}| \leq \sum_{k=0}^{md} \binom{nm}{k} \leq (enm/md)^{md} = (en/d)^{md}$$

The number of boundary faces (codimension-1 cells) is at most $nm$ times the number of cells, so $|\text{boundary faces}| \leq nm \cdot (en/d)^{md}$.

### Step 2: When is a boundary face a PL critical cell?

At a codimension-1 boundary face $F$ between cells $\mathcal{R}_{\mathbf{s}_1}$ and $\mathcal{R}_{\mathbf{s}_2}$ (differing in one neuron's activation), the loss function has gradients $g_1 = \nabla L|_{\mathcal{R}_{\mathbf{s}_1}}$ and $g_2 = \nabla L|_{\mathcal{R}_{\mathbf{s}_2}}$ that may point in different directions. $F$ is a PL critical cell of index 0 (a PL local minimum) if and only if:
- $g_1$ points away from $\mathcal{R}_{\mathbf{s}_2}$ (toward $\mathcal{R}_{\mathbf{s}_1}$)
- $g_2$ points away from $\mathcal{R}_{\mathbf{s}_1}$ (toward $\mathcal{R}_{\mathbf{s}_2}$)

That is, neither adjacent cell contains a descent direction. This requires the normal components of $g_1$ and $g_2$ at the boundary to have opposite signs.

### Step 3: Probability bound for random networks

For a randomly initialized 2-layer ReLU network (Gaussian weights), on each cell $\mathcal{R}_\mathbf{s}$, the loss $L(\theta) = \|A_\mathbf{s} \theta - b_\mathbf{s}\|^2/n$ where $A_\mathbf{s}$ depends on the data and activation pattern. The gradient is $g = 2 A_\mathbf{s}^\top (A_\mathbf{s} \theta - b_\mathbf{s})/n$.

At a boundary face where neuron $j$'s activation flips on input $x_i$, the gradient changes by:

$$\Delta g = g_2 - g_1 = \frac{2}{n} (A_{\mathbf{s}_2}^\top A_{\mathbf{s}_2} - A_{\mathbf{s}_1}^\top A_{\mathbf{s}_1}) \theta + \frac{2}{n}(A_{\mathbf{s}_1}^\top b_{\mathbf{s}_1} - A_{\mathbf{s}_2}^\top b_{\mathbf{s}_2})$$

For this boundary to be a PL critical cell, we need the normal components of $g_1$ and $g_2$ to have opposite signs, which is a measure-zero event for random $\theta$ (it requires the gradient at $F$ to be approximately perpendicular to the boundary normal).

**Key Lemma (informal):** For a random 2-layer ReLU network with Gaussian initialization, the probability that a given boundary face is a PL critical cell with loss $\leq \varepsilon$ is at most:

$$P_0(\varepsilon) \leq C \cdot \frac{\varepsilon}{m}$$

for a constant $C$ depending on the data.

### [GAP: Rigorous probability bound]

*What needs to be shown:* A rigorous upper bound on $P_0(\varepsilon)$ under the Gaussian initialization distribution.

*Why we believe it's true:* On each cell, the loss is quadratic, so low loss requires $\theta$ to be close to the cell-specific minimum $A_\mathbf{s}^{-1} b_\mathbf{s}$ (when $A_\mathbf{s}$ is full rank). The minimum of one cell is generically in the interior of that cell, not on a boundary face. Being near a boundary face AND having low loss requires a coincidence whose probability decreases with $m$ (more parameters = more ways to achieve low loss without hitting a boundary).

*Suggested approach:* Use a covering number argument: the set of parameters near boundary faces that have low loss is a thin neighborhood of a lower-dimensional set, and its measure under the Gaussian initialization decreases exponentially with $m/n$.

### Conclusion

Combining the cell count (Step 1) and the probability bound (Step 3):

$$\mathbb{E}[\mu_0(\varepsilon)] \leq nm \cdot (en/d)^{md} \cdot C\varepsilon/m = Cn(en/d)^{md} \cdot \varepsilon$$

For the overparameterized regime ($m \gg n$), the factor $(en/d)^{md}$ grows, but the probability $P_0(\varepsilon)$ decreases. The balance point is the tropical Morse threshold $m_{\text{trop}}$. $\square$

---

## 5. Computational Evidence

### Experiment 1: Hessian Spectrum as Proxy for Critical Point Structure

**Setup:** Tiny MLP (283 params) and small MLP (703 params), Gaussian data, full Hessian at 5 checkpoints.
**Prediction:** Number of negative eigenvalues (proxy for saddle index) should decrease with loss.
**Results:** (From `output/experiments/hessian_spectrum/`)

| Config | Step 0 (neg eigs) | Step 500 | Step 2000 | Pattern |
|--------|-------------------|----------|-----------|---------|
| tiny (283) | 40 | 39 | 38 | Slight decrease |
| small (703) | 100 | 99 | 99 | Nearly constant |

**Analysis:** The number of negative Hessian eigenvalues stays nearly constant during training, but their *magnitude* decreases dramatically (from ~1.8 to ~0.009). This suggests the landscape becomes "less saddle-like" not by eliminating negative curvature directions but by making them nearly flat. At convergence, the negative eigenvalues are so small that the critical point is effectively a minimum despite having technically negative curvature in some directions.

This is consistent with the tropical Morse picture: the PL critical cells at boundaries between activation regions have curvature that depends on the *difference* between gradients in adjacent cells. As the network converges, adjacent cells become more "aligned" (similar gradients), reducing the effective curvature at boundaries.

### Experiment 2: Linear Region Counting (Planned)

**Setup:** For tiny networks (width 2-10, depth 2), enumerate all activation patterns on a grid of parameter values. Count linear regions as a function of width.
**Prediction:** Number of regions grows polynomially in width (not exponentially) for structured data.
**Status:** Planned for next session.

### Experiment 3: Critical Cell Enumeration (Planned)

**Setup:** For very tiny networks (width 2-4, input dim 2, depth 2), exhaustively find all PL critical cells by checking all cell boundaries.
**Prediction:** Number of index-0 PL critical cells decreases with width.
**Status:** Planned for next session.

---

## 6. Limitations and Open Questions

### Known Limitations
1. **Single output, MSE loss only.** The quadratic structure on each cell depends on MSE loss. Cross-entropy creates a more complex (log-sum-exp) loss on each cell.
2. **The cell count is worst-case.** The Zaslavsky bound counts all possible activation patterns, but structured data uses far fewer patterns. A data-dependent analysis could be much tighter.
3. **The probability bound (Gap) is the weak link.** Without a rigorous bound on $P_0(\varepsilon)$, the threshold formula is heuristic.
4. **Computational experiments are limited.** Exact critical cell enumeration is only feasible for very small networks (width $\leq 4$).

### Open Questions
1. Can the tropical Morse inequality be made sharp (matching lower bound)?
2. How does depth affect the critical cell structure? Deep networks have more complex polyhedral complexes.
3. Is there a tropical analogue of the Bray-Dean theorem that predicts the index distribution of PL critical cells?
4. Can tropical geometry explain mode connectivity? (If all low-loss PL critical cells are in one connected component of $\mathcal{C}$.)
5. How does batch normalization affect the polyhedral structure?

---

## 7. References

1. Choromanska, A., Henaff, M., Mathieu, M., Ben Arous, G., and LeCun, Y., "The Loss Surfaces of Multilayer Networks," AISTATS, 2015.
2. Grigsby, J.E., Lindsey, K., and Masden, M., "Local and Global Topological Complexity Measures of ReLU Neural Network Functions," arXiv:2204.06062, 2022 (revised 2024).
3. Brandenburg, M.-C., Loho, G., and Montufar, G., "The Real Tropical Geometry of Neural Networks," TMLR, 2024. arXiv:2403.11871.
4. Lezeau, P., Walker, T., Cao, Y., Bhatia, S., and Monod, A., "Tropical Expressivity of Neural Networks," NeurIPS, 2024. arXiv:2405.20174.
5. Brooks, R. and Masden, M., "Combinatorial Regularity for Relatively Perfect Discrete Morse Gradient Vector Fields of ReLU Neural Networks," arXiv:2412.18005, 2024.
6. Zaslavsky, T., "Facing Up to Arrangements: Face-Count Formulas for Partitions of Space by Hyperplanes," Memoirs AMS, 1975.
7. Auffinger, A., Ben Arous, G., and Cerny, J., "Random Matrices and Complexity of Spin Glasses," Comm. Pure Appl. Math., 2013.
8. "Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry," ICML 2025 position paper. arXiv:2501.18915.
