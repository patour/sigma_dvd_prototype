---
title: "Blame Assignment for Dynamic IR-Drop in LTI RC Power Distribution Networks"
author: "PDN Analysis Team"
date: "March 2026"
abstract: |
  This report derives the method for decomposing dynamic IR-drop at a victim
  node into exact per-instance aggressor contributions under a linear
  time-invariant RC model. The primary derivation uses the impulse response
  (Green's function) of the RC network: by superposition, the victim's droop
  is a convolution of each source's current waveform with the network's
  sensitivity kernel. A single backward-in-time evaluation of this kernel
  yields all aggressor contributions simultaneously. The Lagrangian adjoint
  formulation, which generalizes to nonlinear and time-varying systems, is
  presented in an appendix along with the spectral interpretation.
---

# 1. Introduction and Problem Statement

## 1.1 The Blame Question

Consider a power distribution network (PDN) modeled as an RC mesh with $N$ nodes, $P$ voltage-source pads held at $V_{dd}$, and $M$ time-varying current sources $I_1(t), \ldots, I_M(t)$ representing switching loads. At observation time $T$, the voltage at a victim node $v$ has drooped below $V_{dd}$:

$$\Delta V_v(T) \;=\; V_{dd} - V_v(T) \;>\; 0 \qquad (\text{mV}).$$

The **blame assignment problem** asks: *how much of $\Delta V_v(T)$ is attributable to each aggressor instance?*  That is, find scalars $\{c_j\}_{j=1}^{M}$ such that

$$\Delta V_v(T) \;=\; \sum_{j=1}^{M} c_j,$$

where $c_j$ quantifies the contribution of current source $j$ to the victim's droop. Once computed, the top-$K$ aggressors by $|c_j|$ identify the dominant switching noise contributors for design remediation.

## 1.2 Linearity Enables Exact Decomposition

The key enabler is **linearity**. Under the LTI RC model, the nodal voltage $V_v(T)$ is a linear functional of the current excitations. By superposition, the response to the combined excitation $I(t) = \sum_j I_j(t)$ equals the sum of individual responses:

$$V_v(T) \;=\; V_v^{(0)}(T) + \sum_{j=1}^{M} V_v^{(j)}(T),$$

where $V_v^{(j)}$ is the voltage response to source $j$ alone and $V_v^{(0)}$ is the homogeneous response from the initial condition and pad voltages.

Rather than running $M$ separate forward simulations (prohibitively expensive for $M > 10^5$ sources in production PDNs), the **impulse response method** computes all $M$ contributions with a single backward-in-time evaluation of the network's sensitivity kernel.


# 2. System Formulation

## 2.1 Nodal Equations

The complete set of nodal equations for the RC network is:

$$C \frac{dV}{dt} + G\, V = -\, I(t),$$

where $G$ is the conductance matrix (mS), $C$ is the capacitance matrix (fF), $V$ is the node voltage vector, and $I(t)$ is the vector of injected load currents (mA). The negative sign follows the convention that positive current denotes a *sink* drawing current from the grid.

## 2.2 Reduced System

Pad nodes are held at constant voltage $V_p = V_{dd}$ (Dirichlet boundary conditions). Eliminating the pad rows and columns via standard Schur complement reduction yields the **reduced system** on the $N_u$ unknown (non-pad) nodes:

$$\boxed{C_{uu}\, \dot{V}_u + G_{uu}\, V_u = -I_u(t), \qquad V_u(0) = V_u^{(0)},}$$

where $I_u(t)$ is the load current vector on the unknown nodes (positive = sink). The constant pad contribution through resistive coupling is absorbed into the equilibrium voltage and does not affect the time-varying blame decomposition. Both $G_{uu}$ and $C_{uu}$ are symmetric positive semi-definite for a passive RC network; $G_{uu}$ is SPD when every unknown node has a resistive path to at least one pad. In the PDN, all tile-level capacitors are grounded, so $C_{uu}$ is diagonal.

## 2.3 Unit System

The PDN uses internally consistent units: resistance in k$\Omega$, conductance in mS, capacitance in fF, current in mA, voltage in V. This gives $RC$ time constants in ns (k$\Omega \times$ fF = ns). For time discretization, the time step is scaled to picoseconds: $\Delta t_s = \Delta t \times 10^{12}$, ensuring dimensional consistency: $G\text{[mS]} + C\text{[fF]} / \Delta t_s\text{[ps]} = \text{mS}$.


# 3. Blame via Impulse Response

This section derives the blame decomposition using only linearity and the impulse response of the RC network. No variational calculus or Lagrange multipliers are needed.

## 3.1 Solution of the Reduced LTI System

The reduced ODE from Section 2.2 is a linear, time-invariant system. Its solution is given by the standard variation-of-parameters formula:

$$V_u(T) = \Phi(T)\, V_u^{(0)} - \int_0^T \mathcal{G}(T - \tau)\, I_u(\tau)\, d\tau,$$

where:

- $\Phi(s) = \exp\!\bigl(-C_{uu}^{-1} G_{uu}\, s\bigr)$ is the **state transition matrix**, describing how the homogeneous (unforced) system evolves.
- $\mathcal{G}(s) = \Phi(s)\, C_{uu}^{-1}$ is the **impulse response matrix** (Green's function) of the RC network.

The entry $\mathcal{G}_{ij}(s)$ has a direct physical meaning: it is the voltage at node $i$ measured $s$ seconds after a unit impulse of current ($1\;\text{mA} \cdot \text{ps}$) is injected at node $j$. The impulse response satisfies:

$$C_{uu}\, \frac{d\mathcal{G}}{ds} + G_{uu}\, \mathcal{G}(s) = 0, \qquad s > 0, \qquad \mathcal{G}(0^+) = C_{uu}^{-1}.$$

Since $G_{uu}$ and $C_{uu}$ are symmetric, so is $\mathcal{G}(s)$: the voltage at node $i$ from an impulse at $j$ equals the voltage at $j$ from an impulse at $i$. This is the **reciprocity** property of passive RC networks.

## 3.2 Blame as Convolution

The current source vector decomposes as $I_u(\tau) = \sum_{j=1}^{M} e_{n_j}\, I_j(\tau)$, where $e_{n_j}$ is the unit vector at the grid node $n_j$ where source $j$ injects current. Substituting into the solution and projecting onto the victim node $v$:

$$V_v(T) = \underbrace{e_v^\top \Phi(T)\, V_u^{(0)}}_{V_v^{\text{hom}}(T)\;(\text{IC contribution})} \;-\; \sum_{j=1}^{M} \int_0^T \mathcal{G}_{v, n_j}(T-\tau)\, I_j(\tau)\, d\tau.$$

The IR-drop contributed by source $j$ is therefore:

$$\boxed{c_j = \int_0^T \mathcal{G}_{v, n_j}(T - \tau)\; I_j(\tau)\; d\tau.}$$

This is a **convolution** of the impulse response with the source's current waveform. The result is **exact** — it follows from nothing more than linearity and the definition of the impulse response.

**Conservation.** Summing over all sources:

$$\sum_{j=1}^{M} c_j = V_v^{\text{hom}}(T) - V_v(T) = \Delta V_v(T) - \Delta V_v^{\text{hom}}(T).$$

For zero initial conditions ($V_u^{(0)} = V_{dd} \cdot \mathbf{1}$), the homogeneous response maintains $V_{dd}$ and the sum equals the total IR-drop $\Delta V_v(T)$.

## 3.3 Efficient Evaluation: The Sensitivity Kernel

Computing the blame integral for all $M$ sources naively would require evaluating $M$ different entries $\mathcal{G}_{v, n_j}(s)$ of the impulse response matrix. But there is a crucial simplification: **all $M$ blames share the same column of $\mathcal{G}$.**

Define the **sensitivity kernel**:

$$h(s) = \mathcal{G}(s)\, e_v,$$

the $v$-th column of the impulse response matrix. Then:

$$c_j = \int_0^T h_{n_j}(T - \tau)\; I_j(\tau)\; d\tau,$$

where $h_{n_j}(s)$ is the $n_j$-th component of $h(s)$. All $M$ blame integrals are computed from the **single vector-valued function** $h(s)$.

The sensitivity kernel satisfies the homogeneous RC ODE with a jump initial condition:

$$C_{uu}\, \frac{dh}{ds} + G_{uu}\, h(s) = 0, \qquad s > 0, \qquad C_{uu}\, h(0^+) = e_v.$$

Equivalently, extending $h(s) = 0$ for $s < 0$, the kernel satisfies the distributional ODE with a **Dirac source**:

$$\boxed{C_{uu}\, \frac{dh}{ds} + G_{uu}\, h(s) = e_v\, \delta(s).}$$

The two forms are equivalent: integrating across $s = 0$ gives $C_{uu}[h(0^+) - h(0^-)] = e_v$, and since $h(0^-) = 0$ this recovers the jump condition $C_{uu}\, h(0^+) = e_v$. The Dirac form unifies the initial impulse with the dynamics into a single equation and maps naturally to the discrete formulation (Section 4.2).

Physically, $h(s)$ is a standard RC decay from an impulse concentrated at the victim node. The **backward sweep** of the blame assignment algorithm is precisely the numerical evaluation of $h(s)$ at discrete lag values $s = 0, \Delta t, 2\Delta t, \ldots$

```
    Impulse at victim v                    Sensitivity kernel h(s)
           |                               decays outward from v
    t=T    |  s=0: h(0) = C_uu^{-1} e_v   (concentrated at v)
           |
    t=T-dt |  s=dt: h(dt)                  (spread to neighbors)
           |
    t=T-2dt|  s=2dt: h(2dt)               (further spread)
           |
           v
    t=0    |  s=T: h(T)                    (decayed to near-zero)

    At each lag s = T - t_k:
      c_j += h_{n_j}(s) * I_j(t_k)

    Figure 1: The sensitivity kernel h(s) starts as a unit impulse at
    the victim and decays outward through the RC network. At each time
    step, the kernel value at each source's node is multiplied by the
    source's current to accumulate blame.
```


# 4. Discrete-Time Formulation (Backward Euler)

## 4.1 Forward Discretization

Discretize the reduced ODE (Section 2.2) using Backward Euler at uniform time step $\Delta t$ (seconds). With $\Delta t_s = \Delta t \times 10^{12}$ (ps), the discrete forward system is:

$$A\, V_u^{n+1} = -I_u(t_{n+1}) + B\, V_u^n,$$

where:

$$A = G_{uu} + \frac{C_{uu}}{\Delta t_s}, \qquad B = \frac{C_{uu}}{\Delta t_s}.$$

The matrix $A$ is SPD, admitting a sparse Cholesky factorization.

## 4.2 Discrete Impulse Response

Apply Backward Euler to the Dirac form of the sensitivity kernel ODE (Section 3.3), $C_{uu}\, \dot{h} + G_{uu}\, h = e_v\, \delta(s)$. Denote the sampled continuous kernel at lag $s = l \cdot \Delta t$ by $h_l$. With $h_{-1} = 0$:

$$\frac{C_{uu}}{\Delta t_s}(h_l - h_{l-1}) + G_{uu}\, h_l = \frac{e_v}{\Delta t_s}\, \delta_{l,0},$$

where $\delta_{l,0}$ is the Kronecker delta and $1/\Delta t_s$ is the Dirac-to-Kronecker conversion. Rearranging:

$$A\, h_l = B\, h_{l-1} + \frac{e_v}{\Delta t_s}\, \delta_{l,0}.$$

The continuous blame is an integral ($\int h\, I\, d\tau$), but the discrete blame is a sum ($\sum \lambda\, I$). Define the **discrete kernel** $\lambda_l = \Delta t_s \cdot h_l$ to absorb the integration measure. Multiplying through by $\Delta t_s$:

$$\boxed{A\, \lambda_l = B\, \lambda_{l-1} + e_v\, \delta_{l,0}.}$$

At $l = 0$: $A\, \lambda_0 = e_v$, so $\lambda_0 = A^{-1}\, e_v$. For $l \geq 1$: $A\, \lambda_l = B\, \lambda_{l-1}$.

Re-indexing to backward time notation (as used in the implementation), with $\lambda_k \equiv \lambda_{N-1-k}$ where $k$ now indexes forward time:

$$\boxed{\lambda_{N-1} = A^{-1}\, e_v, \qquad A\, \lambda_k = B\, \lambda_{k+1} \quad (k = N-2, \ldots, 0).}$$

This is the **backward sweep**: the same $A$ factorization is reused at every step.

**Verification from the discrete forward equations.** The same result follows independently from differentiating the forward system $A\, V^{n+1} = -I_u^{n+1} + B\, V^n$. The sensitivity of the victim's IR-drop to the forcing at step $k$ is $e_v^\top (A^{-1} B)^{N-1-k} A^{-1}$. At the final step this is $e_v^\top A^{-1}$, so $\lambda_{N-1} = A^{-1} e_v$ captures this sensitivity by symmetry of $A$, confirming the terminal condition from a purely discrete perspective.

## 4.3 Discrete Blame Formula

The discrete blame is:

$$\boxed{c_j = \sum_{k=0}^{N-1} \lambda_k[n_j] \;\cdot\; I_j(t_k).}$$

**No $\Delta t$ factor.** The $\Delta t_s$ is already absorbed into $\lambda$: since $\lambda_l = \Delta t_s \cdot h_l$, the discrete sum $\sum_k \lambda_k \cdot I_j(t_k)$ approximates the continuous integral $\int h_{n_j}(T-\tau)\, I_j(\tau)\, d\tau$ directly.

**Side-by-side comparison:**

| Continuous | Discrete (BE) |
|---|---|
| $c_j = \int_0^T h_{n_j}(T-\tau)\, I_j(\tau)\, d\tau$ | $c_j = \sum_{k=0}^{N-1} \lambda_k[n_j] \cdot I_j(t_k)$ |
| $C_{uu}\, \dot{h} + G_{uu}\, h = e_v\, \delta(s)$ | $A\, \lambda_l = B\, \lambda_{l-1} + e_v\, \delta_{l,0}$ |
| $h(s)$ has units k$\Omega$/ps | $\lambda_k$ has units k$\Omega$ (= V/mA), absorbs $\Delta t_s$ |

## 4.4 Unit Analysis

| Quantity | Symbol | Units | Derivation |
|----------|--------|-------|------------|
| System matrix | $A$ | mS | $G_{uu}\text{[mS]} + C_{uu}\text{[fF]} / \Delta t_s\text{[ps]}$ |
| History matrix | $B$ | mS | $C_{uu}\text{[fF]} / \Delta t_s\text{[ps]}$ |
| Sensitivity kernel (discrete) | $\lambda_k$ | k$\Omega$ (= V/mA) | From $A^{-1}$ where $[A] =$ mS |
| Source current | $I_j(t_k)$ | mA | PDN convention |
| Single-step contribution | $\lambda_k[n_j] \cdot I_j$ | V | k$\Omega \times$ mA |
| Blame (accumulated) | $c_j$ | V (reported as mV) | $\sum_k$ of above, $\times 1000$ |

```
    ALGORITHM: Backward Sweep (Discrete Impulse Response)
    ======================================================

    Input:  victim index v, time array [t_0, ..., t_{N-1}],
            factored A (LU or Cholesky), matrix B,
            current sources {I_j(t)}

    Output: blame {c_j} for each source j

    1. FACTOR A (or reuse from forward transient solve)
       lu_A = cholesky(A)             // O(nnz^{3/2}), one-time

    2. TERMINAL CONDITION (impulse response at lag 0)
       e_v = zeros(n_unknown)
       e_v[v] = 1.0
       lambda_next = lu_A.solve(e_v)  // O(nnz)

    3. INITIALIZE accumulators
       c[j] = 0   for all j

    4. BACKWARD SWEEP: k = N-1, N-2, ..., 0

       if k == N-1:
           lambda_current = lambda_next
       else:
           rhs = B * lambda_next       // sparse matvec, O(nnz)
           lambda_current = lu_A.solve(rhs)  // O(nnz)

       // Evaluate currents and accumulate (O(M))
       for each source j:
           c[j] += lambda_current[n_j] * I_j(t_k)

       swap(lambda_current, lambda_next)

    5. CONVERT to mV:  c[j] *= 1000

    Memory: O(n + M)  -- only 2 lambda vectors + M accumulators
    Time:   O(N * (nnz + M))  -- N back-solves + N * M accumulations

    Figure 2: Algorithmic flowchart. The LU factorization from the
    forward transient solve is reused (symmetric A). Only two lambda
    vectors are kept in memory at any time (O(n) swap trick).
```


# 5. Initial Conditions

## 5.1 Zero IC Mode

In **zero IC mode**, the initial condition is $V_u(0) = V_{dd} \cdot \mathbf{1}$ (zero IR-drop at $t = 0$). The blame attributes the **total IR-drop** at the victim:

$$c_j^{\text{zero}} = \sum_{k=0}^{N-1} \lambda_k[n_j] \;\cdot\; I_j(t_k).$$

The sum $\sum_j c_j^{\text{zero}} = \Delta V_v(T)$ accounts for the entire droop from the pristine $V_{dd}$ state.

**When to use:** When the engineer wants total accountability of all switching activity within the observation window. Appropriate when the memory window encompasses the full simulation from startup.

## 5.2 DC IC Mode

In **DC IC mode**, the initial condition is the DC operating point $V_u(0) = V_{DC}$, computed by solving the static system:

$$G_{uu}\, V_{DC} = -I_{DC} \qquad (\text{pad contribution implicit}),$$

where $I_{DC}$ is the static (DC) component of the load currents. The blame then attributes only the **incremental IR-drop** above the DC baseline:

$$c_j^{\text{inc}} = \sum_{k=0}^{N-1} \lambda_k[n_j] \;\cdot\; \bigl(I_j(t_k) - I_{j,DC}\bigr).$$

The DC baseline IR-drop is $\Delta V_v^{DC} = V_{dd} - V_{DC,v}$ (mV).

## 5.3 Relationship Between Modes

Define the **static contribution** of source $j$:

$$c_j^{\text{static}} = \sum_{k=0}^{N-1} \lambda_k[n_j] \;\cdot\; I_{j,DC}.$$

Then:

$$\boxed{c_j^{\text{zero}} = c_j^{\text{inc}} + c_j^{\text{static}}.}$$

This identity holds exactly by linearity. The total blame decomposes into an incremental part (switching above DC level) and a static part (constant background current). DC IC mode isolates switching noise from the static voltage map.


# 6. Per-Instance Blame Aggregation and Top-K Ranking

## 6.1 Aggregation Hierarchy

The blame formula (Section 4.3) computes blame at the level of individual current sources $c_j$. In practice, multiple sources may be associated with a single design instance. The aggregation proceeds:

1. **Per-source:** $c_j$ for each current source $j$.
2. **Per-node:** $c_{\text{node}} = \sum_{j: \text{node}(j) = \text{node}} c_j$ summing all sources at the same grid node.
3. **Per-instance:** $c_{\text{inst}} = \sum_{j \in \text{inst}} c_j$ summing all sources belonging to the same physical instance (cell/macro).

The per-instance level is the most actionable for design: it tells the engineer which block to resize, move, or add decap near.

## 6.2 Self-Contribution

The victim's own current sources (if any) are separated:

$$c_{\text{self}} = \sum_{j \in \text{sources}(v)} c_j.$$

This quantifies how much of the droop is self-inflicted versus caused by remote aggressors.

## 6.3 Top-K Ranking

After aggregation, sort instances by $|c_{\text{inst}}|$ descending and report the top $K$. Each entry includes:

- Instance identifier
- Contribution in mV and as a percentage of total IR-drop
- List of constituent source names
- (Optional) Current waveform over the memory window

## 6.4 Attribution Efficiency

Define:

$$\eta = \frac{\sum_{j=1}^{M} c_j}{\Delta V_v(T)}.$$

For a linear system with the memory window encompassing all relevant history, $\eta = 1.0$ exactly. Deviations from unity indicate truncation or numerical artifacts (see Section 8).


# 7. Algorithmic Complexity and Implementation Notes

## 7.1 Cost Breakdown

| Phase | Cost | Notes |
|-------|------|-------|
| LU factorization of $A$ | $O(n_{\text{nnz}}^{3/2})$ | One-time; shared with forward transient |
| Terminal condition solve | $O(n_{\text{nnz}})$ | Single back-substitution |
| Each backward step | $O(n_{\text{nnz}})$ | SpMV ($B \cdot \lambda$) + back-sub |
| Current evaluation per step | $O(M)$ | Vectorized PWL evaluation |
| Contribution accumulation per step | $O(M)$ | Dot product per source |
| **Total backward sweep** | **$O(L \cdot (n_{\text{nnz}} + M))$** | $L$ = number of time steps |

For typical PDN sizes ($n \sim 10^6$ unknowns, $M \sim 10^5$ sources, $L \sim 100$ steps), the backward sweep is dominated by the $L$ sparse triangular solves.

## 7.2 Memory

The backward sweep requires only **two $\lambda$ vectors** ($2 \times n$ doubles) via the swap trick, plus $M$ scalar accumulators. Total: $O(n + M)$. The LU factors (from the forward solve) are the dominant memory consumer.

## 7.3 LU Reuse

The system matrix $A = G_{uu} + C_{uu}/\Delta t_s$ is identical for the forward transient and backward sweep (since $A^\top = A$ for symmetric $G_{uu}$, $C_{uu}$). The implementation shares the cached factorization, avoiding a redundant $O(n_{\text{nnz}}^{3/2})$ factorization. This typically saves 200--500 MB of memory.

## 7.4 Vectorized Current Evaluation

Current sources are stored in a `VectorizedCurrentSources` structure that supports $O(M)$ batch evaluation of all PWL/pulse waveforms at a given time $t$. The per-source evaluation returns individual source currents needed for the accumulation inner loop.

## 7.5 Spatial Filtering

When only nearby aggressors are of interest (e.g., within a spatial window around the victim), the candidate source set can be pruned *a priori*. This reduces the inner-loop cost from $O(M)$ to $O(M_{\text{window}})$ per step. The sensitivity kernel $\lambda$ is still computed over the full grid.


# 8. Practical Considerations and Sensitivity Analysis

## 8.1 Attribution Efficiency Deviations

The attribution efficiency $\eta$ (Section 6.4) should be exactly 1.0 for a linear system when the memory window and source set are complete. In practice, $\eta$ deviates due to:

1. **Memory window truncation**: If the backward sweep starts at $t_0 > 0$ rather than $t = 0$, contributions from the interval $[0, t_0)$ are missed.

2. **Spatial window exclusion**: Sources outside the spatial window are not counted, even though they contribute through the grid's far-field coupling modes (see Appendix B).

3. **Numerical precision**: Accumulation over many small terms introduces floating-point rounding errors, typically at the $10^{-10}$ level.

## 8.2 Sensitivity to Memory Window Length

The sensitivity kernel decays exponentially with lag. The spectral radius of the one-step propagator $A^{-1} B$ determines the decay rate:

$$\rho(A^{-1} B) = \max_k \frac{1/\Delta t_s}{\mu_k + 1/\Delta t_s} = \frac{\tau_{\max}}{\tau_{\max} + \Delta t_s},$$

where $\tau_{\max} = 1/\mu_{\min}$ is the slowest RC time constant (see Appendix B for the spectral decomposition). After $L$ steps, the residual kernel magnitude is bounded by:

$$\|\lambda_{N-1-L}\| \leq \|\lambda_{N-1}\| \cdot \rho^L.$$

To ensure truncation error below a fraction $\epsilon$:

$$L > \frac{\ln(1/\epsilon)}{\ln(1/\rho)} \approx \frac{\tau_{\max}}{\Delta t_s} \cdot \ln(1/\epsilon).$$

**Rule of thumb:** For $\epsilon = 1\%$ ($\eta > 0.99$), set $L \cdot \Delta t > 5 \cdot \tau_{\max}$. For $\epsilon = 0.1\%$, use $7 \cdot \tau_{\max}$.

```
    Attribution efficiency eta vs. memory window length
    (normalized to tau_max)

    eta
    1.00 |                              _______________
         |                         ____/
    0.95 |                    ____/
         |               ____/
    0.90 |          ____/
         |     ____/
    0.85 | ___/
         |/
    0.80 +----------+----------+----------+----------+-->
         0       2*tau      4*tau      6*tau      8*tau
                         Memory window length

    Figure 3: Attribution efficiency as a function of memory window
    length. The curve approaches 1.0 exponentially; 5*tau_max gives
    >99% efficiency for most PDN topologies.
```

## 8.3 Sensitivity to Time Step $\Delta t$

**Stiff regime** ($\tau_{\max} \ll \Delta t$): The kernel decays to near-zero in a single step ($\rho \approx 0$). Only the terminal step contributes:

$$c_j \approx \lambda_{N-1}[n_j] \cdot I_j(T) = (A^{-1} e_v)[n_j] \cdot I_j(T) \approx (G_{uu}^{-1} e_v)[n_j] \cdot I_j(T).$$

This converges to the **static sensitivity** $G_{uu}^{-1}$. In the stiff limit, a single matrix solve suffices.

**Non-stiff regime** ($\tau_{\max} \gg \Delta t$): Memory extends over many steps. Multiple past time steps contribute significantly. Reducing $\Delta t$ improves temporal resolution at the cost of more backward steps.

**Crossover:** At $\Delta t \approx \tau_{\max}$, the ranking transitions from being dominated by recent/local activity to including historical/distant contributions.

## 8.4 Sensitivity to Spatial Window

Restricting the spatial window (Section 7.5) trades completeness for speed. The "leakage" — blame attributed to sources outside the window — is carried by the low-frequency eigenmodes of the RC network (see Appendix B for the spectral interpretation):

- **Near-field modes** (localized, fast decay): fully captured by a moderate window.
- **Far-field modes** (global, slow decay): extend across the grid. Sources outside the window contribute through these modes, and their blame is *missed*.

For PDNs with strong far-field coupling (sparse pad placement, uniform current loads), a large spatial window or no window is needed.

## 8.5 Sensitivity to Instance Grouping

The top-K ranking can change depending on the aggregation level:

- **Per-node ranking**: Fine-grained. Two sources on different nodes of the same instance appear as separate entries.
- **Per-instance ranking**: Coarse-grained. An instance with many small-contribution sources may rank higher than a single large-contribution source.

**Recommendation:** Report both per-instance blame and the number of constituent sources. An instance with 50 sources each contributing 0.1 mV (total 5 mV) has a qualitatively different character than one source contributing 5 mV.


# 9. Validation

The blame assignment can be validated through four independent checks:

## 9.1 Aggressor-Only Simulation

For each top aggressor $j$, run a forward transient with *only* source $j$ active (all others zeroed), starting from $V_u(0) = V_{dd}$. The IR-drop at the victim at time $T$ should match the blame:

$$\Delta V_v^{(j)}(T) \approx c_j.$$

The match is exact when the memory window encompasses the full simulation interval.

## 9.2 Superposition Check

Run the full forward transient and verify:

$$\sum_{j=1}^{M} c_j \approx \Delta V_v(T), \qquad \eta \approx 1.0.$$

## 9.3 Stiff-System Convergence

For $\tau_{\max} \ll \Delta t$, verify that the dynamic blame matches the static sensitivity $G_{uu}^{-1} e_v \cdot I_j(T)$.

## 9.4 DC Mode Decomposition

Verify the identity from Section 5.3:

$$c_j^{\text{inc}} + c_j^{\text{static}} = c_j^{\text{zero}} \qquad \forall\, j.$$

This holds to machine precision by linearity.


# 10. Conclusion

## Summary

This report derived the impulse-response-based method for exact per-instance decomposition of dynamic IR-drop in LTI RC power distribution networks. The key results are:

1. **Exact decomposition**: The victim's IR-drop is a sum of convolutions of the impulse response with each source's current waveform. Linearity guarantees exactness.

2. **Single backward sweep**: The sensitivity kernel $h(s)$ — a single column of the impulse response matrix — captures all $M$ aggressor contributions. Computational cost: $O(L \cdot (n_{\text{nnz}} + M))$ with $O(n + M)$ memory.

3. **Physical interpretation**: $h(s)$ is the RC network's response to an impulse at the victim, decaying outward with distance and time. Nearby, recent switching dominates; distant, historical switching contributes through slow global modes (Appendix B).

4. **Generalization**: The Lagrangian adjoint formulation (Appendix C) recovers the same result for LTI systems and extends to nonlinear and time-varying networks where the impulse response approach does not apply.

## Limitations

- **Memory truncation**: Contributions from before the backward sweep window are invisible.
- **No inductance**: Package-level $Ldi/dt$ droop is not captured. Extending to RLC requires a non-symmetric formulation.
- **Flat formulation**: The distributed DDM extension (tile-based Schur complement) is implemented but documented separately.

## Extensions

- **Trapezoidal rule**: Uses $A = G_{uu} + 2C_{uu}/\Delta t_s$, $B = 2C_{uu}/\Delta t_s - G_{uu}$.
- **Distributed DDM**: Per-tile local sweeps with a coordinator interface solve.
- **RLC networks**: Coupled backward sweep on voltage and current variables.


# Appendix A: Notation Table

| Symbol | Description | Units |
|--------|-------------|-------|
| $G_{uu}$ | Reduced conductance matrix (unknowns only) | mS |
| $C_{uu}$ | Reduced capacitance matrix (unknowns only) | fF |
| $A$ | System matrix: $G_{uu} + C_{uu}/\Delta t_s$ | mS |
| $B$ | History matrix: $C_{uu}/\Delta t_s$ | mS |
| $V_u$ | Voltage vector at unknown nodes | V |
| $V_{dd}$ | Supply voltage | V |
| $I_j(t)$ | Current drawn by source $j$ at time $t$ | mA (positive = sink) |
| $e_v$ | Unit vector at victim index $v$ | dimensionless |
| $\Phi(s)$ | State transition matrix: $\exp(-C_{uu}^{-1} G_{uu}\, s)$ | dimensionless |
| $\mathcal{G}(s)$ | Impulse response matrix: $\Phi(s)\, C_{uu}^{-1}$ | k$\Omega$/ps |
| $h(s)$ | Sensitivity kernel: $\mathcal{G}(s)\, e_v$ ($v$-th column) | k$\Omega$/ps |
| $\lambda_k$ | Discrete sensitivity kernel at step $k$ ($= \Delta t_s \cdot h_k$, absorbs integration measure) | k$\Omega$ (= V/mA) |
| $c_j$ | Blame contribution of source $j$ | mV |
| $\Delta V_v(T)$ | IR-drop at victim: $V_{dd} - V_v(T)$ | mV |
| $\eta$ | Attribution efficiency: $\sum c_j / \Delta V_v(T)$ | dimensionless |
| $\Delta t$ | Time step | seconds |
| $\Delta t_s$ | Scaled time step: $\Delta t \times 10^{12}$ | ps |
| $T$ | Observation time | seconds |
| $L$ | Number of backward steps (memory window) | integer |
| $\tau_k = 1/\mu_k$ | $k$-th RC time constant | ns |
| $\rho$ | Spectral radius of $A^{-1}B$ | dimensionless |
| $\phi_k$ | $k$-th eigenmode of $C_{uu}^{-1} G_{uu}$ | (normalized) |
| $n_j$ | Grid node at which source $j$ injects current | (node ID) |
| $N_u$ | Number of unknown nodes | integer |
| $M$ | Number of current sources | integer |


# Appendix B: Spectral Interpretation

## B.1 Eigenmode Decomposition

Consider the generalized eigenproblem:

$$G_{uu}\, \phi_k = \mu_k\, C_{uu}\, \phi_k, \qquad \phi_k^\top C_{uu}\, \phi_l = \delta_{kl}.$$

The eigenvalues $\mu_k > 0$ set the **RC time constants** $\tau_k = 1/\mu_k$. The eigenmodes $\phi_k$ are spatial patterns on the grid. The impulse response decomposes as:

$$\mathcal{G}(s) = \sum_{k=1}^{N_u} \phi_k\, \phi_k^\top\, e^{-\mu_k s}.$$

Substituting into the blame integral (Section 3.2):

$$c_j = \sum_{k=1}^{N_u} \phi_k(v)\; \phi_k(n_j)\; \underbrace{\int_0^T e^{-\mu_k(T-\tau)}\, I_j(\tau)\, d\tau}_{\text{filtered current through mode } k}.$$

## B.2 Spatial-Temporal Factorization of Blame

The spectral form reveals a fundamental factorization:

$$c_j = \sum_k \underbrace{\phi_k(v)\; \phi_k(n_j)}_{\text{spatial coupling}} \;\times\; \underbrace{\hat{I}_{j,k}(T)}_{\text{temporally filtered current}}.$$

**Low-frequency modes** (small $\mu_k$, large $\tau_k$):
- Spatially smooth, extending across the grid
- Decay slowly ($e^{-\mu_k s}$ persists for large $s$)
- Carry **far-field blame**: distant aggressors contribute through these global modes
- Integrate long current history

**High-frequency modes** (large $\mu_k$, small $\tau_k$):
- Spatially localized, concentrated near the source
- Decay rapidly ($e^{-\mu_k s} \to 0$ quickly)
- Carry **near-field blame**: only local, recent switching events contribute
- Respond primarily to the current at or near time $T$

This explains why distant aggressors contribute to IR-drop primarily through slow, global RC modes, and nearby aggressors dominate due to fast, local modes with large $\phi_k(v) \cdot \phi_k(n_j)$ products.

```
    Mode k=1 (global, slow decay):
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  tau_1 = large
     Spatially smooth across entire grid

    Mode k=2 (intermediate):
     ~~~~  ~~~~  ~~~~  ~~~~  ~~~~          tau_2 = moderate
     Half-wavelength oscillation

    Mode k=N (local, fast decay):
     ___|___                               tau_N = small
     Sharp peak near source location

    Decay envelopes:
    exp(-mu_k * s)
      |
    1 |---___  k=1 (slow)
      |      ---___
      |   ----___   k=2
      |          ---___
      |  -------___     k=N (fast)
      |___________________|______> s (time lag)
      0          5*tau_1

    Figure B.1: Spatial eigenmodes (top) and their temporal decay
    envelopes (bottom). Low-frequency modes are spatially global and
    persist in time; high-frequency modes are spatially local and
    decay rapidly.
```


# Appendix C: Lagrangian Adjoint — Generalization

## C.1 Motivation: Beyond LTI

The impulse response approach of Section 3 exploits two properties of the PDN model: **linearity** (superposition) and **time-invariance** (the impulse response $\mathcal{G}(s)$ depends only on the lag $s = T - \tau$, not on $\tau$ itself). For systems that violate either property — nonlinear circuit elements, time-varying topology, or constrained optimization over design parameters — the impulse response does not exist as a fixed kernel.

The **Lagrangian adjoint method** generalizes blame assignment to these settings. For our LTI RC PDN, it recovers exactly the same result as Section 3, providing an independent validation.

## C.2 Continuous Lagrangian Derivation

Working on the reduced system (Section 2.2), define the output of interest:

$$J = e_v^\top V_u(T).$$

Introduce the adjoint variable $\lambda(t)$ as a Lagrange multiplier enforcing the forward ODE. The Lagrangian is:

$$\mathcal{L} = e_v^\top V_u(T) + \int_0^T \lambda(t)^\top \bigl[ -I_u(t) - C_{uu}\, \dot{V}_u(t) - G_{uu}\, V_u(t) \bigr]\, dt.$$

Integrate the $\lambda^\top C_{uu}\, \dot{V}_u$ term by parts:

$$\int_0^T \lambda^\top C_{uu}\, \dot{V}_u\, dt = \bigl[\lambda^\top C_{uu}\, V_u\bigr]_0^T - \int_0^T \dot{\lambda}^\top C_{uu}\, V_u\, dt.$$

Substituting and collecting terms involving $V_u$:

$$\mathcal{L} = \bigl(e_v - C_{uu}\lambda(T)\bigr)^\top V_u(T) + \lambda(0)^\top C_{uu}\, V_u(0) + \int_0^T \bigl[\dot{\lambda}^\top C_{uu} - \lambda^\top G_{uu}\bigr] V_u\, dt - \int_0^T \lambda^\top I_u\, dt.$$

Setting the variation $\delta \mathcal{L} / \delta V_u = 0$ yields:

**Adjoint ODE:**

$$-C_{uu}\, \dot{\lambda}(t) + G_{uu}\, \lambda(t) = 0, \qquad t \in [0, T).$$

**Terminal condition** (from the boundary term at $T$):

$$C_{uu}\, \lambda(T) = e_v.$$

The **blame** of source $j$ is obtained from the variation with respect to $I_j$:

$$c_j = \int_0^T \lambda_{n_j}(\tau)\; I_j(\tau)\; d\tau.$$

## C.3 Equivalence with the Impulse Response

**Claim:** $\lambda(\tau) = h(T - \tau)$ where $h(s)$ is the sensitivity kernel from Section 3.3.

**Proof.** The sensitivity kernel satisfies:

$$C_{uu}\, \frac{dh}{ds} + G_{uu}\, h(s) = 0, \qquad C_{uu}\, h(0^+) = e_v.$$

Define $\lambda(\tau) = h(T - \tau)$. Then $\dot{\lambda}(\tau) = -h'(T - \tau)$, so:

$$-C_{uu}\, \dot{\lambda}(\tau) + G_{uu}\, \lambda(\tau) = C_{uu}\, h'(T-\tau) + G_{uu}\, h(T-\tau) = 0.$$

And at $\tau = T$: $C_{uu}\, \lambda(T) = C_{uu}\, h(0^+) = e_v$.

So $\lambda$ and $h$ satisfy the same ODE with the same boundary condition, just parameterized differently: $\lambda$ runs backward in physical time $\tau$, while $h$ runs forward in lag $s = T - \tau$. The blame integrals are identical. $\square$

## C.4 Discrete Lagrangian Derivation

The discrete Lagrangian over time steps $n = 0, \ldots, N-1$:

$$\mathcal{L}_d = e_v^\top V_u^{N-1} + \sum_{n=0}^{N-2} \lambda_n^\top \bigl[ -I_u^{n+1} + B\, V_u^n - A\, V_u^{n+1} \bigr].$$

Taking variations with respect to $V_u^k$ for $0 < k < N-1$ gives the **discrete adjoint recurrence**:

$$A^\top \lambda_{k-1} = B^\top \lambda_k.$$

For symmetric $A$ and $B$ (passive RC): $A\, \lambda_k = B\, \lambda_{k+1}$, with terminal condition $A\, \lambda_{N-1} = e_v$. This is identical to the discrete impulse response recurrence (Section 4.2), confirming consistency.

The discrete Lagrangian also explains the **no-$\Delta t$-factor**: varying $\mathcal{L}_d$ with respect to $I_j(t_k)$ gives $\partial \mathcal{L}_d / \partial I_j(t_k) = \lambda_k[n_j]$, so the blame $c_j = \sum_k \lambda_k[n_j] \cdot I_j(t_k)$ has no explicit $\Delta t$.

## C.5 When the Lagrangian is Needed

| Setting | Impulse Response | Lagrangian Adjoint |
|---------|-----------------|-------------------|
| LTI RC (this report) | Primary method | Equivalent, provides validation |
| Nonlinear elements | Not applicable (no superposition) | Required |
| Time-varying systems | Kernel depends on both $t$ and $\tau$ | Handles naturally |
| Optimization (decap placement) | Not applicable | Natural framework |
| RLC with inductance | Requires non-symmetric extension | Handles naturally |

For our LTI RC PDN, the impulse response derivation (Section 3) is complete and sufficient. The Lagrangian is not needed but validates the result from a different mathematical angle and provides the generalization path for future extensions.
