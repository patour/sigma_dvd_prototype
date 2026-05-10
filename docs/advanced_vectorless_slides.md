# Advanced Vectorless

## Worst-case dynamic IR-drop via decomposition + per-instance blame

**PDN Analysis Team**  
**March 2026**

---

# Goal and approach

**Goal**

High-coverage, vector-independent simulation providing 100% power grid noise coverage for all design instances.

**Method**

- Far IR-drop: existing vector (VCD or generated vectorless)
- Near IR-drop: statistical worst case

**100% coverage**

- Identifies worst-case potential aggressor cells
- Calculates voltage drops without additional transient simulations

---

# Worst-case IR-drop is a constrained switching problem

**What drives the drop**

- Many instances switching in the same neighborhood
- Switching currents overlap in time
- Overlap is bounded by per-instance timing windows

**What makes it hard**

- Millions of instances
- Each with a window, not a fixed time
- Enumerating legal combinations is impractical

**Key observation**

> High-frequency victim impact is **local**. Distant switching contributes a smoother low-frequency background.

---

# Decomposition: near + far

![](ir_drop_decomp_near_plus_far.png){width=80%}

$$\Delta V_v(t) \approx \Delta V_{\text{far,VCD}}(t) + \Delta V_{\text{near,opt}}(t)$$

- **Far** (VCD-driven): fixed context, smoothed by the M8–M10 low-pass stack
- **Near** (optimized): victim + nearby aggressors, switching times within timing windows

---

# Algorithm overview

1. **Far context** — full-chip transient with VCD; treated as fixed background; yields $\Delta V_{\text{far}}(t)$ at every node.
2. **Blame** — find the top-$K$ aggressors for each design instance.
3. **Near optimization** — take the top-$K$ aggressors from blame; optimize switching within timing windows; yields $\Delta V_{\text{near,opt}}(t)$.

Worst-case dynamic IR-drop at the victim:

$$\Delta V_v(t) \approx \Delta V_{\text{far,VCD}}(t) + \Delta V_{\text{near,opt}}(t)$$

---

# The Blame Question

::::: columns
:::: {.column width="55%"}

![](PDNVictimAgressors.png)

::::
:::: {.column width="45%"}

**IR-drop at victim node $v$**

At time $T$, the victim sees its worst IR-drop:

$$V_v(T)=X\text{ mV}$$

**Question:**

> Which switching instances caused it, and how many mV did each contribute?

**Goal:**

$$V_v(T)\approx c_{\text{init}}+\sum_j c_j$$

- $c_j$: contribution from aggressor instance $j$
- $c_{\text{init}}$: residual initial-condition contribution
- top-$K$: largest positive $c_j$

::::
:::::

---

# Linear IR-drop model

**Reduced system** *(positive $V_u$ is IR drop)*

$$C_{uu}\dot V_u(t) + G_{uu} V_u(t) = I_u(t), \qquad V_v(t) = e_v^\top V_u(t)$$

**LTI variation of parameters**

$$V_u(T) = \Phi(T) V_u^{(0)} + \int_0^T \mathcal G(T-\tau)\, I_u(\tau)\, d\tau$$

$$\Phi(s) = e^{-C_{uu}^{-1} G_{uu}\, s}, \qquad \mathcal G(s) = \Phi(s)\, C_{uu}^{-1}$$

$\mathcal G_{i,j}(s)$ = IR drop at $i$ from a unit current impulse at $j$.

---

# Per-instance contribution and sensitivity kernel

Per-source decomposition:

$$I_u(\tau) = \sum_{j=1}^{M} e_{n_j} I_j(\tau) \;\Longrightarrow\; V_v(T) = e_v^\top \Phi(T) V_u^{(0)} + \sum_j c_j$$

$$\boxed{c_j = \int_0^T e_v^\top \mathcal G(T-\tau)\, e_{n_j}\, I_j(\tau)\, d\tau}$$

Define the victim sensitivity field, then:

$$h_v(s) \equiv \mathcal G(s)^\top\, e_v \quad\Rightarrow\quad e_v^\top \mathcal G(s)\, e_{n_j} = h_v(s)[n_j]$$

$$\boxed{c_j = \int_0^T h_v(T-\tau)[n_j]\, I_j(\tau)\, d\tau}$$

*Multi-pin stamp*: replace $h_v[n_j]$ with $s_j^\top h_v$.

---

# Reciprocity

Adjoint sensitivity equation:

$$C_{uu}^\top h_v'(s) + G_{uu}^\top h_v(s) = e_v\, \delta(s)$$

Symmetric RC: $C_{uu}^\top = C_{uu}$, $G_{uu}^\top = G_{uu}$, so:

> Inject a unit current impulse at the victim and observe the response at each aggressor node.

Reciprocity: $\mathcal G_{v,n_j}(s) = \mathcal G_{n_j,v}(s)$.

$$\boxed{\text{one victim-centered response} \Rightarrow \text{all aggressor sensitivities}}$$

---

# Discrete kernel (Backward Euler)

BE matrices:

$$A = G + \frac{C}{\Delta t_s}, \qquad B = \frac{C}{\Delta t_s}$$

Discrete sensitivity in lag $\ell$:

$$h_v^{BE}[0] = A^{-\top} e_v, \qquad A^\top h_v^{BE}[\ell] = B^\top h_v^{BE}[\ell-1]$$

Reindex to forward time $k$ with $\ell = N - 1 - k$:

$$\boxed{\lambda_k = h_v^{BE}[N - 1 - k]}$$

> $\lambda_k$ is the time-reversed BE-discrete victim sensitivity kernel.

---

# Backward sweep and accumulation

Initialize and sweep:

$$\lambda_{N-1} = A^{-\top} e_v, \qquad A^\top \lambda_k = B^\top \lambda_{k+1} \quad (k = N-2, \dots, 0)$$

Per-aggressor accumulation:

$$\boxed{c_j = \sum_{k=0}^{N-1} \lambda_k[n_j]\, I_j[k]}$$

Interpretation: $\lambda_k[n_j] = \partial V_v[N-1] / \partial I_j[k]$.

For symmetric RC, $A^\top = A$ and $B^\top = B$.

---

# Complexity, pruning, validation

Definitions: $n$ unknowns, $M$ aggressors, $N$ time steps in the blame window, horizon $H = N \Delta t_s$.

**Cost**: $O(N \cdot \text{Solve})$.  **Memory**: $O(n)$.

**Pruning**

- Active aggressors only in $[T - H, T]$
- Spatial: keep nodes with $\max_k |\lambda_k[n_j]| > \epsilon$

**Validation**: $V_v(T) \approx c_{\text{init}} + \sum_j c_j$.

$$\boxed{\text{One victim sensitivity sweep gives per-instance IR-drop blame.}}$$

---

# Next step

**Covered today**

- Decomposition: $\Delta V_v = \Delta V_{\text{far,VCD}} + \Delta V_{\text{near,opt}}$
- Per-instance blame at a chosen victim from a single backward sweep

**Deferred to follow-up — Step 3 of the optimization flow**

Given the top-$K$ aggressor set $A$ from blame, optimize switching times $\{t_j \in W_j\}_{j \in A}$ to maximize $\Delta V_v(T)$ subject to per-instance timing-window constraints.
