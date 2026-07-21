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

# Discrete kernel: one forward BE transient

BE matrices (same $A$ the forward engine already factors):

$$A = G + \frac{C}{\Delta t_s}, \qquad B = \frac{C}{\Delta t_s}$$

BE-discretize the kernel equation $C\,h_v'(s) + G\,h_v(s) = e_v\,\delta(s)$ — symmetric $G, C$, so no transposes:

$$\boxed{A\, h_v[0] = e_v, \qquad A\, h_v[\ell] = B\, h_v[\ell-1] \quad (\ell \ge 1)}$$

This is *exactly* a **stock forward BE transient** that

- starts from the **no-load DC state** — all loads off, so every node sits at its pad voltage $V_{dd}$ (**zero IR-drop**, $V_u = 0$),
- injects a **unit current pulse at victim $v$** (width one $\Delta t_s$) at step $0$,
- then runs **source-free** (free decay).

**Pad cancellation.** $V \equiv V_{dd}$ is the exact steady state of the pulse-free grid, so by superposition $V(t) = V_{dd} - V_{\text{pulse}}(t)$: the recorded IR-drop $\Delta V = V_{dd} - V(t)$ *is* the pure pulse response — no reference-run subtraction:

$$\boxed{h_v[\ell][n_j] = \Delta V_{n_j}(\ell\,\Delta t_s)}$$

*(unit pulse — for a pulse of amplitude $I_p$, divide by $I_p$)*

> $h_v[\ell]$ = voltage probe of one ordinary transient, $\ell$ steps after the victim pulse. No adjoint code.

---

# mPower recipe: one transient + a correlation

**Reuse the existing forward engine.** Blame = one special transient run + a post-processing sum.

**Step 1 — kernel run** *(transient iteratons)*

- zero out all instance current sources
- add **one** synthetic load at victim $v$: rectangular pulse, amplitude $I_p$, width one $\Delta t_s$
- initial condition = **no-load DC operating point** (all nodes at $V_{dd}$ — zero IR-drop)
- run $L$ steps ($L$ = memory window, a few grid RC constants); probe voltages at **aggressor nodes only**

$$\boxed{h_v[\ell][n_j] = \Delta V_{n_j}(\ell\,\Delta t_s)\,/\,I_p}$$

**Step 2 — blame accumulation** *(post-processing)*

$$\boxed{c_j = \sum_{\ell=0}^{L-1} h_v[\ell][n_j]\; I_j\!\left(T - \ell\,\Delta t_s\right)}$$

- **Lag-reversed pairing**: freshest sample $h_v[0]$ pairs with the aggressor current **at** observation time $T$.

Rank $|c_j|$ $\to$ top-$K$. Interpretation: $h_v[\ell][n_j] = \partial V_v(T) / \partial I_j(T - \ell\,\Delta t_s)$.

---

# Requirements, cost, validation

**Exactness conditions** *(all hold for an RC PDN)*

- **Integrator-consistent kernel** — BE shown. TR variant: $A = G + \tfrac{2C}{\Delta t_s}$, $B = \tfrac{2C}{\Delta t_s} - G$, same kernel run, but each sample pairs with **two** adjacent current samples: $c_j = \sum_\ell h_v[\ell][n_j]\bigl(I_j(T\!-\!\ell\Delta t_s) + I_j(T\!-\!(\ell\!+\!1)\Delta t_s)\bigr)$. BE kernels are smoother (no TR ringing).
- **Symmetric $G, C$** — buys reciprocity / self-adjointness
- **Pads at constant voltage (Dirichlet)** — guarantees exact background cancellation
- Same $\Delta t_s$ as the target analysis; **one kernel run per victim**

**Cost & memory**

- One extra $L$-step transient per victim, plus $O(M \cdot L)$ correlation
- $O(\text{probed aggressors} \times L)$ for dumped waveforms (or $O(n + M)$ if streamed)

**Pruning**

- Active aggressors only in $[T - L\,\Delta t_s,\, T]$
- Spatial: keep nodes with $\max_\ell |h_v[\ell][n_j]| > \epsilon$

**Validation** — cross-check against mPower's own forward run:

$$\boxed{\sum_j c_j \approx V_v(T)}$$

---

# Next step

**Covered today**

- Decomposition: $\Delta V_v = \Delta V_{\text{far,VCD}} + \Delta V_{\text{near,opt}}$
- Per-instance blame at a chosen victim from a single backward sweep

**Deferred to follow-up — Step 3 of the optimization flow**

Given the top-$K$ aggressor set $A$ from blame, optimize switching times $\{t_j \in W_j\}_{j \in A}$ to maximize $\Delta V_v(T)$ subject to per-instance timing-window constraints.
