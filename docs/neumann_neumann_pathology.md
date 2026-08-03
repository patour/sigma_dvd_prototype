# Why Neumann–Neumann/BDD loses to the diagonal on torn PDN interfaces — the two-port derivation

Companion to `docs/brcm_distributed_runtime_optimization.md` §7.16–§7.17 (the
measurement record) and `docs/interface_precond_sota_research.md` (the SOTA survey
that ranked the NN/BDD fine space as Candidate 1). This doc explains the *mechanism*:
what matrix structure makes every tile Schur block near-singular while the assembled
interface operator is well-behaved, why no local-solve base (block-Jacobi slices,
true-block BJ, weighted NN at any regularization) can win here, and why the diagonal
base does. The complete pathology is reproduced in a 24-node circuit
(`scripts/benchmark/microbench/nn_pathology_demo.py`, runs in <1 s) and derived by
hand on a 2-port example below.

Measured anchors this doc explains (mi200k_v2, 64 tiles / 168,586 unknowns, cold DC
@1e-8): jacobi+PoU **34** iters; NN reg=1e-3 **282**; reg=1e-4 **869**; reg≤1e-5
maxiter; reg=0 stagnates at rel-res 1.6e-5. In the healthy 36-tile regime: jacobi
**27**, NN **111**, true-block BJ **206** — the classical in-family ordering
(NN > BJ) holds; the surprise is that the *diagonal* beats the whole family.

---

## 1. What a tile Schur block is, physically

`S_i` is the tile's Dirichlet-to-Neumann map: hold the tile's ports (the nodes on its
cut planes) at a voltage pattern `u`, let the interior relax, read the port currents.
Its quadratic form `uᵀS_i u` is the power the tile dissipates under that boundary
pattern — an eigenvalue of `S_i` is small exactly when there exists a port pattern
the tile can absorb *almost without conducting anything anywhere*.

## 2. Ports are shared cut nodes — one per crossing stripe

A node whose coordinate lies on the tile boundary appears in **both** tiles' netlist
files; the DDM makes it a single shared interface unknown, and it is a port of both
tiles. So a tile's port count is the number of nodes on its cut perimeter — one per
metal stripe crossing the cut — thousands in practice (probe: blocks up to
15,685×15,685). The mi200k logs show the multiplicity directly: 334,630 boundary
*entries* over 167,659 distinct interface *nodes* ≈ 2.0 tiles per cut node.

`S_i` is a scalar only in the degenerate case of exactly one crossing. That scalar
case matters for intuition — see §5 — because it is the case weighted NN handles
*exactly*.

## 3. Where the near-null modes come from: severed via anchoring

A PDN is not a homogeneous conducting medium. Current reaches ground/pads only
through sparse discrete channels (via stacks), and lower-metal stripes run *across*
tile cuts. When a geometric cut severs a stripe whose only via stack lies on the far
side, the near-side stripe half becomes a resistive dead-end **within that tile**:
strongly coupled to its neighbors through the tile's rails, but with (near-)zero
total conductance to anything grounded inside the tile.

Each such **anchor-severed group** of ports contributes one (near-)null eigenvector
to `S_i`: the constant pattern over the group. A blind geometric bisection slices
hundreds of stripes → hundreds of manufactured near-null directions per split tile.

Direct structural evidence (spectrum probe, §7.16): every **natural** 36-tile block
has **exactly 1** near-null direction (its floating constant — textbook, covered
exactly by the PoU coarse column), while **B1-split** sub-tiles carry **6–460**
depending on where the synthetic cut fell relative to the via stacks. Total across
the 64-tile bundle: ~2,905 directions below `1e-8·λ_max` (plateau — a genuinely
separated cluster).

These modes are **tearing artifacts, not physical low modes of S**: the champion's
own 34-iteration cold solve proves the assembled operator is well-conditioned under
a diagonal+PoU preconditioner. The mode tile A cannot ground is precisely the mode
tile B grounds hard — the anchor lives in B by construction.

## 4. The minimal true instance: two stripes, two ports

One crossing is benign (§5), so the smallest real instance has two crossings whose
A-side halves interconnect. Node roles (`[p1]`, `[p2]` sit ON the cut, shared by
both tiles; `g = 1e-4` mS is A's only path to ground — the parasitic leak):

```
                tile A                 │                tile B
                                       │
 gnd ──g── r ──2mS── a1 ──2mS──── [p1] ● ────2mS── b1 ──100mS── pad     stripe 1
           │                           │
           └──2mS─── a2 ──2mS──── [p2] ● ────2mS── b2 ──100mS── pad     stripe 2
                                       │
                                  cut plane
```

Eliminate each tile's interior (`a1, a2, r` for A; `b1, b2` for B — the pad is
Dirichlet, folded to a diagonal term):

```
S_A = [  0.5   −0.5 ]        S_B = [ 1.9608    0    ]        S = S_A + S_B
      [ −0.5    0.5 ]              [   0     1.9608 ]
```

(Exact: `S_A = [(1+g)/(2+g), −1/(2+g); −1/(2+g), (1+g)/(2+g)]`;
`S_B = (200/102)·I`.) Everything diagonalizes in the common/differential basis
`{(1,1)/√2, (1,−1)/√2}`:

| mode | λ(S_A) | λ(S_B) | λ(S) |
|---|---|---|---|
| common (1,1) | **5.0e-5** = g/(2+g) — the leak | 1.9608 | 1.9609 |
| differential (1,−1) | 1.0 — the through-A rail path | 1.9608 | 2.9608 |

Read the structure off these numbers:

- **`S_A`'s off-diagonal exists because p1 and p2 talk to each other through A's
  interior** (stripe → rail → stripe). That inter-port path is what makes `S_A` a
  genuine matrix — and it loads the *diagonal* (0.5) while contributing **zero** to
  the common mode (current pushed in at p1 comes back out at p2; the row sums cancel
  — Kirchhoff: strong internal circulation, no net path out).
- **The diagonal is blind to the weakness.** `diag(S_A) = 0.5` uniformly, yet
  `uᵀS_A u = 5e-5` for the common mode — a diagonal-to-energy contrast of 10⁴ here,
  2.4×10⁶ in the 24-node demo, and ~10⁶ measured on the real blocks (§7.8's
  `x·Mx/x·Ax`). The weakness is a *subspace* property, invisible to any per-port
  quantity.
- **Assembly heals it**: energies add (`vᵀSv = Σᵢ vᵀS_i v`), and the tiles' weak
  subspaces are complementary — B's healthy 1.9608 lands exactly on A's weak mode.
  λ_min jumps 5e-5 → 1.96. And `diag(S) = 2.46` is post-assembly data: automatically
  immune to tearing.

## 5. Why the NN recipe is trusted at all: the scalar exactness identity

The weighted NN preconditioner is `M⁻¹ = Σᵢ Rᵢᵀ Dᵢ S_i⁻¹ Dᵢ Rᵢ` with diagonal
partition-of-unity weights (`Σᵢ Dᵢ = I` per port). Stiffness weights use diagonals:
`w_A = diag(S_A)/(diag(S_A)+diag(S_B))` per port.

For a **single shared port** (1×1 blocks) this is *exact*, by algebra:

```
w_A²/s_A + w_B²/s_B  =  (s_A²/s_A + s_B²/s_B) / (s_A+s_B)²  =  1/(s_A+s_B)  =  S⁻¹
```

Stiffness-weighted NN is a harmonic-mean combiner that reproduces `S⁻¹` exactly
whenever tiles couple only through diagonals. A fully-torn scalar port is also fine
(`s_A ≈ 0 → w_A ≈ 0`: the weak side weights itself out). **Every failure below is
therefore purely an off-diagonal / subspace effect.**

## 6. The M⁻¹ derivation on the two-port example

**Weights** (per-port scalars; both ports symmetric here):

```
w_A = 0.5/2.4608 = 0.2032        w_B = 1.9608/2.4608 = 0.7968        (sum = 1 ✓)
```

**Per-tile terms** (`B_i = w_i²·S_i⁻¹`; `S_A⁻¹ ≈ [10001, 10000; 10000, 10001]`,
`S_B⁻¹ = 0.51·I`):

```
B_A ≈ [ 413.0  412.9 ]        B_B = 0.3238·I
      [ 412.9  413.0 ]

M⁻¹ = B_A + B_B ≈ [ 413.3  412.9 ]
                  [ 412.9  413.3 ]
```

Since M⁻¹ shares the eigenbasis, its action per mode is `m = w_A²/λ_A + w_B²/λ_B`:

```
m_common = 0.0413/5.0e-5 + 0.635/1.9608 = 826.1
m_diff   = 0.0413/1      + 0.635/1.9608 = 0.365
```

**Preconditioned spectrum** (`eig(M⁻¹S) = m·λ(S)` per mode):

| mode | m (NN) | λ(S) | eig(M⁻¹S) NN | eig jacobi = λ(S)/2.4608 |
|---|---|---|---|---|
| common | 826.1 | 1.9609 | **1620** | 0.797 |
| differential | 0.365 | 2.9608 | 1.081 | 1.203 |

```
κ_NN ≈ 1620/1.081 ≈ 1500          κ_jacobi ≈ 1.203/0.797 ≈ 1.5
```

A thousandfold conditioning penalty from a single torn stripe-pair.

## 7. Where exactly it breaks: per-port weights vs per-mode weakness

Apply the §5 identity **per eigenmode**. If weights could be chosen per mode,
`w = λ_A/(λ_A+λ_B)`, the same algebra gives `m = 1/(λ_A+λ_B) = 1/λ(S)` — a perfect
preconditioner, mode by mode:

- differential mode: mode-exact weight `1/(1+1.96) = 0.338`; the actual diagonal
  weight 0.2032 is close → treated well (eig 1.08).
- common mode: mode-exact weight `5e-5/(5e-5+1.96) = 2.5e-5`; the actual weight is
  **0.2032 — 8,000× too large — entering squared over a 5e-5 eigenvalue**:
  `0.2032²/5e-5 ≈ 826` instead of `≈ 0.51`.

One sentence: **the weights are one scalar per port, computed from diagonals; the
diagonal is set by the strong (differential) coupling, so the weak (common) mode
inherits the strong mode's weight — a 10³–10⁴× over-trust of the tile's
near-singular inverse, squared.** No diagonal weighting can repair this, because the
correct weight differs per *subspace*, not per port. Deluxe scaling (block weights
per shared object) is richer but still built from tile-local data — and no
computation confined to tile A can discover that the common mode's anchor lives in
tile B. That last clause is the general no-go: *the information needed to weight a
torn mode correctly does not exist inside the tile that tore it.*

## 8. The knobs, in closed form on this example

- **Tikhonov reg δ** (`interface_neumann_reg`) replaces `1/λ_A` by
  `1/(λ_A + δ·diag)`: at δ=1e-3 the common-mode term drops 826 → ~75 (still ~150×
  worse than jacobi); δ→0 climbs back to 826. This is the measured reg-ladder
  monotonicity (282 → 869 → maxiter as δ shrinks) in closed form.
- **Deflation** removes a torn mode entirely *iff its vector is in Z*. Here one
  column would fix it — which is exactly why classical BDD works (few kernels, known
  a priori: rigid-body modes/constants, covered exactly by balancing). Our probe
  counted **~2,905 independent group-modes spread over decades of λ** — the coarse
  space the family would need, and its price tag (the retained-SZ DEF projection at
  T'≈2.9k costs ~+0.25 s/iter → break-even needs warm ≤ ~5, below the plausible
  band; §7.16).
- **Jacobi** never inverts a torn object: `diag(S)` is post-assembly data, and both
  modes land within ~20% of 1 automatically. With the PoU coarse space removing the
  only *physical* low modes (per-tile smooth sags), the remaining diag-scaled
  spectrum is compact — hence 27–34 iterations.

## 9. Why the classical NN/BDD theory does not apply here

The polylog condition bounds `κ ≤ C(1+log(H/h))²` rest on assumptions this operator
violates:

1. **Ker(S_i) small and known** (rigid-body modes/constants), covered exactly by the
   balancing space. Ours: near-kernel dimension = number of severed anchor-groups —
   geometry-dependent, a priori unknown, up to ~460 per block.
2. **Quasi-uniform elliptic coefficients**: FEM subdomain Schur spectra behave like
   √(interface Laplacian) — cond ~ H/h, polynomially bounded, gapless. Ours is
   topology-channeled: 91% of block eigenvalues sit below `1e-2·λ_max`, spread over
   10+ decades of via-ladder contrast. There is no h and no H.
3. **Thin interfaces, weak inter-subdomain coupling** relative to interior
   stiffness. Our S is ~10% dense — every port couples to every other port of its
   tile through the conducting interior — far outside the trace-theorem regime the
   constants come from.

The literature's own remedy for violated assumptions 1–2 is GenEO-class adaptive
coarse spaces — "find the modes with small `uᵀS_i u` relative to assembled energy
and deflate them." The spectrum probe *is* that computation, and it returns the
price: ~2,905 columns. So "wrong family" has a precise meaning: **any preconditioner
built from torn-operator blocks must pay a coarse space proportional to the tearing
damage; a preconditioner built from assembled data pays nothing.** Corollary for
future work: if anything ever beats jacobi+PoU here, it will come from the
assembled-data family too (sparse approximate inverse of S, Chebyshev/polynomial on
diag-scaled S) — never from tile-local solves.

## 10. Scaling the 2×2 up to the measured campaign

Multiply the two-port story by ~1,450 torn groups (2,905 modes) spread over decades
of leak strength and you reproduce mi200k quantitatively: reg=0 stagnation (1e10
amplification on ~2,900 directions the 65-column PoU space cannot cover), the
monotone reg ladder, CG paying roughly one iteration per amplified direction
(24-node demo: 4 directions → +5 iters over jacobi; real bundle: hundreds), and the
healthy-regime ordering NN 111 < BJ 206 (classical theory's in-family ranking) with
both far behind jacobi 27.

Reproduce it yourself:

```bash
venv/bin/python scripts/benchmark/microbench/nn_pathology_demo.py
# S_A cond 4.8e6 with uniform diagonal; assembled κ(diag⁻¹S)=2.0;
# NN κ(M⁻¹S): 199 → 19,055 → 381,821 at reg 1e-3 / 1e-5 / 1e-7.
```

Raw campaign data: `results_neumann_*.json`, `results_tile_block_spectra_mi200k.json`
in `scripts/benchmark/microbench/`; implementation: `_build_neumann` /
`_nn_apply_*` in `src/distributed/interface_iterative.py` (knobs
`interface_two_level_base`, `interface_neumann_{weight,reg,max_bytes}`).

**Exact validation on a real (small) PDN.** `netlist/netlist_multi_tile` (9 tiles,
interface system n=112) is small enough to compute everything above exactly instead
of by sampling: every tile block has exactly one machine-zero tearing mode (the
tiles are *fully* floating — VDD arrives only via the package, the pure limit of
§3), the embedded-vs-local Rayleigh proof holds per-vector (−1e-14 local vs
0.10–0.98 embedded), and the exact κ(M⁻¹S) ladder reproduces the h2h iteration
ordering (champion 21 cold iters vs NN 42–286, bj slices 125). See
`docs/brcm_distributed_runtime_optimization.md` §7.18 and
`scripts/benchmark/microbench/analyze_interface_exact_multitile.py`.
