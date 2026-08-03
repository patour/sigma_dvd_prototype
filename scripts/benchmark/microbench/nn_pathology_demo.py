"""Minimal circuit reproducing the NN pathology: severed via anchoring.

Two tiles A|B. 8 M1 stripes cross the cut (one port each). In A, stripes are
joined in pairs by M2 rail stubs (port-port coupling through A's interior) but
each pair's only path to ground inside A is a WEAK leak (its via stack lies in
B). In B, every stripe reaches a strong via to the pad.

Shows: S_A near-singular with HEALTHY diagonal, assembled S fine, jacobi PCG
fast, weighted-NN PCG slow -- the mi200k/36-tile phenomenon in 24 nodes.
"""
import numpy as np

g_stripe = 10.0   # mS, along-stripe segment conductance
g_rail = 50.0     # A-side M2 stub joining a pair of stripes
g_via = 100.0     # B-side via to pad (Dirichlet, folded to diagonal)
leaks = [1e-3, 1e-4, 1e-5, 1e-6]   # A-side weak leaks, one per stripe-pair

m = 4  # interior nodes per stripe per side
n_stripes = 8

# Node layout per side: stripe s has interior chain; port = cut node (shared).
# Build tile conductance matrices with ports LAST, then Schur-eliminate.

def tile_matrix(side):
    # nodes: for each stripe: m interior; plus per-pair rail node (A only);
    # ports appended last (one per stripe).
    n_int = n_stripes * m + (n_stripes // 2 if side == 'A' else 0)
    n = n_int + n_stripes
    G = np.zeros((n, n))
    def add(i, j, g):
        G[i, i] += g; G[j, j] += g; G[i, j] -= g; G[j, i] -= g
    def add_diag(i, g):
        G[i, i] += g
    for s in range(n_stripes):
        base = s * m
        chain = list(range(base, base + m))
        port = n_int + s
        # port -- chain[0] -- chain[1] ... chain[m-1] (far end)
        add(port, chain[0], g_stripe)
        for a, b in zip(chain, chain[1:]):
            add(a, b, g_stripe)
        if side == 'B':
            add_diag(chain[-1], g_via)        # strong via to pad in B
    if side == 'A':
        for p in range(n_stripes // 2):
            rail = n_stripes * m + p
            s1, s2 = 2 * p, 2 * p + 1
            add(rail, s1 * m + (m - 1), g_rail)   # far ends joined by rail
            add(rail, s2 * m + (m - 1), g_rail)
            add_diag(rail, leaks[p])              # weak leak to ground in A
    return G, n_int

def schur(G, n_int):
    Gii = G[:n_int, :n_int]; Gip = G[:n_int, n_int:]; Gpp = G[n_int:, n_int:]
    return Gpp - Gip.T @ np.linalg.solve(Gii, Gip)

SA = schur(*tile_matrix('A'))
SB = schur(*tile_matrix('B'))
S = SA + SB

wA = np.linalg.eigvalsh(SA)
print(f"S_A eigenvalues: min={wA[0]:.2e}  max={wA[-1]:.2e}  "
      f"cond={wA[-1]/max(wA[0],1e-300):.1e}")
print(f"S_A diagonal:    min={SA.diagonal().min():.3f}  "
      f"max={SA.diagonal().max():.3f}   <- HEALTHY (blind to the weak modes)")
# diagonal-vs-energy contrast for the weakest eigvector
w, V = np.linalg.eigh(SA)
u = V[:, 0]
print(f"weak mode u: u'S_A u = {u @ SA @ u:.2e}   "
      f"u'diag(S_A)u = {u @ (SA.diagonal() * u):.3f}   "
      f"contrast = {(u @ (SA.diagonal() * u)) / (u @ SA @ u):.1e}")
wS = np.linalg.eigvalsh(S)
print(f"assembled S:     min={wS[0]:.3f}  max={wS[-1]:.3f}  "
      f"cond={wS[-1]/wS[0]:.1f}   <- assembly heals every weak mode")

# --- preconditioned spectra -------------------------------------------------
def pcg_iters(S, Minv, tol=1e-10, maxiter=2000):
    n = S.shape[0]
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)
    x = np.zeros(n); r = b.copy(); z = Minv @ r; p = z.copy()
    rz = r @ z; nb = np.linalg.norm(b)
    for k in range(1, maxiter + 1):
        Sp = S @ p
        alpha = rz / (p @ Sp)
        x += alpha * p; r -= alpha * Sp
        if np.linalg.norm(r) <= tol * nb:
            return k
        z = Minv @ r; rz_new = r @ z
        p = z + (rz_new / rz) * p; rz = rz_new
    return maxiter

Dj = np.diag(1.0 / S.diagonal())                       # jacobi on assembled S
dA, dB = SA.diagonal(), SB.diagonal()
WA = np.diag(dA / (dA + dB)); WB = np.diag(dB / (dA + dB))  # stiffness weights
def make_nn(reg):
    SA_r = SA + reg * np.diag(dA); SB_r = SB + reg * np.diag(dB)
    return WA @ np.linalg.inv(SA_r) @ WA + WB @ np.linalg.inv(SB_r) @ WB

for name, Minv in [('jacobi (assembled diag)', Dj),
                   ('weighted NN (reg=1e-3)', make_nn(1e-3)),
                   ('weighted NN (reg=1e-5)', make_nn(1e-5)),
                   ('weighted NN (reg=1e-7)', make_nn(1e-7))]:
    lam = np.linalg.eigvalsh(
        np.linalg.cholesky(np.linalg.inv(Minv)).T @ np.zeros((0, 0))
        if False else None) if False else None
    # generalized eigenvalues of (Minv S) via similarity with Minv^{1/2}
    wM, VM = np.linalg.eigh(Minv)
    Mh = (VM * np.sqrt(wM)) @ VM.T
    lam = np.linalg.eigvalsh(Mh @ S @ Mh)
    print(f"{name:26s} kappa(M^-1 S) = {lam[-1]/lam[0]:9.1f}   "
          f"PCG iters to 1e-10: {pcg_iters(S, Minv)}")
