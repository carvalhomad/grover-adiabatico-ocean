import numpy as np
import time
import warnings
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

start_total = time.time()

# ============================================================
# PARÂMETROS DO PROBLEMA
# ============================================================
N = 4
A = 10.0
B = 1.0

# Comece com isso; depois aumente se quiser
T = 25.0
n_s = 51
n_t = 101

D = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0],
], dtype=float)

n = N * N
dim = 1 << n  # 2^n

def bit_u(i, p):
    return i * N + p

# ============================================================
# CONSTRUÇÃO DA DIAGONAL DE H_P
# ============================================================
def build_HP_diag():
    hp = np.zeros(dim, dtype=np.float64)
    ip_of_u = [(u // N, u % N) for u in range(n)]

    cost_terms = []
    for p in range(N):
        q = (p + 1) % N
        for i in range(N):
            u = bit_u(i, p)
            for j in range(N):
                if i != j:
                    v = bit_u(j, q)
                    cost_terms.append((u, v, D[i, j]))

    for state in range(dim):
        row_sum = [0] * N
        col_sum = [0] * N

        s = state
        while s:
            lsb = s & -s
            u = (lsb.bit_length() - 1)
            i, p = ip_of_u[u]
            row_sum[i] += 1
            col_sum[p] += 1
            s ^= lsb

        pen = sum((row_sum[i] - 1) ** 2 for i in range(N))
        pen += sum((col_sum[p] - 1) ** 2 for p in range(N))

        cost = 0.0
        for (u, v, coef) in cost_terms:
            if (state >> u) & 1 and (state >> v) & 1:
                cost += coef

        hp[state] = A * pen + B * cost

    return hp

print("[1/5] Construindo H_P diagonal...")
HP_diag = build_HP_diag()
print("      OK.")

# ============================================================
# H0 = - Σ σ_x  (ação por flips de bits)
# ============================================================
idx = np.arange(dim, dtype=np.int64)

def H0_matvec(psi):
    out = np.zeros_like(psi, dtype=np.complex128)
    for q in range(n):
        out -= psi[idx ^ (1 << q)]
    return out

# ============================================================
# H(s) como LinearOperator
# ============================================================
def Hs_operator(s):
    def mv(x):
        x = x.astype(np.complex128, copy=False)
        return (1.0 - s) * H0_matvec(x) + s * (HP_diag * x)
    return LinearOperator((dim, dim), matvec=mv, dtype=np.complex128)

# ============================================================
# EIGEN SOLVER ROBUSTO
# ============================================================
def smallest_two_eigs(Hs, label="", base_tol=1e-6, base_maxiter=4000):
    """
    Tenta obter os 2 menores autovalores de Hs.
    Estratégia:
      1) eigsh padrão (SA)
      2) se falhar, aumentar maxiter e relaxar tol
      3) se falhar, usar shift-invert com sigma próximo do mínimo esperado (heurística: min diag)
      4) se tudo falhar, retorna apenas o menor autovalor (k=1) e NaN para o segundo
    """
    # Tentativa 1
    try:
        vals = eigsh(
            Hs, k=2, which="SA",
            return_eigenvectors=False,
            tol=base_tol, maxiter=base_maxiter
        )
        vals = np.sort(vals.real)
        return vals[0], vals[1]
    except ArpackNoConvergence:
        print(f"      [Aviso] ARPACK não convergiu (k=2) em {label}. Tentando recuperação...")

    # Tentativa 2: mais iterações, tol relaxada
    try:
        vals = eigsh(
            Hs, k=2, which="SA",
            return_eigenvectors=False,
            tol=1e-5, maxiter=20000
        )
        vals = np.sort(vals.real)
        print(f"      [Recuperado] convergiu com tol=1e-5, maxiter=20000 em {label}.")
        return vals[0], vals[1]
    except ArpackNoConvergence:
        print(f"      [Aviso] Ainda sem convergência (k=2) em {label}. Tentando shift-invert...")

    # Tentativa 3: shift-invert (mais robusto para autovalores extremos)
    # Heurística de sigma: mínimo da diagonal do termo HP escalado por s (aproximação) - para guiar perto do ground
    # Nota: como H0 não é diagonal, isto é apenas um guia; ainda assim costuma ajudar.
    try:
        # sigma pequeno (perto do mínimo)
        sigma = np.min(HP_diag) * 0.0  # zero é geralmente uma escolha boa para o "menor"
        vals = eigsh(
            Hs, k=2, sigma=sigma, which="LM",
            return_eigenvectors=False,
            tol=1e-6, maxiter=30000
        )
        vals = np.sort(vals.real)
        print(f"      [Recuperado] convergiu com shift-invert (sigma={sigma}) em {label}.")
        return vals[0], vals[1]
    except Exception as e:
        print(f"      [Falha] shift-invert não resolveu em {label}: {type(e).__name__}: {e}")

    # Tentativa 4: pelo menos E0(s)
    try:
        val0 = eigsh(
            Hs, k=1, which="SA",
            return_eigenvectors=False,
            tol=1e-5, maxiter=30000
        )[0].real
        print(f"      [Parcial] retornando apenas E0 em {label}.")
        return val0, np.nan
    except Exception as e:
        raise RuntimeError(f"Falha total no solver em {label}: {e}")

def smallest_vec(Hs, tol=1e-6, maxiter=4000):
    # Ground state (vetor) com tolerância ajustável
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=tol, maxiter=maxiter)
    return vals[0].real, vecs[:, 0]

# ============================================================
# ESPECTRO INSTANTÂNEO
# ============================================================
print("[2/5] Calculando espectro (robusto)...")
s_grid = np.linspace(0.0, 1.0, n_s)
E0 = np.zeros(n_s)
E1 = np.zeros(n_s)
gap = np.zeros(n_s)

for k, s in enumerate(s_grid):
    print(f"      s = {s:.3f}  ({k+1}/{n_s})")
    Hs = Hs_operator(s)
    e0, e1 = smallest_two_eigs(Hs, label=f"s={s:.3f}")
    E0[k] = e0
    E1[k] = e1
    gap[k] = e1 - e0 if np.isfinite(e1) else np.nan

gap_min = np.nanmin(gap)
print("      gap_min (ignorando NaN) =", gap_min)

# ============================================================
# ESTADO INICIAL |+>^n
# ============================================================
psi0 = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

def s_linear(t):
    return t / T

# ============================================================
# EQUAÇÃO DE SCHRÖDINGER
# ============================================================
print("[3/5] Integrando Schrödinger...")

def schrodinger(t, y):
    psi = y.view(np.complex128)
    s = s_linear(t)
    Hs = Hs_operator(s)
    dpsi = -1j * (Hs @ psi)
    return dpsi.view(np.float64)

t_eval = np.linspace(0.0, T, n_t)

sol = solve_ivp(
    schrodinger,
    (0.0, T),
    psi0.view(np.float64),
    t_eval=t_eval,
    rtol=1e-7,
    atol=1e-9
)
print("      OK.")

# ============================================================
# PROBABILIDADE NO ESTADO FUNDAMENTAL P0(t)
# ============================================================
print("[4/5] Calculando P0(t)...")
P0 = np.zeros_like(t_eval)

for idx_t, t in enumerate(t_eval):
    s = s_linear(t)
    Hs = Hs_operator(s)

    # Aqui pedimos o vetor do ground state; pode precisar de mais iterações perto do gap mínimo
    try:
        _, g = smallest_vec(Hs, tol=1e-6, maxiter=8000)
    except ArpackNoConvergence:
        # fallback mais relaxado
        _, g = smallest_vec(Hs, tol=1e-5, maxiter=30000)

    psi = sol.y[:, idx_t].view(np.complex128)
    P0[idx_t] = np.abs(np.vdot(g, psi)) ** 2

print("      P0(T) =", P0[-1])

# ============================================================
# GRÁFICOS
# ============================================================
print("[5/5] Gerando gráficos...")

plt.figure()
plt.plot(s_grid, E0, label="E0")
plt.plot(s_grid, E1, label="E1")
plt.plot(s_grid, gap, label="Δ=E1−E0")
plt.legend()
plt.xlabel("s")
plt.title("Espectro Instantâneo (robusto)")
plt.show()

plt.figure()
plt.plot(t_eval, P0)
plt.xlabel("t")
plt.ylabel("P0(t)")
plt.title("Probabilidade de permanecer no estado fundamental")
plt.show()

print("\nTempo total:", round(time.time() - start_total, 2), "segundos")