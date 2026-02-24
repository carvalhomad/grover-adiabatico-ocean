import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

start_total = time.time()

# ============================================================
# PARÂMETROS DO PROBLEMA
# ============================================================

N = 4
A = 10.0
B = 1.0

# Exija mais da máquina aqui se quiser:
T = 25.0      # tempo total de evolução
n_s = 51      # pontos do espectro
n_t = 101     # pontos de tempo para P0(t)

D = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10,4, 8, 0],
], dtype=float)

n = N*N
dim = 1 << n  # 2^n

# ============================================================
# UTILIDADES
# ============================================================

def bit_u(i, p):
    return i*N + p

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
        row_sum = [0]*N
        col_sum = [0]*N

        s = state
        while s:
            lsb = s & -s
            u = (lsb.bit_length() - 1)
            i, p = ip_of_u[u]
            row_sum[i] += 1
            col_sum[p] += 1
            s ^= lsb

        pen = sum((row_sum[i] - 1)**2 for i in range(N))
        pen += sum((col_sum[p] - 1)**2 for p in range(N))

        cost = 0.0
        for (u, v, coef) in cost_terms:
            if (state >> u) & 1 and (state >> v) & 1:
                cost += coef

        hp[state] = A*pen + B*cost

    return hp

print("[1/5] Construindo H_P diagonal...")
HP_diag = build_HP_diag()
print("      OK.")

# ============================================================
# H0 = - Σ σ_x
# ============================================================

def H0_matvec(psi):
    out = np.zeros_like(psi, dtype=np.complex128)
    for q in range(n):
        out -= psi[np.arange(dim) ^ (1 << q)]
    return out

# ============================================================
# H(s)
# ============================================================

def Hs_operator(s):
    def mv(x):
        x = x.astype(np.complex128, copy=False)
        return (1.0 - s) * H0_matvec(x) + s * (HP_diag * x)
    return LinearOperator((dim, dim), matvec=mv, dtype=np.complex128)

# ============================================================
# ESPECTRO INSTANTÂNEO
# ============================================================

print("[2/5] Calculando espectro...")
s_grid = np.linspace(0.0, 1.0, n_s)
E0 = np.zeros(n_s)
E1 = np.zeros(n_s)
gap = np.zeros(n_s)

for k, s in enumerate(s_grid):
    print(f"      s = {s:.3f}  ({k+1}/{n_s})")
    Hs = Hs_operator(s)
    vals = eigsh(Hs, k=2, which="SA", return_eigenvectors=False,
                 tol=1e-6, maxiter=4000)
    vals = np.sort(vals.real)
    E0[k], E1[k] = vals[0], vals[1]
    gap[k] = E1[k] - E0[k]

print("      gap_min =", gap.min())

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
# PROBABILIDADE NO ESTADO FUNDAMENTAL
# ============================================================

print("[4/5] Calculando P0(t)...")
P0 = np.zeros_like(t_eval)

for idx, t in enumerate(t_eval):
    Hs = Hs_operator(s_linear(t))
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=1e-6, maxiter=4000)
    g = vecs[:, 0]
    psi = sol.y[:, idx].view(np.complex128)
    P0[idx] = np.abs(np.vdot(g, psi))**2

print("      P0(T) =", P0[-1])

# ============================================================
# GRÁFICOS
# ============================================================

print("[5/5] Gerando gráficos...")

plt.figure()
plt.plot(s_grid, E0, label="E0")
plt.plot(s_grid, E1, label="E1")
plt.plot(s_grid, gap, label="Gap")
plt.legend()
plt.xlabel("s")
plt.title("Espectro Instantâneo")
plt.show()

plt.figure()
plt.plot(t_eval, P0)
plt.xlabel("t")
plt.ylabel("P0(t)")
plt.title("Probabilidade de permanecer no estado fundamental")
plt.show()

print("\nTempo total:", round(time.time() - start_total, 2), "segundos")