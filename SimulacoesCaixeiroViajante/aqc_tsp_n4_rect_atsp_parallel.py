import os
import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.integrate import solve_ivp

# ============================================================
# CONFIGURAÇÃO
# ============================================================
N = 4
A = 10.0
B = 1.0
T = 20.0

n_s = 21       # pontos do espectro (mais que fast)
n_t = 51
n_p0 = 21

NPROC = 12     # usar 12 núcleos

epsilon = 0.07

# ============================================================
# Geometria retângulo levemente deformado
# ============================================================
coords = np.array([
    [0.0, 0.0],
    [1.05, 0.0],
    [1.05, 1.0],
    [0.0, 1.0],
])

def dist(i, j):
    return np.linalg.norm(coords[i] - coords[j])

D = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            base = dist(i, j)
            if i < j:
                D[i, j] = base * (1 + epsilon)
                D[j, i] = base * (1 - epsilon)

print("\nMatriz D (direcional):")
print(D)

# ============================================================
# Dimensão
# ============================================================
n = N * N
dim = 1 << n
IDX = np.arange(dim, dtype=np.int64)

def bit_u(i, p):
    return i * N + p

# ============================================================
# Hamiltoniano Problema
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
        row = [0]*N
        col = [0]*N

        s = state
        while s:
            lsb = s & -s
            u = lsb.bit_length() - 1
            i, p = ip_of_u[u]
            row[i] += 1
            col[p] += 1
            s ^= lsb

        penalty = sum((row[i]-1)**2 for i in range(N))
        penalty += sum((col[p]-1)**2 for p in range(N))

        cost = 0.0
        for (u, v, coef) in cost_terms:
            if ((state >> u) & 1) and ((state >> v) & 1):
                cost += coef

        hp[state] = A*penalty + B*cost

    return hp

# ============================================================
# H0
# ============================================================
def H0_matvec(psi):
    out = np.zeros_like(psi, dtype=np.complex128)
    for q in range(n):
        out -= psi[IDX ^ (1 << q)]
    return out

def Hs_operator(s, HP_diag):
    def mv(x):
        return (1-s)*H0_matvec(x) + s*(HP_diag*x)
    return LinearOperator((dim, dim), matvec=mv, dtype=np.complex128)

# ============================================================
# Função paralelizável do espectro
# ============================================================
def compute_gap_at_s(args):
    s, HP_diag = args
    Hs = Hs_operator(s, HP_diag)
    vals = eigsh(Hs, k=2, which="SA", return_eigenvectors=False)
    vals = np.sort(vals.real)
    gap = vals[1] - vals[0]
    return s, vals[0], vals[1], gap

# ============================================================
# MAIN
# ============================================================
def main():
    t_total = time.time()

    print("\n=== AQC TSP N=4 (RETÂNGULO + ATSP PARALLEL) ===")
    print(f"Usando {NPROC} processos\n")

    t0 = time.time()
    HP = build_HP_diag()
    print("HP construído. tempo =", round(time.time()-t0,2), "s")

    # ============================================================
    # ESPECTRO PARALELO
    # ============================================================
    print("\n[Espectro paralelo]")

    s_vals = np.linspace(0,1,n_s)
    args = [(s, HP) for s in s_vals]

    with mp.Pool(processes=NPROC) as pool:
        results = pool.map(compute_gap_at_s, args)

    results.sort()
    s_vals = np.array([r[0] for r in results])
    E0 = np.array([r[1] for r in results])
    E1 = np.array([r[2] for r in results])
    gap = np.array([r[3] for r in results])

    print("gap mínimo =", gap.min())

    # ============================================================
    # INTEGRAÇÃO
    # ============================================================
    print("\n[Integração Schrödinger]")

    psi0 = np.ones(dim, dtype=np.complex128)/np.sqrt(dim)

    def rhs(t, y):
        psi = y.view(np.complex128)
        s = t/T
        Hs = Hs_operator(s, HP)
        dpsi = -1j*(Hs @ psi)
        return dpsi.view(np.float64)

    t_eval = np.linspace(0,T,n_t)
    sol = solve_ivp(rhs,(0,T),psi0.view(np.float64),
                    t_eval=t_eval,rtol=1e-7,atol=1e-9)

    # ============================================================
    # P0(t)
    # ============================================================
    print("\n[P0(t)]")

    idxs = np.linspace(0,len(t_eval)-1,n_p0).astype(int)
    P0 = []

    for k in idxs:
        t = t_eval[k]
        s = t/T
        Hs = Hs_operator(s, HP)
        val, g = eigsh(Hs,k=1,which="SA")
        psi = sol.y[:,k].copy().view(np.complex128)
        P0.append(np.abs(np.vdot(g[:,0],psi))**2)

    print("P0(T) =", P0[-1])

    print("\nTempo total =", round(time.time()-t_total,2),"s")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()