import os
import time
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import get_context, cpu_count
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence
from scipy.integrate import solve_ivp

# ============================================================
# CONFIGURAÇÕES (puxe a máquina aqui)
# ============================================================
N = 4
A = 10.0
B = 1.0

T = 25.0       # tempo total
n_s = 51       # pontos do espectro
n_t = 101      # pontos no tempo para P0(t)

# Quantos processos para o espectro (recomendado: 12 a 16 no i7-12700H)
NPROC = 16

# Distâncias (exemplo)
D = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0],
], dtype=float)

# ============================================================
# DIMENSÕES
# ============================================================
n = N * N
dim = 1 << n
IDX = np.arange(dim, dtype=np.int64)

def bit_u(i, p):
    return i * N + p

# ============================================================
# CONSTRÓI DIAGONAL DE H_P (vetor de energias)
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

# ============================================================
# H0 = - sum sigma_x (aplicação por flips XOR)
# ============================================================
def H0_matvec(psi):
    out = np.zeros_like(psi, dtype=np.complex128)
    for q in range(n):
        out -= psi[IDX ^ (1 << q)]
    return out

# ============================================================
# LinearOperator de H(s)
# ============================================================
def Hs_operator(s, HP_diag):
    def mv(x):
        x = x.astype(np.complex128, copy=False)
        return (1.0 - s) * H0_matvec(x) + s * (HP_diag * x)
    return LinearOperator((dim, dim), matvec=mv, dtype=np.complex128)

# ============================================================
# Solver robusto para 2 menores autovalores (para cada s)
# ============================================================
def smallest_two_eigs(Hs, base_tol=1e-6, base_maxiter=8000):
    try:
        vals = eigsh(
            Hs, k=2, which="SA",
            return_eigenvectors=False,
            tol=base_tol, maxiter=base_maxiter
        )
        vals = np.sort(vals.real)
        return vals[0], vals[1]
    except ArpackNoConvergence:
        # Recuperação: relaxa tol e aumenta iterações
        vals = eigsh(
            Hs, k=2, which="SA",
            return_eigenvectors=False,
            tol=1e-5, maxiter=30000
        )
        vals = np.sort(vals.real)
        return vals[0], vals[1]

def smallest_vec(Hs, tol=1e-6, maxiter=12000):
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=tol, maxiter=maxiter)
    return vals[0].real, vecs[:, 0]

# ============================================================
# Multiprocessing: worker do espectro
# (cada processo recebe HP_diag via initializer)
# ============================================================
_HP_SHARED = None

def _init_worker(hp_diag):
    global _HP_SHARED
    _HP_SHARED = hp_diag

def _worker_spectrum(s):
    # cada worker calcula E0,E1 no seu s
    Hs = Hs_operator(s, _HP_SHARED)
    e0, e1 = smallest_two_eigs(Hs)
    return (s, e0, e1, e1 - e0)

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("[1/6] Construindo H_P diagonal...")
    HP_diag = build_HP_diag()
    print("      OK.")

    # ---------- ESPECTRO (PARALELO) ----------
    s_grid = np.linspace(0.0, 1.0, n_s)

    # Ajuste automático: não usar mais que CPU_count()
    nproc = min(NPROC, cpu_count())
    print(f"[2/6] Espectro em paralelo com {nproc} processos (i7-12700H usa bem 12–16).")

    # No Windows, use spawn
    ctx = get_context("spawn")
    with ctx.Pool(processes=nproc, initializer=_init_worker, initargs=(HP_diag,)) as pool:
        results = pool.map(_worker_spectrum, s_grid)

    results.sort(key=lambda x: x[0])
    s_out = np.array([r[0] for r in results])
    E0 = np.array([r[1] for r in results])
    E1 = np.array([r[2] for r in results])
    gap = np.array([r[3] for r in results])

    gap_min = gap.min()
    print(f"      gap_min = {gap_min}")

    # Salva espectro
    np.savez("aqc_tsp_N4_spectrum.npz", s=s_out, E0=E0, E1=E1, gap=gap)
    print("      Espectro salvo em aqc_tsp_N4_spectrum.npz")

    # ---------- DINÂMICA (SEQUENCIAL) ----------
    print("[3/6] Integrando Schrödinger...")

    # Estado inicial |+>^n
    psi0 = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    def s_linear(t):
        return t / T

    def schrodinger(t, y):
        psi = y.view(np.complex128)
        s = s_linear(t)
        Hs = Hs_operator(s, HP_diag)
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

    # ---------- P0(t) (SEQUENCIAL) ----------
    print("[4/6] Calculando P0(t)...")
    P0 = np.zeros_like(t_eval)

    for k, t in enumerate(t_eval):
        s = s_linear(t)
        Hs = Hs_operator(s, HP_diag)
        try:
            _, g = smallest_vec(Hs, tol=1e-6, maxiter=12000)
        except ArpackNoConvergence:
            _, g = smallest_vec(Hs, tol=1e-5, maxiter=40000)

        psi = sol.y[:, k].view(np.complex128)
        P0[k] = np.abs(np.vdot(g, psi))**2

    print(f"      P0(T) = {P0[-1]}")

    # Salva dinâmica
    np.savez("aqc_tsp_N4_dynamics.npz", t=t_eval, P0=P0)
    print("      Dinâmica salva em aqc_tsp_N4_dynamics.npz")

    # ---------- GRÁFICOS ----------
    print("[5/6] Plotando...")

    plt.figure()
    plt.plot(s_out, E0, label="E0")
    plt.plot(s_out, E1, label="E1")
    plt.plot(s_out, gap, label="Δ=E1−E0")
    plt.legend()
    plt.xlabel("s")
    plt.title("Espectro instantâneo (paralelo)")
    plt.show()

    plt.figure()
    plt.plot(t_eval, P0)
    plt.xlabel("t")
    plt.ylabel("P0(t)")
    plt.title("Probabilidade no estado fundamental instantâneo")
    plt.show()

    print("[6/6] Concluído.")
    print("Tempo total:", round(time.time() - t0, 2), "s")

if __name__ == "__main__":
    main()