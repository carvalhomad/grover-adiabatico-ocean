import os
import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence
from scipy.integrate import solve_ivp

# ==========================
# Config
# ==========================
N = 4
A = 10.0
B = 1.0
T = 20.0

n_s = 21          # pontos no espectro
n_t = 51          # pontos na integração
n_p0 = 21         # pontos de P0

NPROC = 12
epsilon = 0.07

SPECTRUM_FILE = "aqc_tsp_N4_rect_atsp_spectrum.npz"
SCHRO_FILE    = "aqc_tsp_N4_rect_atsp_schrodinger.npz"
DYN_FILE      = "aqc_tsp_N4_rect_atsp_dynamics.npz"
FIG_SPEC      = "espectro.png"
FIG_P0        = "p0.png"

# ==========================
# Dimensão
# ==========================
n = N * N
dim = 1 << n
IDX = np.arange(dim, dtype=np.int64)

def bit_u(i, p):
    return i * N + p

# ==========================
# Globais para os workers
# ==========================
_HP_SHARED = None

def _init_worker(hp_diag):
    global _HP_SHARED
    _HP_SHARED = hp_diag

def H0_matvec(psi):
    out = np.zeros_like(psi, dtype=np.complex128)
    for q in range(n):
        out -= psi[IDX ^ (1 << q)]
    return out

def Hs_operator(s, HP_diag):
    def mv(x):
        x = x.astype(np.complex128, copy=False)
        return (1.0 - s) * H0_matvec(x) + s * (HP_diag * x)
    return LinearOperator((dim, dim), matvec=mv, dtype=np.complex128)

def smallest_two_eigs(Hs):
    try:
        vals = eigsh(Hs, k=2, which="SA", return_eigenvectors=False, tol=1e-6, maxiter=15000)
        vals = np.sort(vals.real)
        return vals[0], vals[1]
    except ArpackNoConvergence:
        vals = eigsh(Hs, k=2, which="SA", return_eigenvectors=False, tol=1e-5, maxiter=60000)
        vals = np.sort(vals.real)
        return vals[0], vals[1]

def _worker_spectrum(s):
    # usa HP_diag global (evita serialização a cada tarefa)
    Hs = Hs_operator(s, _HP_SHARED)
    e0, e1 = smallest_two_eigs(Hs)
    return (s, e0, e1, e1 - e0)

# ==========================
# H_P diagonal
# ==========================
def build_HP_diag(D):
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

# ==========================
# Construção de D: retângulo deformado + assimetria direcional
# ==========================
def build_D_rect_atsp():
    coords = np.array([
        [0.0, 0.0],
        [1.05, 0.0],
        [1.05, 1.0],
        [0.0, 1.0],
    ], dtype=float)

    def dist(i, j):
        return np.linalg.norm(coords[i] - coords[j])

    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            base = dist(i, j)
            # assimetria fraca e controlada
            if i < j:
                D[i, j] = base * (1 + epsilon)
                D[j, i] = base * (1 - epsilon)
    return D

def fmt(sec): return f"{sec:.2f} s"

# ==========================
# MAIN
# ==========================
def main():
    t_all = time.perf_counter()

    print("=== AQC TSP N=4 (RETÂNGULO + ATSP) PARALLEL CLEAN ===")
    print(f"A={A}, B={B}, T={T}, n_s={n_s}, n_t={n_t}, n_p0={n_p0}, NPROC={NPROC}\n")

    # 0) D
    D = build_D_rect_atsp()
    print("Matriz D (direcional):")
    print(D)

    # 1) H_P
    t0 = time.perf_counter()
    print("\n[1/6] Construindo H_P diagonal...")
    HP = build_HP_diag(D)
    print("      OK. tempo =", fmt(time.perf_counter()-t0))

    # 2) ESPECTRO (paralelo + progresso + ETA)
    if os.path.exists(SPECTRUM_FILE):
        t0 = time.perf_counter()
        print(f"\n[2/6] Carregando espectro de {SPECTRUM_FILE} (pulando recomputação).")
        dat = np.load(SPECTRUM_FILE)
        s_out, E0, E1, gap = dat["s"], dat["E0"], dat["E1"], dat["gap"]
        print(f"      gap_min = {gap.min():.6e}")
        print("      OK. tempo =", fmt(time.perf_counter()-t0))
    else:
        t0 = time.perf_counter()
        print(f"\n[2/6] Espectro em paralelo com {NPROC} processos...")
        s_grid = np.linspace(0.0, 1.0, n_s)

        ctx = mp.get_context("spawn")
        results = []
        done = 0
        tick = time.perf_counter()

        with ctx.Pool(processes=NPROC, initializer=_init_worker, initargs=(HP,)) as pool:
            for r in pool.imap_unordered(_worker_spectrum, s_grid, chunksize=1):
                results.append(r)
                done += 1

                # update a cada 1 resultado (n_s pequeno) com ETA
                dt = time.perf_counter() - tick
                rate = done / max(dt, 1e-9)
                left = n_s - done
                eta = left / max(rate, 1e-9)
                print(f"      progresso: {done}/{n_s}  ETA~{eta:.0f}s")

        results.sort(key=lambda x: x[0])
        s_out = np.array([r[0] for r in results])
        E0 = np.array([r[1] for r in results])
        E1 = np.array([r[2] for r in results])
        gap = np.array([r[3] for r in results])

        np.savez(SPECTRUM_FILE, s=s_out, E0=E0, E1=E1, gap=gap, A=A, B=B, T=T, epsilon=epsilon)
        print(f"      gap_min = {gap.min():.6e}")
        print(f"      salvo em {SPECTRUM_FILE}")
        print("      OK. tempo =", fmt(time.perf_counter()-t0))

    # 3) Schrödinger (checkpoint)
    if os.path.exists(SCHRO_FILE):
        t0 = time.perf_counter()
        print(f"\n[3/6] Carregando Schrödinger de {SCHRO_FILE} (pulando integração).")
        dat = np.load(SCHRO_FILE)
        t_eval, y_float = dat["t_eval"], dat["y_float"]
        print("      OK. tempo =", fmt(time.perf_counter()-t0))
    else:
        t0 = time.perf_counter()
        print("\n[3/6] Integrando Schrödinger...")
        psi0 = np.ones(dim, dtype=np.complex128)/np.sqrt(dim)

        def rhs(t, y):
            psi = y.view(np.complex128)
            s = t / T
            Hs = Hs_operator(s, HP)
            dpsi = -1j * (Hs @ psi)
            return dpsi.view(np.float64)

        t_eval = np.linspace(0.0, T, n_t)
        sol = solve_ivp(rhs, (0.0, T), psi0.view(np.float64),
                        t_eval=t_eval, rtol=1e-7, atol=1e-9)

        y_float = sol.y
        np.savez(SCHRO_FILE, t_eval=t_eval, y_float=y_float, A=A, B=B, T=T, epsilon=epsilon)
        print(f"      OK. salvo em {SCHRO_FILE}")
        print("      tempo =", fmt(time.perf_counter()-t0))

    # 4) P0(t) (amostrado, para não custar caro)
    t0 = time.perf_counter()
    print(f"\n[4/6] Calculando P0(t) em {n_p0} pontos...")
    idxs = np.linspace(0, len(t_eval)-1, n_p0).astype(int)
    t_p0 = t_eval[idxs]
    P0 = np.zeros(n_p0, dtype=float)

    v0 = None
    for j, k in enumerate(idxs, start=1):
        Hs = Hs_operator(t_eval[k]/T, HP)
        vals, vecs = eigsh(Hs, k=1, which="SA", v0=v0, tol=1e-6, maxiter=25000)
        g = vecs[:, 0]
        v0 = np.real(g)

        psi = y_float[:, k].copy().view(np.complex128)
        P0[j-1] = np.abs(np.vdot(g, psi))**2
        if j % 5 == 0 or j == n_p0:
            print(f"      P0 progresso: {j}/{n_p0}   (t={t_p0[j-1]:.2f}, P0={P0[j-1]:.4f})")

    np.savez(DYN_FILE, t=t_p0, P0=P0, A=A, B=B, T=T, epsilon=epsilon)
    print(f"      P0(T) ~ {P0[-1]:.6f}")
    print(f"      salvo em {DYN_FILE}")
    print("      OK. tempo =", fmt(time.perf_counter()-t0))

    # 5) Figuras
    t0 = time.perf_counter()
    print("\n[5/6] Gerando figuras...")
    plt.rcParams.update({"font.size": 14})

    plt.figure(figsize=(10,7))
    plt.plot(s_out, E0, linewidth=2, label="E0")
    plt.plot(s_out, E1, linewidth=2, label="E1")
    plt.plot(s_out, gap, linewidth=2, label="Δ")
    plt.xlabel("s")
    plt.ylabel("Energia")
    plt.title("Espectro instantâneo (amostrado)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(FIG_SPEC, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,7))
    plt.plot(t_p0, P0, linewidth=2, marker="o")
    plt.xlabel("t")
    plt.ylabel("P0(t)")
    plt.title("P0(t) (amostrado)")
    plt.grid(alpha=0.3)
    plt.savefig(FIG_P0, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"      salvo: {FIG_SPEC}, {FIG_P0}")
    print("      OK. tempo =", fmt(time.perf_counter()-t0))

    # 6) Fim
    print("\n[6/6] Concluído.")
    print("Tempo total =", fmt(time.perf_counter()-t_all))


if __name__ == "__main__":
    # obrigatório no Windows
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()