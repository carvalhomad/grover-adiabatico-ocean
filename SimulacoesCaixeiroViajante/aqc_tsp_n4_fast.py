import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.integrate import solve_ivp

# ----------------------------
# Problema (N=4) e pesos
# ----------------------------
N = 4
A = 10.0   # penalidade (restrições)
B = 1.0    # custo (distâncias)
T = 12.0   # tempo total (aumente depois)
n_s = 21   # pontos no espectro (aumente depois)
n_t = 61   # pontos no P0(t) (aumente depois)

D = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10,4, 8, 0],
], dtype=float)

n = N*N
dim = 1 << n  # 2^n

# Mapeamento u = i*N + p (bit u representa x_{i,p})
def bit_u(i, p):
    return i*N + p

# ----------------------------
# 1) Construir H_P como DIAGONAL (vetor de energias)
# ----------------------------
def build_HP_diag():
    hp = np.zeros(dim, dtype=np.float64)

    # Pré-compute para acelerar: para cada bit u, qual (i,p)
    ip_of_u = [(u // N, u % N) for u in range(n)]

    # Para acelerar o custo: lista de termos (u,v,coef) de custo
    # custo soma D_ij x_{i,p} x_{j,p+1}
    cost_terms = []
    for p in range(N):
        q = (p + 1) % N
        for i in range(N):
            u = bit_u(i, p)
            for j in range(N):
                if i == j:
                    continue
                v = bit_u(j, q)
                cost_terms.append((u, v, D[i, j]))

    # Loop em todos os estados computacionais |z>
    # Cada estado "state" codifica todos x_{i,p} como bits
    for state in range(dim):
        # Ler matriz X (N x N) em forma de contagens
        # row_sum[i] = somatório p x_{i,p}
        # col_sum[p] = somatório i x_{i,p}
        row_sum = [0]*N
        col_sum = [0]*N

        # Percorre bits 1 do estado
        s = state
        while s:
            lsb = s & -s
            u = (lsb.bit_length() - 1)
            i, p = ip_of_u[u]
            row_sum[i] += 1
            col_sum[p] += 1
            s ^= lsb

        # Penalidade: sum_i (row_sum[i]-1)^2 + sum_p (col_sum[p]-1)^2
        pen = 0
        for i in range(N):
            pen += (row_sum[i] - 1)**2
        for p in range(N):
            pen += (col_sum[p] - 1)**2

        # Custo: soma dos termos ativados
        cost = 0.0
        for (u, v, coef) in cost_terms:
            if (state >> u) & 1 and (state >> v) & 1:
                cost += coef

        hp[state] = A*pen + B*cost

    return hp

print("[Build] Construindo diagonal de H_P (isso pode levar alguns segundos)...")
HP_diag = build_HP_diag()
print("[Build] OK.")

# ----------------------------
# 2) H0 = - sum sigma_x  (ação por flips de bits)
#    Em vetor psi: (H0 psi)[k] = - sum_q psi[k xor (1<<q)]
# ----------------------------
def H0_matvec(psi):
    out = np.zeros_like(psi, dtype=np.complex128)
    for q in range(n):
        out -= psi[np.arange(dim) ^ (1 << q)]
    return out

# ----------------------------
# 3) H(s) como LinearOperator: H(s)psi = (1-s)H0 psi + s HP_diag*psi
# ----------------------------
def Hs_operator(s):
    def mv(x):
        x = x.astype(np.complex128, copy=False)
        return (1.0 - s) * H0_matvec(x) + s * (HP_diag * x)
    return LinearOperator((dim, dim), matvec=mv, dtype=np.complex128)

# ----------------------------
# 4) Espectro instantâneo: E0(s), E1(s), gap(s)
# ----------------------------
s_grid = np.linspace(0.0, 1.0, n_s)
E0 = np.zeros(n_s)
E1 = np.zeros(n_s)
gap = np.zeros(n_s)

print("[Espectro] Calculando E0(s), E1(s), gap(s)...")
for k, s in enumerate(s_grid):
    print(f"  s={s:.3f} ({k+1}/{n_s})")
    Hs = Hs_operator(s)
    vals = eigsh(Hs, k=2, which="SA", return_eigenvectors=False, tol=1e-6, maxiter=3000)
    vals = np.sort(vals.real)
    E0[k], E1[k] = vals[0], vals[1]
    gap[k] = E1[k] - E0[k]

print("[Espectro] OK.")
print("gap_min =", gap.min())

# ----------------------------
# 5) Estado inicial |psi(0)> = ground state de H0
#    H0 tem ground state conhecido: |+>^{\otimes n}, vetor uniforme
# ----------------------------
psi0 = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

# Agenda s(t)
def s_linear(t):
    return t / T

# Equação de Schrödinger: dpsi/dt = -i H(s(t)) psi
def schrodinger(t, y):
    psi = y.view(np.complex128)
    s = s_linear(t)
    Hs = Hs_operator(s)
    dpsi = -1j * (Hs @ psi)
    return dpsi.view(np.float64)

t_eval = np.linspace(0.0, T, n_t)
print("[Dinâmica] Integrando Schrödinger...")
sol = solve_ivp(schrodinger, (0.0, T), psi0.view(np.float64),
                t_eval=t_eval, rtol=1e-7, atol=1e-9)
print("[Dinâmica] OK.")

# ----------------------------
# 6) Probabilidade no ground state instantâneo P0(t)
#    Aqui precisamos do autovetor fundamental em cada t.
# ----------------------------
P0 = np.zeros_like(t_eval, dtype=float)

print("[Dinâmica] Calculando P0(t) (ground state instantâneo em cada ponto)...")
for idx, t in enumerate(t_eval):
    s = s_linear(t)
    Hs = Hs_operator(s)
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=1e-6, maxiter=3000)
    g = vecs[:, 0]
    psi = sol.y[:, idx].view(np.complex128)
    P0[idx] = np.abs(np.vdot(g, psi))**2

print("[Dinâmica] OK.")
print("P0(T) =", P0[-1])

# (Opcional) Gráficos
try:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(s_grid, E0, label="E0")
    plt.plot(s_grid, E1, label="E1")
    plt.plot(s_grid, gap, label="Δ=E1−E0")
    plt.legend()
    plt.xlabel("s")
    plt.title("Espectro instantâneo")
    plt.show()

    plt.figure()
    plt.plot(t_eval, P0)
    plt.xlabel("t")
    plt.ylabel("P0(t)")
    plt.title("Probabilidade de estar no estado fundamental instantâneo")
    plt.show()

except Exception as e:
    print("[Aviso] matplotlib indisponível ou erro ao plotar:", e)