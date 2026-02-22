"""
grover_adiabatico_8qubits.py
========================================
"Grover adiabático" (versão de validação do Hamiltoniano final) usando Ocean SDK
em SIMULAÇÃO CLÁSSICA (Simulated Annealing).

O que este script faz (com rigor e sem prometer dinâmica quântica completa):
- Constrói um Hamiltoniano Ising simples cujo estado fundamental corresponde ao estado-alvo
  (neste exemplo: número 3 em 8 qubits -> |00000011>).
- Usa o SimulatedAnnealingSampler (clássico) para encontrar o mínimo de energia.
- Converte a melhor amostra de spins para bits e para o número inteiro.
- Mostra estatísticas de sucesso (quantas vezes o alvo apareceu em num_reads).

Requisitos:
- ambiente virtual ativo (ex.: .venv)
- pacotes do Ocean instalados:
    pip install dwave-ocean-sdk

Execução (PowerShell):
    ./.venv/Scripts/Activate.ps1
    python grover_adiabatico_8qubits.py
"""

from __future__ import annotations

import sys
from collections import Counter

from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler  # <-- CORRETO (não é dwave.system)


# -----------------------------
# Utilitários
# -----------------------------
def int_to_bits(x: int, nbits: int) -> list[int]:
    """Converte inteiro x para lista de bits (MSB -> LSB) com nbits."""
    if x < 0:
        raise ValueError("x deve ser não-negativo.")
    bits_lsb_first = [(x >> i) & 1 for i in range(nbits)]
    return list(reversed(bits_lsb_first))


def bits_to_int(bits_msb_first: list[int]) -> int:
    """Converte lista de bits (MSB -> LSB) para inteiro."""
    value = 0
    for b in bits_msb_first:
        if b not in (0, 1):
            raise ValueError("bits devem ser 0 ou 1.")
        value = (value << 1) | b
    return value


def bits_to_spins(bits_msb_first: list[int]) -> list[int]:
    """Mapeamento bit -> spin (Ising): 0 -> -1, 1 -> +1."""
    return [-1 if b == 0 else +1 for b in bits_msb_first]


def spins_dict_to_bits(sample: dict[int, int], nbits: int) -> list[int]:
    """Converte amostra {i: spin} em bits (MSB -> LSB) assumindo i=0..nbits-1 na ordem MSB->LSB."""
    # Aqui mantemos a convenção: variável 0 é o bit mais significativo (MSB)
    bits = []
    for i in range(nbits):
        s = sample[i]
        if s not in (-1, +1):
            raise ValueError("spin deve ser -1 ou +1.")
        bits.append(0 if s == -1 else 1)
    return bits


def format_bits(bits: list[int]) -> str:
    """Formata bits como string '0101...'."""
    return "".join(str(b) for b in bits)


# -----------------------------
# Parâmetros do problema
# -----------------------------
N_QUBITS = 8
TARGET_NUMBER = 3
NUM_READS = 1000

# -----------------------------
# Diagnóstico do ambiente
# -----------------------------
print("=== Diagnóstico do ambiente ===")
print("Python executável :", sys.executable)
print("Versão do Python  :", sys.version.split()[0])
print()

# -----------------------------
# 1) Estado-alvo
# -----------------------------
target_bits = int_to_bits(TARGET_NUMBER, N_QUBITS)  # MSB -> LSB
target_spins = bits_to_spins(target_bits)

print("=== Estado-alvo ===")
print(f"Número alvo       : {TARGET_NUMBER}")
print(f"Bits (MSB->LSB)   : {format_bits(target_bits)}  (lista: {target_bits})")
print(f"Spins (Ising)     : {target_spins}  (0->-1, 1->+1)")
print()

# -----------------------------
# 2) Construção do Hamiltoniano Ising (simples)
# -----------------------------
# Um Hamiltoniano de campos locais que minimiza a energia exatamente no padrão alvo:
#   E(s) = sum_i h_i s_i
# Se escolhermos h_i = - target_spins[i], então o mínimo ocorre quando s_i = target_spins[i]
# (pois cada termo vira -1 e a soma fica mais negativa possível).
h = {i: -target_spins[i] for i in range(N_QUBITS)}
J = {}  # sem acoplamentos

bqm = BinaryQuadraticModel.from_ising(h, J)

print("=== Hamiltoniano Ising (BQM) ===")
print("Campos locais h_i:")
for i in range(N_QUBITS):
    print(f"  h[{i}] = {h[i]:+d}")
print("Acoplamentos J_ij: (vazio)")
print()

# -----------------------------
# 3) Sampler clássico (Recozimento Simulado)
# -----------------------------
sampler = SimulatedAnnealingSampler()

print("=== Executando recozimento simulado (clássico) ===")
sampleset = sampler.sample(bqm, num_reads=NUM_READS)

# -----------------------------
# 4) Melhor amostra e conversões
# -----------------------------
best = sampleset.first
best_sample = best.sample  # dict: {var: spin}
best_energy = best.energy

best_bits = spins_dict_to_bits(best_sample, N_QUBITS)
best_number = bits_to_int(best_bits)

print("\n=== Melhor solução ===")
print("Spins encontrados  :", best_sample)
print("Energia            :", best_energy)
print("Bits encontrados   :", format_bits(best_bits), f"(lista: {best_bits})")
print("Número encontrado  :", best_number)
print()

# -----------------------------
# 5) Estatística de sucesso
# -----------------------------
# Contar quantas vezes o estado alvo apareceu nas leituras
target_bitstring = format_bits(target_bits)
counts = Counter()

for row in sampleset.data(fields=["sample", "energy", "num_occurrences"]):
    sample_dict = row.sample
    occ = row.num_occurrences
    bits = spins_dict_to_bits(sample_dict, N_QUBITS)
    counts[format_bits(bits)] += occ

successes = counts[target_bitstring]
success_rate = successes / NUM_READS

print("=== Estatística ===")
print(f"Total de leituras (num_reads) : {NUM_READS}")
print(f"Ocorrências do alvo ({target_bitstring}) : {successes}")
print(f"Taxa de sucesso               : {success_rate:.3f}")
print()

# Mostrar as 5 strings mais frequentes
print("Top-5 resultados mais frequentes (bitstring: contagem):")
for bitstring, cnt in counts.most_common(5):
    print(f"  {bitstring}: {cnt}")

print("\nObservação:")
print("- Este script valida a codificação do estado-alvo como mínimo do Hamiltoniano final.")
print("- Ele NÃO simula a dinâmica unitária adiabática completa (Roland–Cerf) — é um solver clássico.")