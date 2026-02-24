Computação Quântica Adiabática Aplicada ao TSP (N = 4)
Visão Geral do Projeto

Este repositório documenta uma investigação numérica completa da Computação Quântica Adiabática (AQC) aplicada ao Problema do Caixeiro Viajante (TSP) com quatro cidades.

Os objetivos do estudo foram:

Formular o TSP como um problema QUBO

Construir o Hamiltoniano problema 
𝐻
𝑃
H
P
	​


Definir um Hamiltoniano inicial do tipo campo transversal

𝐻
0
=
−
∑
𝜎
𝑥
H
0
	​

=−∑σ
x
	​


Analisar o espectro instantâneo de

𝐻
(
𝑠
)
=
(
1
−
𝑠
)
𝐻
0
+
𝑠
𝐻
𝑃
H(s)=(1−s)H
0
	​

+sH
P
	​


Calcular o gap espectral mínimo 
Δ
min
⁡
Δ
min
	​


Integrar numericamente a equação de Schrödinger dependente do tempo

Avaliar a probabilidade de permanência no estado fundamental 
𝑃
0
(
𝑡
)
P
0
	​

(t)

Todas as simulações foram realizadas classicamente em Python.

1. Codificação do Problema

Para 
𝑁
=
4
N=4 cidades:

Número de variáveis binárias: 
𝑁
2
=
16
N
2
=16

Dimensão do espaço de Hilbert: 
2
16
=
65.536
2
16
=65.536

O Hamiltoniano problema inclui:

Penalidades de restrição (coeficiente 
𝐴
=
10,0
A=10,0)

Termo de distância (coeficiente 
𝐵
=
1,0
B=1,0)

2. Instâncias Investigadas

Três configurações geométricas foram testadas.

2.1 Quadrado Simétrico

Arestas de comprimento unitário

Diagonais iguais a 
2
2
	​


Degenerescência natural (rota no sentido horário e anti-horário)

2.2 Retângulo Levemente Deformado

Pequena quebra de simetria geométrica

Tentativa de remover degenerescência estrutural

2.3 Retângulo com Custos Direcionais (ATSP)

Distâncias assimétricas 
𝑑
𝑖
𝑗
≠
𝑑
𝑗
𝑖
d
ij
	​


=d
ji
	​


Objetivo: eliminar degenerescência energética entre rotas reversas

3. Códigos Desenvolvidos

Diversas versões foram implementadas ao longo da investigação para otimização computacional.

3.1 Lista de Arquivos
Arquivo	Status	Observações
aqc_tsp_n4_full.py	Incompleto	Execução muito lenta
aqc_tsp_n4_full_robust.py	Incompleto	Problemas de convergência (ARPACK)
aqc_tsp_n4_parallel.py	Incompleto	Erro de reshape
aqc_tsp_n4_resume.py	Concluído	Primeira execução completa
aqc_tsp_n4_square_fast.py	Incompleto	Ainda lento
aqc_tsp_n4_rect_atsp_fast.py	Incompleto	Teste preliminar
aqc_tsp_n4_rect_atsp_parallel.py	Incompleto	Saída redundante
aqc_tsp_n4_rect_atsp_parallel_clean.py	Concluído	Implementação final estável
4. Resultados Numéricos Finais

Instância final analisada: Retângulo + ATSP

4.1 Gap Espectral Mínimo
Δ
min
⁡
=
7,27
×
10
−
8
Δ
min
	​

=7,27×10
−8

Mesmo após:

Quebra de simetria geométrica

Introdução de assimetria direcional

o gap permaneceu extremamente pequeno.

4.2 Evolução Adiabática

Com tempo total:

𝑇
=
20
T=20

Obteve-se:

𝑃
0
(
𝑇
)
≈
0,1775
P
0
	​

(T)≈0,1775

A evolução foi fortemente não adiabática.

Observou-se uma transição diabática significativa na região do gap mínimo.

5. Figuras Geradas

Foram geradas automaticamente:

espectro.png

p0.png

As figuras mostram:

Anti-crossing estreito

Região crítica localizada

Queda acentuada na população do estado fundamental

6. Sobre a Figura do Grafo das Cidades

A figura do grafo das cidades não foi incluída porque:

Apenas 4 vértices estão envolvidos

A geometria é trivial (retângulo levemente deformado)

A matriz de distâncias está explicitamente definida no código

A visualização não acrescenta informação relevante à análise espectral

A inclusão poderia ser feita apenas para fins didáticos.

7. Desempenho Computacional

Tempo total da execução final:

4463 s (~74 minutos)

Distribuição:

66 min → cálculo do espectro

7 min → integração da equação de Schrödinger

26 s → cálculo de 
𝑃
0
(
𝑡
)
P
0
	​

(t)

1,5 s → geração das figuras

Mesmo com paralelização em 12 núcleos, o cálculo espectral dominou o tempo total.

8. Interpretação Física

Os resultados estão alinhados com a literatura sobre AQC:

Anti-crossings estreitos são comuns

Gaps podem ser extremamente pequenos

Schedules lineares são vulneráveis

Transições diabáticas surgem naturalmente

Mesmo em 
𝑁
=
4
N=4, o gap já foi da ordem de 
10
−
8
10
−8
.

Isso sugere que:

A dificuldade não é apenas degenerescência geométrica

Penalidades quadráticas podem induzir avoided crossings estreitos

O driver transversal simples pode não ser ideal

9. Conclusões Científicas

Quebra de simetria geométrica não eliminou o pequeno gap.

Introdução de direcionalidade não removeu o gargalo espectral.

Evolução com schedule linear mostrou-se insuficiente.

O gargalo espectral parece estrutural ao encoding escolhido.

Isso não significa que o TSP não possa ser tratado por AQC.

Significa que:

O caminho adiabático e a escolha do encoding são determinantes para o comportamento espectral.

10. Próximas Direções

Sugestões para investigação futura:

Schedule adiabático local

Drivers alternativos

Redução do espaço de Hilbert (eliminação de estados inválidos)

Estudo sistemático variando 
𝐴
A

Busca adaptativa do gap mínimo

Comparação com Simulated Annealing
