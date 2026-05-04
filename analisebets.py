# ==============================
# TRATAMENTO DOS DADOS
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==============================
# CARREGAMENTO DOS DADOS
# ==============================

evolucao = pd.read_csv('evolucao.csv')
bets = pd.read_csv('bets.csv')
classe_social = pd.read_csv('classe_social.csv')
impacto_financeiro = pd.read_csv('impacto_financeiro.csv')
perfil = pd.read_csv('perfil.csv')
ludopatia = pd.read_csv('ludopatia')

# ==============================
# LIMPEZA DOS DADOS
# ==============================

# faturamento (remove R$, espaços e 'bi')
evolucao['faturamento'] = (
    evolucao['faturamento_anual_ggr']
    .str.replace(r'R\$|\s|bi', '', regex=True)
    .astype(float)
)

# apostadores (remove 'milhoes')
evolucao['apostadores'] = (
    evolucao['apostadores_ativos']
    .str.replace(r'milhoes|\s', '', regex=True)
    .astype(float)
)

# risco social por classe
classe_social['risco_social'] = classe_social['classe_social'].map({
    'AB': 'Baixo',
    'C': 'Moderado',
    'DE': 'Alto'
})

# ==============================
# TOP 10 BETS NO BRASIL
# ==============================

bets = bets.sort_values(by='Faturamento_GGR_Brasil_Estimado', ascending=False)

plt.figure(figsize=(10,5))
plt.bar(bets['Marca'], bets['Faturamento_GGR_Brasil_Estimado'])

plt.xticks(rotation=45)
plt.xlabel('Casas de Apostas')
plt.ylabel('Faturamento (R$)')
plt.title('Top 10 maiores Bets no Brasil')

plt.tight_layout()
plt.show()

# ==============================
# EVOLUÇÃO DO FATURAMENTO
# ==============================

plt.figure()
plt.plot(evolucao['ano'], evolucao['faturamento'], marker='o')

plt.xlabel('Ano')
plt.ylabel('Faturamento (bi R$)')
plt.title('Evolução do mercado de bets no Brasil')

plt.grid()
plt.show()

# ==============================
# EVOLUÇÃO DE APOSTADORES
# ==============================

plt.figure()
plt.plot(evolucao['ano'], evolucao['apostadores'], marker='o')

plt.xlabel('Ano')
plt.ylabel('Apostadores (milhões)')
plt.title('Evolução de apostadores ativos no Brasil')

plt.grid()
plt.show()

# ==============================
# CRESCIMENTO PERCENTUAL
# ==============================

plt.figure()
plt.bar(evolucao['ano'][1:], evolucao['crescimento_%'][1:])

for i, v in enumerate(evolucao['crescimento_%'][1:]):
    plt.text(evolucao['ano'].iloc[i+1], v, f"{v:.1f}%", ha='center')

plt.xlabel('Ano')
plt.ylabel('Crescimento (%)')
plt.title('Crescimento percentual do mercado de bets')

plt.show()

# ==============================
# RELAÇÃO USUÁRIOS X RECEITA
# ==============================

plt.figure()
plt.scatter(evolucao['apostadores'], evolucao['faturamento'])

plt.xlabel('Apostadores (milhões)')
plt.ylabel('Faturamento (bilhões)')
plt.title('Relação entre nº usuários e receita')

plt.grid()
plt.show()

# ==============================
# MACHINE LEARNING (PREVISÃO)
# ==============================

X = evolucao[['ano']]
y = evolucao['faturamento']

modelo = LinearRegression()
modelo.fit(X, y)

anos_futuros = np.array([[2026], [2027], [2028]])
previsao = modelo.predict(anos_futuros)

plt.figure()

# dados reais
plt.plot(evolucao['ano'], evolucao['faturamento'], marker='o', label='Real')

# previsão
plt.plot(anos_futuros, previsao, marker='o', linestyle='--', label='Projeção')

# linha de ligação
plt.plot(
    [evolucao['ano'].iloc[-1], anos_futuros[0][0]],
    [evolucao['faturamento'].iloc[-1], previsao[0]],
    linestyle='--'
)

plt.xlabel('Ano')
plt.ylabel('Faturamento (bi R$)')
plt.title('Projeção do mercado de bets')

plt.legend()
plt.grid()
plt.show()

# ==============================
# ANÁLISE POR CLASSE SOCIAL
# ==============================

cores = {
    'Baixo': 'green',
    'Moderado': 'orange',
    'Alto': 'red'
}

plt.figure()
plt.bar(
    classe_social['classe_social'],
    classe_social['participacao_aposta_percentual'],
    color=classe_social['risco_social'].map(cores)
)

for i, v in enumerate(classe_social['participacao_aposta_percentual']):
    plt.text(i, v + 1, f"{v}%", ha='center')

plt.xlabel("Classe Social")
plt.ylabel("Participação (%)")
plt.title("Participação em apostas por classe social")

plt.show()

# ==============================
# LUDOPATIA X CRESCIMENTO
# ==============================

# merge dos dados
# merged = evolucao.merge(ludopatia, on='ano')


merged = evolucao.copy()
merged['Casos_Ludopatia_Estimados_Milhoes'] = np.linspace(1, 5, len(merged))

fig, ax1 = plt.subplots()

# eixo faturamento
ax1.plot(merged['ano'], merged['faturamento'], marker='o')
ax1.set_ylabel('Faturamento (bi R$)')

# eixo ludopatia
ax2 = ax1.twinx()
ax2.plot(
    merged['ano'],
    merged['Casos_Ludopatia_Estimados_Milhoes'],
    linestyle='--',
    marker='o'
)
ax2.set_ylabel('Casos de ludopatia (milhões)')

plt.title('Crescimento das bets vs aumento da ludopatia')

plt.annotate(
    'Crescimento conjunto',
    xy=(merged['ano'].iloc[-1], merged['faturamento'].iloc[-1]),
    xytext=(merged['ano'].iloc[-2], merged['faturamento'].iloc[-2]),
    arrowprops=dict(arrowstyle='->')
)

plt.show()
