import numpy as np
import matplotlib.pyplot as plt

# Carrega o dataset 'aerogerador.dat'.
df = np.loadtxt('aerogerador.dat')

# Qntd. de Amostras (N) e Qntd. de variáveis (p)
N, p = df.shape

"""
1. Faça uma visualização inicial dos dados através do gráfico
de espalhamento. Nessa etapa levante hipóteses sobre quais serão
as características de um modelo que consegue entender o padrão
entre variáveis regressoras e variáveis observadas.
"""

# Plota um gráfico de dispersão do dataset.
plt.figure(figsize=(10, 6))
plt.scatter(df[:,0], df[:,1], c='blue')
plt.xlabel('Variável Independentes')
plt.ylabel('Variável Dependentes')
plt.title(
    'Gráfico de Dispersão\n' +
    'Relação entre Variáveis Dependentes e Independentes'
)
plt.show()

"""
2. Em seguida, organize os dados de modo que as variáveis regressoras
sejam armazenadas em uma matriz de dimensão R^Nxp. Faça o memso para
o vetor de variáveis observadas, organizando em um vetor de dimensão
R^Nx1.
"""

# Separa as variáveis dependentes e independentes.
x = df[:, 0].reshape(N,1) # R^Nxp (+ interceptor)
y = df[:, 1].reshape(N,1) # R^Nx1

"""
3. Para validar os modelos utilizados na tarefa de regressão, é necessário
definir uma quantidade específica de rodadas de treinamento e teste dos
modelos. Assim, defina essa quantidade de rodadas com o valor 1000.
"""

# Quantidade de épocas.
epochs = 1000

"""
4. Os modelos a serem implementados nessa estapa são: 
    MQO tradicional, 
    MQO regularizado (Tikhonov),
    Média de valores observáveis.
"""

def MQO_tradicional(X, y):
    """MQO_tradicional:
    Regressão Linear de Mínimos Quadrados Ordinários (MQO).
        y = X.B + e
        y - variáveis dependentes.
        X - matriz de variáveis independentes.
        B - estimativa dos coeficientes do modelo.
        e - erro aleatório (ruído)
        B = (X^T.X)^-1.X^T.y
    Objetivo: 
    Encontrar os valores de B que minimizam a soma
    dos quadrados dos resíduos.
    """
    # Adiciona uma coluna de 1s (uns) em X (interceptor).
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Calcula a matriz de coeficientes.
    # B = (X^T.X)^-1.X^T.y
    coef = np.linalg.inv(X.T @ X) @ X.T @ y

    # Retorna os coeficientes e o interceptor.
    return coef[1:], coef[0]

def MQO_regularizado(X, y, alpha):
    """MQO_regularizado:
    Extensão do MQO tradicional que inclui um termo de
    regularização na função de perda.
        Loss = ∑i=1,n(yi - y_hati)^2 + a.∑j=1,p.B^2,j
        1. Soma dos quadrados dos resíduos.
        2. Termo de Regularização λ (lambda).
        3. Bj - Coeficientes dos recursos.
    Objetivo:
    Minimizar essa função de perda (Loss) modificada.
    """
    # Adiciona uma coluna de 1s (uns) em X (interceptor).
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Calcula a matriz de coeficientes.
    reg_term = alpha * np.identity((X.T @ X).shape[0]) # Termo de regularização.
    coef = np.linalg.inv(X.T @ X + reg_term) @ X.T @ y

    # Retorna os coeficientes e o interceptor.
    return coef[1:], coef[0]

def Media_valores_observaveis(y_train):
    """Média dos Valores Observáveis:
    Método de predição onde o valor previsto para um novo
    exemplo é a média dos valores observados no conjunto
    de treinamento.
    """
    return np.mean(y_train)

"""
5. Como o modelo regularizado depende da definição do valor λ, é de
interesse encontrar aquele que tem o valor médio mínimo de EQM.
Discuta qual foi o valor encontrado.
"""
alpha = 0.1

"""
7. Os dados selecionados para teste, são utilizados para validar o modelo.
Assim, é necessário computar o Erro Quadrático Médio (EQM) e armazenar essas
medidas em uma lista/vetor que representa o EQM em cada uma das rodadas.
"""

def EQM(y_real, y_pred):
    """Erro Quadrático Médio."""
    return np.mean((y_real - y_pred) ** 2)

# Armazenará os EQM de cada modelo.
EQM_MQO_tradicional = []
EQM_MQO_regularizado = []
EQM_Media_valores_observaveis = []

for e in range(epochs):
    """
    6. Para validação de tais modelos, em cada rodada deve-se embaralhar
    as amostras do conjunto de dados e em seguida realizar o particionamento
    em 80% dos dados de treinamento e 20% para teste.
    """
    # Embaralha as amostras do conjunto de dados.
    seed = np.random.permutation(N)
    X_random = x[seed,:]
    y_random = y[seed,:]

    # Divide os dados de treino (80%).
    X_train = X_random[0:int(N * 0.8),:]
    y_train = y_random[0:int(N * 0.8),:]

    # Divide os dados de teste (20%).
    X_test = X_random[int(N * 0.8):,:]
    y_test = y_random[int(N * 0.8):,:]

    # ----------- MQO Tradicional -----------
    # Treina com o modelo "MQO traidiconal" com os dados de treino.
    coef, interceptor = MQO_tradicional(X_train, y_train)
    # Faz a previsão nos dados de teste.
    y_pred = interceptor + X_test @ coef
    # Calcula o EQM para esta época.
    EQM_MQO_tradicional.append(EQM(y_test, y_pred))
    # ---------------------------------------

    # ----------- MQO Regularizado -----------
    # Treina com o modelo "MQO regularizado" com os dados de treino.
    coef, interceptor = MQO_regularizado(X_train, y_train, alpha)
    # Faz a previsão nos dados de teste.
    y_pred = interceptor + X_test @ coef
    # Calcula o EQM para esta época.
    EQM_MQO_regularizado.append(EQM(y_test, y_pred))
    # ----------------------------------------

    # ----------- Média dos Valores Observáveis -----------
    # Treina com o modelo "Média dos valores observáveis" com os dados de treino.
    mean = Media_valores_observaveis(y_train)
    # Faz a previsão nos dados de teste.
    y_pred = np.full_like(y_test, mean) # Cria um novo array com memso tipo e forma de outro array (mean).
    # Calcula o EQM para esta época.
    EQM_Media_valores_observaveis.append(EQM(y_test, y_pred))
    # -----------------------------------------------------

"""
8. Ao final das 1000 rodadas calcula para cada modelo utilizado, compute a
média, desvio-padrão, valor maior, valor menor de cada EQM. Coloque esses
valores em um gráfico ou tabela e discuta os resultados obtidos.
"""
# MQO Tradicional: Média, Desvio Padrão, Máximo e Mínimo
mean_mqo_tradicional = np.mean(EQM_MQO_tradicional)
std_mqo_tradicional = np.std(EQM_MQO_tradicional)
max_mqo_tradicional = np.max(EQM_MQO_tradicional)
min_mqo_tradicional = np.min(EQM_MQO_tradicional)

# MQO Regularizado: Média, Desvio Padrão, Máximo e Mínimo
mean_mqo_regularizado = np.mean(EQM_MQO_regularizado)
std_mqo_regularizado = np.std(EQM_MQO_regularizado)
max_mqo_regularizado = np.max(EQM_MQO_regularizado)
min_mqo_regularizado = np.min(EQM_MQO_regularizado)

# Média dos Valores Observáveis: Média, Desvio Padrão, Máximo e Mínimo
mean_media_valores_observaveis = np.mean(EQM_Media_valores_observaveis)
std_media_valores_observaveis = np.std(EQM_Media_valores_observaveis)
max_media_valores_observaveis = np.max(EQM_Media_valores_observaveis)
min_media_valores_observaveis = np.min(EQM_Media_valores_observaveis)

# Rótulos
labels = ['MQO Tradicional', 'MQO Regularizado', 'Média Valores Observáveis']

# Métricas
metrics = ['Média', 'Desvio Padrão', 'Máximo', 'Mínimo']

# Valores correspondentes a cada métrica.
vals = [
    [mean_mqo_tradicional, std_mqo_tradicional, max_mqo_tradicional, min_mqo_tradicional],
    [mean_mqo_regularizado, std_mqo_regularizado, max_mqo_regularizado, min_mqo_regularizado],
    [mean_media_valores_observaveis, std_media_valores_observaveis, max_media_valores_observaveis, min_media_valores_observaveis]
]

# Qntd. de métricas.
num_metrics = len(metrics)

# Índices.
indexes = np.arange(num_metrics)

# Largura das barras.
width = 0.2

# Cria um gráfico para barras de erro.
fig, ax = plt.subplots(figsize=(12, 6))
for i, label in enumerate(labels):
    val = vals[i]
    ax.bar(indexes + i * width, val, width, label=label)

# Define os rótulos do eixo x.
ax.set_xticks(indexes + width)
ax.set_xticklabels(metrics)

# Habilita a legenda.
ax.legend()

# Adiciona os rótulos.
for i in range(len(labels)):
    for j in range(num_metrics):
        val = vals[i][j]
        ax.text(indexes[j] + i * width, val, f'{val:.2f}', ha='center', va='bottom')

# Define o título e rótulos dos eixos.
ax.set_title('Métricas de EQM para:\nMQO Tradicional, MQO Regularizado e Média dos Valores Observáveis.')
ax.set_xlabel('Métricas')
ax.set_ylabel('Valores')

# Mostra o gráfico.
plt.tight_layout()
plt.show()

