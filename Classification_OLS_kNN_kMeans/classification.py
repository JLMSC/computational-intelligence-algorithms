import statistics
import numpy as np
import matplotlib.pyplot as plt

# Carrega o dataset 'EMG.csv'.
df = np.loadtxt('EMG.csv', delimiter=',')

# Qntd. de Amostras (N) e Qntd. de variáveis (p)
N, p = df.shape

# Cada 10.000 amostras pertence a uma classe.
# As classes são: NEUTRO, SORRINDO, ABERTO, SURPRESO, RABUGENTO.
NEUTRO = np.tile(np.array([[1, 0, 0, 0, 0]]), (1000, 1))
SORRINDO = np.tile(np.array([[0, 1, 0, 0, 0]]), (1000, 1))
ABERTO = np.tile(np.array([[0, 0, 1, 0, 0]]), (1000, 1))
SURPRESO = np.tile(np.array([[0, 0, 0, 1, 0]]), (1000, 1))
RABUGENTO = np.tile(np.array([[0, 0, 0, 0, 1]]), (1000, 1))

# Separa as variáveis dependentes e independentes.
x = df[:,:].reshape(N, p) # R^Nxp (+ interceptor)
y = np.tile(np.concatenate((NEUTRO, SORRINDO, ABERTO, SURPRESO, RABUGENTO)), (10, 1)) # R^Nxc (c é a quantidade de classes.)

"""
1. Faça uma visualização inicial dos dados através do gráfico de espalhamento.
Nessa etapa levante hipóteses sobre quais serão as características de um modelo
que consegue separar as classes do problema.
"""

# Cores de cada classe.
colors = ['gray', 'green', 'blue', 'yellow', 'red']
classes = ['Neutro', 'Sorrindo', 'Aberto', 'Surpreso', 'Rabugento']
k = 0
for i in range(10):
    for j in range(5):
        plt.scatter(df[k:k + 1000, 0], df[k:k+1000, 1],
                    color=colors[j], edgecolors='k', label=classes[j])
        k += 1000
plt.legend(labels=classes)
plt.show()

"""
2. Para validar os modelos utilizados na tarefa de classificação, é necessário
definir uma quantidade específica de rodadas de treinamento e teste dos modelos.
Assim, defina essa quantidade de rodadas com o valor 100.
"""

# Quantidade de épocas.
epochs = 100

"""
3. Os modelos a serem implementados nessa etapa serão:
    MQO tradicional,
    MQO regularizado (Tikhonov),
    Classificador k-Vizinhos mais Próximos (k-NN),
    Distância Mínima ao Centróide (DMC).
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

def kNN(X_train, y_train, X_test, k):
    """Classificador k-Vizinhos mais Próximos (k-NN):
    Os pontos de dados que estão próximos no espaço de
    características tendem a ter rótulos semelhantes ou
    valores alvos semelhantes.
    """
    # Calcula as distância entre todos os pontos dos dados de teste
    # e todos os pontos dos dados de treino.
    distances = np.sqrt(np.sum((X_test[:, np.newaxis] - X_train) ** 2, axis=2))
    # Ordena as distância calculadas em ordem crescente para cada ponto de teste.
    neighbors_index = np.argsort(distances, axis=1)[:, :k]
    # Obtém as classes correspondentes (de y_train) dos vizinhos mais próximos.
    neighbors_classes = y_train[neighbors_index]
    # Calcula a classe mais frequente para cada ponto de teste com base nas classes
    # dos vizinhos mais próximos.
    y_pred = np.argmax(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=neighbors_classes), axis=1)
    
    # Retorna as previsões do modelo.
    return y_pred

def DMC(X_train, y_train, X_test):
    """Distância Mínima ao Centróide (DMC):
    Técnica baseada na ideia de que, se os dados forem
    projetados em um espaço de características diferentes,
    onde as classes sejam mais separáveis, a classificação
    se tornará mais simples.
    """

    # Calcula os centróides para cada classe nos dados de treinamento.
    centroids = np.zeros((y_train.shape[1], X_train.shape[1]))
    for class_ in range(y_train.shape[1]):
        # Filtra os pontos de treinamento que pertencem a essa classe
        # e calcula a média.
        centroids[class_] = np.mean(X_train[y_train[:, class_] == 1], axis=0)

    # Cria um array para armazenar as previsões do modelo.
    y_pred = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        # Calcula a distância entre o ponto de teste atual e todos
        # os centróides.
        distances = [np.sqrt(np.sum((X_test[i] - centroid) ** 2))
                     for centroid in centroids]
        # Encontra a classe com o centróide mais próximo usando a
        # menor distância.
        most_frequent_class = np.argmin(distances)
        # Atribui a classe predominante como previsão para o ponto
        # de teste atual.
        y_pred[i] = most_frequent_class

    # Retorna as previsões do modelo.
    return y_pred

"""
4. Como os modelos de regularização necessitam da definição de seus hiperparâmetros,
é de interesse encontrar aquele que tem o valor médio maior de Acurácia. Discuta qual
foi o valor encontrado para OLS (Tikhonov) e k elementos para o k-NN.
"""
alpha = 0.1
k = 2

"""
6. Os dados selecionados para teste, são utilizados para validar o modelo.
Assim é necessário computar a acurácia de cada modelo e armazenar essa
medida em uma lista/vetor que representa a acurácia em cada uma das rodadas.
"""

def accuracy(y_pred, y_test):
    """Acurácia."""
    return np.sum(y_pred == np.argmax(y_test, axis=1)) / len(y_test)

# Armazenará as Acurácias de cada modelo.
acc_MQO_tradicional = []
acc_MQO_regularizado = []
acc_kNN = []
acc_DMC = []

for e in range(epochs):
    """
    5. Para validação de tais modelos, em cada rodada deve-se embaralhar as amostras do
    conjunto de dados e em seguida relaizar o particionamento em 80% dos dados para
    treinamento e 20% para teste.
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
    # Treina com o modelo "MQO tradicional" com os dados de treino.
    coef, interceptor = MQO_tradicional(X_train, y_train)
    # Faz a previsão nos dados de teste.
    # (converte para o índice das classes baseado no maior valor do array).
    y_pred = np.argmax(X_test @ coef, axis=1)
    # Calcula a acurácia para esta época.
    acc_MQO_tradicional.append(accuracy(y_pred, y_test))
    # ---------------------------------------

    # ----------- MQO Regularizado -----------
    # Treina com o modelo "MQO regularizado" com os dados de treino.
    coef, interceptor = MQO_regularizado(X_train, y_train, alpha)
    # Faz a previsão nos dados de teste.
    # (converte para o índice das classes baseado no maior valor do array).
    y_pred = np.argmax(X_test @ coef, axis=1)
    # Calcula a acurácia para esta época.
    acc_MQO_regularizado.append(accuracy(y_pred, y_test))
    # ----------------------------------------

    # ---------------------- k-NN ----------------------
    # Treina com o modelo "k-NN" com os dados de treino.
    # Obtém a previsão nos dados de teste.
    # (converte para o índice das classes baseado no maior valor do array).
    y_pred = kNN(X_train, y_train, X_test, k)
    # Calcula a acurácia para esta época.
    acc_kNN.append(accuracy(y_pred, y_test))
    # --------------------------------------------------

    # ---------------------- DMC ----------------------
    # Treina com o modelo "DMC" com os dados de treino.
    # Obtém a previsão nos dados de teste.
    # (converte para o índice das classes baseado no maior valor do array).
    y_pred = DMC(X_train, y_train, X_test)
    # Calcula a acurácia para esta época.
    acc_DMC.append(accuracy(y_pred, y_test))
    # -------------------------------------------------

"""
7. Ao final das 100 rodadas calcula pra cada modelo utilizado, compute a média, desvio-padrão,
valor maior, valor menor e moda de cada acurácia. Coloque esses valores em um gráfico ou tabela
e discuta os resultados obtidos.
"""
# MQO Tradicional: Média, Desvio Padrão, Máximo, Mínimo e Moda.
mean_mqo_tradicional = np.mean(acc_MQO_tradicional)
std_mqo_tradicional = np.std(acc_MQO_tradicional)
max_mqo_tradicional = np.max(acc_MQO_tradicional)
min_mqo_tradicional = np.min(acc_MQO_tradicional)
mode_mqo_tradicional = statistics.mode(acc_MQO_tradicional)

# MQO Regularizado: Média, Desvio Padrão, Máximo, Mínimo e Moda.
mean_mqo_regularizado = np.mean(acc_MQO_regularizado)
std_mqo_regularizado = np.std(acc_MQO_regularizado)
max_mqo_regularizado = np.max(acc_MQO_regularizado)
min_mqo_regularizado = np.min(acc_MQO_regularizado)
mode_mqo_regularizado = statistics.mode(acc_MQO_regularizado)

# k-NN: Média, Desvio Padrão, Máximo, Mínimo e Moda.
mean_kNN = np.mean(acc_kNN)
std_kNN = np.std(acc_kNN)
max_kNN = np.max(acc_kNN)
min_kNN = np.min(acc_kNN)
mode_kNN = statistics.mode(acc_kNN)

# DMC: Média, Desvio Padrão, Máximo, Mínimo e Moda.
mean_DMC = np.mean(acc_DMC)
std_DMC = np.std(acc_DMC)
max_DMC = np.max(acc_DMC)
min_DMC = np.min(acc_DMC)
mode_DMC = statistics.mode(acc_DMC)

# Rótulos
labels = ['MQO Tradicional', 'MQO Regularizado', 'k-NN', 'Distância Mínima ao Centróide']

# Métricas
metrics = ['Média', 'Desvio Padrão', 'Máximo', 'Mínimo', 'Moda']

# Valores correspondentes a cada métrica.
vals = [
    [mean_mqo_tradicional, std_mqo_tradicional, max_mqo_tradicional, min_mqo_tradicional, mode_mqo_tradicional],
    [mean_mqo_regularizado, std_mqo_regularizado, max_mqo_regularizado, min_mqo_regularizado, mode_mqo_regularizado],
    [mean_kNN, std_kNN, max_kNN, min_kNN, mode_kNN],
    [mean_DMC, std_DMC, max_DMC, min_DMC, mode_DMC]
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
ax.set_title('Métricas de Acurácia para:\nMQO Tradicional, MQO Regularizado, k-NN e DMC.')
ax.set_xlabel('Métricas')
ax.set_ylabel('Valores')

# Mostra o gráfico.
plt.tight_layout()
plt.show()