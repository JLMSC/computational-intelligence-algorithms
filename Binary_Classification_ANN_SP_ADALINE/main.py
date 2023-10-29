import numpy as np
import matplotlib.pyplot as plt


# Carrega o dataset 'DataAV2.csv'
df = np.loadtxt('Binary_Classification_ANN_SP_ADALINE/DataAV2.csv', delimiter=',')

# Qntd. de Amostras (N) e Qntd. de Variáveis (p)
N, p = df.shape

# Separa as variáveis dependentes e independentes.
# Duas colunas do .csv são variáveis independentes.
X = df[:, 0:2].reshape(N, 2) # XeR^Nx(p+1) 
# A última coluna é variável dependente.
Y = df[:, 2].reshape(N, 1) # YeR^Nx1


"""
1. Faça a implementação dos modelos Perceptron Simples
(PS) e Adaline utilizando os critérios descritos no
final desta seção.
A implementação dos modelos PS e ADALINE deve ser
realizada com base no pseudocódigo e implementações
disponibilizadas em sala de aula.
"""
from adaline import ADALINE
from percepton_simples import PerceptronSimples


"""
2. Faça uma visualização inicial dos dados através do
gráfico de espalhamento. Nessa etapa, faça uma discussão
inicial sobre quais resultados poderão ser obtidos ao
utilizar o perceptron simples e o ADALINE.
"""
# fig = plt.figure(figsize=(10, 6))
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
# plt.xlabel('Variável Independente: X1')
# plt.ylabel('Variável Independente: X2')
# plt.title('Gráfico de Dispersão\nClassificação Binária')
# cbar = plt.colorbar(label='Variável Dependente: Y')
# cbar.set_ticks([Y.min(), Y.max()])
# plt.show()


"""
3. Para utilização dos modelos implementados, faça uma
organização no conjunto de dados para que se tenha a
nova dimensão, XeR^(p+1)xN.
"""
# Transpor X para o shape (p + 1, N) ao invés de (N, p + 1)
X = X.T


"""
4. Faça a definição do η (passo de aprendizagem), conforme
as discussões realizadas em sala e escrita nos slides.
"""
lr = 0.001


"""
5. Para o modelo ADALINE realize a definição do valor de
precisão.
"""
pr = 0.001


"""
6. Inicia-se a etapa de validação dos modelos, assim deve-se
realizar o processo de treinamento e teste, em 100 rodadas.
Assim, para cada rodada, é necessário realizar a divisão do
conjunto de dados para que se tenha 80% da informação em um
conjunto de treinamento, e 20% da informaçõa para teste.
Considere nesta divisão, que existam variáveis dedicadas para
os rótulos de treinamento e teste.
"""
epochs = 100

perceptron = PerceptronSimples(X, Y)
# O teste também é realizado durante cada época na fase
# de treinamento.
# ou, se desejar, perceptron.eval(Xtest, Ytest) ...
print('Iniciando treinamento com PERCEPTRON SIMPLES')
perceptron.fit(epochs=epochs, lr=lr)


adaline = ADALINE(X, Y)
# O teste também é realizado durante cada época na fase
# de treinamento.
# ou, se desejar, adaline.eval(Xtest, Ytest) ...
print('Iniciando treinamento com ADALINE')
adaline.fit(epochs=epochs, lr=lr, pr=pr)



"""
7. Ao final das rodadas, compute os seguintes resultados para
o PS e ADALINE:
    a) Acurácia Média, com seu desvio padrão, maior e menor
        valor.
    b) Sensibilidade Média, com seu desvio padrão, maior e
        menor valor.
    c) Especificidade Média, com seu desvio padrão, maior e
        menor valor.
    d) Construa uma matriz de confusão (gráfico) para a rodada
        em que se teve a melhor acurácia.
    e) Construa uma matriz de confusão (gráfico) para a rodada
        em que se teve a pior acurácia.
    f) Para esses dois casos, construa também um gráfico que 
        mostre o hiperplano de separação dos dois modelos.
"""
def compute_results(arr: np.ndarray | list):
    print(f'Média: {np.mean(arr):.4f}')
    print(f'Desvio Padrão: {np.std(arr):.4f}')
    print(f'Maior valor: {np.max(arr):.4f}')
    print(f'Menor valor: {np.min(arr):.4f}\n')

# Acurácia = VP + VN / VP + VN + FP + FN
# Sensibilidade = VP / VP + FN
# Especificidade = VN / VN + FP

# Perceptron Simples
print('Resultados PERCEPTRON SIMPLES')
print('Acurácia:')
compute_results(perceptron.accuracies)
print('Resultados PERCEPTRON SIMPLES')
print('Sensibilidade:')
compute_results(perceptron.sensitivities)
print('Resultados PERCEPTRON SIMPLES')
print('Especificidade:')
compute_results(perceptron.specificities)

print('=' * 50, end='\n')

# ADALINE
print('Resultados ADALINE SIMPLES')
print('Acurácia:')
compute_results(adaline.accuracies)
print('Resultados ADALINE SIMPLES')
print('Sensibilidade:')
compute_results(adaline.sensitivities)
print('Resultados ADALINE SIMPLES')
print('Especificidade:')
compute_results(adaline.specificities)



# Gráfico da Acurácia, Sensibilidade e Especificade por Época.
# ? Trocar adaline por perceptron e vice-versa, depende de qual vai usar.
# bar_width = 0.2
# x = range(epochs)
# fig, ax = plt.subplots()
# ax.plot([i - bar_width for i in x], adaline.accuracies, label='Acurácias', color='green')
# ax.plot(x, adaline.sensitivities, label='Sensibilidades', color='blue')
# ax.plot([i + bar_width for i in x], adaline.specificities, label='Especificidades', color='orange')

# # ax.set_xticks(x)
# # ax.set_xticklabels(x, rotation=45)

# ax.legend(loc='lower left')

# ax.set_xlabel('Épocas')
# ax.set_ylabel('Métricas')
# ax.set_title('Acurácia, Sensibilidade e Especificidade por Época do ADALINE')

# plt.grid()
# plt.show()



# Gráfico de Barra Média Acurácia, Sensibilidade e Especificidade.
# def get_metrics(arr):
#     return np.mean(arr), np.std(arr), np.max(arr), np.min(arr)

# labels = ['Acurácia', 'Sensibilidade', 'Especificidade']
# vals = [
#     # ? Trocar adaline por perceptron e vice-versa, depende de qual vai usar.
#     [*get_metrics(adaline.accuracies)],
#     [*get_metrics(adaline.sensitivities)],
#     [*get_metrics(adaline.specificities)],
# ]

# metrics = ['Média', 'Desvio Padrão', 'Máximo', 'Mínimo']
# num_metrics = len(metrics)
# num_categories = len(vals)
# indexes = np.arange(num_metrics)
# bar_width = 0.2

# category_colors = ['blue', 'green', 'red', 'orange']

# fig, ax = plt.subplots(figsize=(12, 6))

# for i, label in enumerate(labels):
#     for j in range(num_categories):
#         val = vals[j]
#         ax.bar(indexes + j * bar_width, val, bar_width, color=category_colors[j])

# ax.set_xticks(indexes + bar_width * (num_categories - 1) / 2)
# ax.set_xticklabels(metrics)

# legend_labels = [f'{label}' for label in labels]
# legend = ax.legend(legend_labels, loc='lower left')

# for i in range(len(labels)):
#     for j in range(num_metrics):
#         val = vals[i][j]
#         ax.text(indexes[j] + i * bar_width, val, f'{val:.2f}%', ha='center', va='bottom')

# # ? Trocar adaline por perceptron e vice-versa, depende de qual vai usar.
# ax.set_title(f'Acurácia, Sensibilidade e Especificidade do ADALINE')
# ax.set_xlabel('Métricas')
# ax.set_ylabel('Valores')

# plt.tight_layout()
# plt.show()



# Matriz de confusão
# ? Trocar adaline por perceptron e vice-versa, depende de qual vai usar.
# ? Trocar as linhas abaixo conforme a necessidade
# acc_i = np.argmax(adaline.accuracies) # Melhor acurácia
# acc_i = np.argmin(adaline.accuracies) # Pior acurácia
# confustion_matrix = np.array([
#     [adaline.VPS[acc_i], adaline.FPS[acc_i]],
#     [adaline.FNS[acc_i], adaline.VNS[acc_i]]
# ])
# classes = ['Verdadeiro', 'Falso']
# plt.figure(figsize=(10, 6))

# cmap = plt.get_cmap('Oranges')

# normalized_matrix = confustion_matrix.astype('float') / confustion_matrix.sum(axis=1)[:, np.newaxis]

# plt.imshow(normalized_matrix, interpolation='nearest', cmap=cmap)
# plt.title('Matriz de Confusão (Rodada Pior Acurácia)\nADALINE')
# plt.colorbar()

# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=45)
# plt.yticks(tick_marks, classes)

# thresh = normalized_matrix.max() / 2.0
# for i in range(len(classes)):
#     for j in range(len(classes)):
#         plt.text(j, i, f'{normalized_matrix[i, j]*100:.2f}%', ha='center', va='center', color='white' if normalized_matrix[i, j] > thresh else 'black')

# plt.tight_layout()
# plt.ylabel('Valor Real')
# plt.xlabel('Valor Predito')

# plt.show()



# Hiperplano de separação.
w1 = adaline.W[0]
w2 = adaline.W[1]

x = np.linspace(-15, 10, 10000)
y = (-w1 * x) / w2

plt.figure(figsize=(8, 6))
plt.plot(x, y, '-', color='orange')

plt.xlim(np.min(X) - 5, np.max(X) + 5)
plt.ylim(np.min(X) - 5, np.max(X) + 5)

plt.title('Hiperplano de Separação do ADALINE')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

class_1_indexes = (Y == 1).flatten()
class_2_indexes = (Y == -1).flatten()
X_1 = X[:, class_1_indexes]
X_2 = X[:, class_2_indexes]

plt.scatter(X_1[0, :], X_1[1, :], c='red', marker='o', label='Classe 1')
plt.scatter(X_2[0, :], X_2[1, :], c='blue', marker='x', label='Classe 2')

plt.legend(loc='upper left')

plt.show()



"""
8. Com os resultados obtidos, faça discussões!
"""