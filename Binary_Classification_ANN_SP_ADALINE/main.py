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
fig = plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
plt.xlabel('Variável Independente: X1')
plt.ylabel('Variável Independente: X2')
plt.title('Gráfico de Dispersão\nClassificação Binária')
cbar = plt.colorbar(label='Variável Dependente: Y')
cbar.set_ticks([Y.min(), Y.max()])
plt.show()


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
lr = 0.001 # TODO: Testar com valores diferentes.


"""
5. Para o modelo ADALINE realize a definição do valor de
precisão.
"""
pr = 0.001 # TODO: Testar com valores diferentes.


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

# TODO:
# (d) Construa uma matriz de confusão (gráfico) para a rodada em que se teve a melhor acurácia.
# (e) Construa uma matriz de confusão (gráfico) para a rodada em que se teve a pior acurácia.
# (f) Para esses dois casos, construa também um gráfico que mostre o hiperplano de separação dos
#     dois modelos.


"""
8. Com os resultados obtidos, faça discussões!
"""
pass
# TODO: Plotar grafos!
# TODO: plot() com as acurácias, sensibilidades e especificidades de cada modelo em cada época.
# TODO: bar() comparando media, desvio padrão, min e max de cada modelo (acc, sens, espc)
# TODO: Comparativo com LR e PR diferentes.