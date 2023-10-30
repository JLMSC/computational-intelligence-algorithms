import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dimensão da imagem.
IMAGE_DIMENSION = (20, 20)

# Carrega o dataset em variáveis independentes (X) e variáveis dependentes (Y).
# Caminho do dataset.
dataset_path = 'Nonlinear_Neural_Network_MLP/faces/'
# Nome dos diretórios dentro do dataset.
dataset_inner_folders = [
    folder_name
    for folder_name in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder_name))
]
# Diretório de cada imagem do dataset. 
dataset_inner_folders_images_path = [
    os.path.join(dataset_path, folder_name, image_name)
    for folder_name in dataset_inner_folders
    for image_name in os.listdir(os.path.join(dataset_path, folder_name))
    if os.path.join(dataset_path, folder_name, image_name).endswith('.pgm')
]
# Remove duplicatas e aleatoriza a ordem do caminho das imagens.
dataset_inner_folders_images_path = np.random.permutation(list(set(dataset_inner_folders_images_path)))

# Variável independente (X).
X = np.empty((IMAGE_DIMENSION[0] * IMAGE_DIMENSION[1], 0))
# Variável dependente (Y).
Y = np.empty((len(dataset_inner_folders), 0))

# Cria o conjunto de dados em arrays.
for path in set(dataset_inner_folders_images_path):
    pgm_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_pgm_image = cv2.resize(pgm_image, IMAGE_DIMENSION)

    normalized_vector = resized_pgm_image.flatten('F')
    target = -np.ones((len(dataset_inner_folders), 1))
    target[dataset_inner_folders.index(path.split('/')[-2]), 0] = 1

    normalized_vector.shape = (len(normalized_vector), 1)

    X = np.append(X, normalized_vector, axis=1)
    Y = np.append(Y, target, axis=1)


print(f'Quantidade de amostras do conjunto de dados: {X.shape[1]}')
print('A quantidade de preditores esta relacionada ao redimensionamento!')
print(f'Para esta rodada escolheu-se um redimensionamento de {IMAGE_DIMENSION}')
print(f'Portanto, a quantidade de preditores desse conjunto de dados: {X.shape[0]}')
print(f'Este conjunto de dados possui {Y.shape[0]} classes')
print(f'X tem ordem {X.shape[0]}x{X.shape[1]}')
print(f'Y tem ordem {Y.shape[0]}x{Y.shape[1]}')

# Qntd. de Amostras (N) e Qntd. de Variáveis (p)
N, p = X.shape[::-1]
# Qntd. de classes (c)
c = Y.shape[0]


"""
1. Realize a implementação dos modelos MLP e rede RBF
com base nas discussões realizadas em sala de aula
considerando as observações no final do presente documento.
"""
from mlp import MLP


"""
2. Com base na magnitude dos rótulos presentes no conjunto
de dados, faça a normalização dos dados utilizando o método
min-max.
"""
# Caso seja usado a função sigmoid.
# X = (X - np.min(X)) / (np.max(X) - np.min(X))
# Caso seja usado a tangente hiperbólica.
X = 2 * ((X - np.min(X)) / (np.max(X) - np.min(X))) - 1


"""
3. Para utilização dos modelos implementados, faça uma
organização no conjunto de dados para que se tenha a nova
dimensão, XeR^(p+1)xN
"""
# X é alterado dentro da classe MLP.


"""
4. Para ambos os modelos, faça a definição do η (passo
de aprendizagem) e precisão, conforme as discussões
realizadas em sala e escrita nos slides.
"""
# Época 41 -> 42
# lr >= 0.002 melhorou em 60% a acurácia do modelo
# X = 2 * ((X - np.min(X)) / (np.max(X) - np.min(X))) - 1
# Função de Ativação: Tangente Hiperbólica
# Variância da Taxa de Aprendizado: Decaimento Exponencial
lr = 0.1
# epochs = 250
epochs = 100


"""
5. Para o modelo MLP, faça uma discussão inicial sobre
overfitting e underfitting. Assim, realize o projeto de
uma rede com poucos neurônios na cadama oculta e produza
resultados de acurácia e Matriz de Confusão. Em seguida,
aumente a quantidade de neurônios e/ou camadas escondidas
até que seja identificado o overfitting. Expresse os
resultados em duas matrizes de confusão.
"""
# Regra do valor médio: q = (p + m) / 2
# Regra da raiz quadrada: q = sqrt(p * m)
# Regra de Kolmogorov: q = 2p + 1
mlp = MLP(hidden_layers=3,
        #   hidden_neurons=[2 * p + 1, int(np.sqrt(p * c)), (p + c) // 2],
          hidden_neurons=[188, 188, 188],
          output_layers=c,
          X=X,
          Y=Y,)
mlp.fit(epochs=epochs, lr=lr, criterion=0.1, patience=5)


"""
6. Após esta análise, faça a validação do modelo escolhendo
uma topologia da rede MLP que não produza underfitting bem
como overfitting. Como este processo pode ser custoso, faça
a definição da topologia com base nas regras discutidas em
sala de aula.
"""


"""
( OPCIONAL )
7. Para o modelo RBF, escolha também uma quantidade de funções
de base, considerando as discussões realizadas em sala de aula.
"""


"""
8. Para compor os resultados desta etapa, sua equipe deve realizar
a construção de matrizes de confusão, box-plots e tabelas que
expressem a acurácia média, desvio padrão, maior e menor valor.
"""


"""
9. Como o tempo de treinamento pode ser custoso, faça com que se
tenham poucas rodadas de validação dos modelos.
"""


"""
10. Exiba nos resultados a quantidade média de épocas que fazem
com que os modelos atinjam a convergência.
"""
# Curva de Aprendizado.
plt.figure(figsize=(10, 6))
plt.plot(range(1, mlp.current_epoch + 1), mlp.train_errors, 'b', label='EQM Treino por Época')
plt.plot(range(1, mlp.current_epoch + 1), mlp.validation_errors, 'r', label='EQM Validação por Época')
plt.title('Curva de Aprendizado (EQMs)')
plt.xlabel('Épocas')
plt.ylabel('EQMs')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, mlp.current_epoch + 1), mlp.accuracies_per_epoch, 'g', label='Acurácia por Época')
plt.title('Curva de Aprendizado (Acurácia)')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid()
plt.show()



# Matriz de confusão
fig = plt.figure(figsize=(10, 6))
plt.imshow(mlp.confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Oranges'))

classes = [dataset_inner_folders[i] for i in range(c)]
plt.xticks(np.arange(c), classes, rotation=45)
plt.yticks(np.arange(c), classes, rotation=45)

for i in range(c):
    for j in range(c):
        plt.text(i, j, str(int(mlp.confusion_matrix[j, i])), ha='center', va='center')

plt.xlabel('Valor Predito')
plt.ylabel('Valor Real')

plt.title('Matriz de Confusão MLP (overfitting)')

plt.colorbar()
plt.show()


"""
11. Projete uma rede MLP que seja superdimensionada, contudo,
identifique quando o overfitting acontecer. Nesse caso, a parada
antecipada deve ser operada. Faça discussões, se essa arquitetura
obteve resultados melhores ou piores com relação às topologias
anteriores e a rede RBF.
"""