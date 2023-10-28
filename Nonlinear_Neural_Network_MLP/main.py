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

# Variável independente (X).
X = np.empty((IMAGE_DIMENSION[0] * IMAGE_DIMENSION[1], 0))
# Variável dependente (Y).
Y = np.empty((len(dataset_inner_folders), 0))

#
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
# O uso do np.min e np.max é para garantir [0, 1] inclusivo.
# (arr - min) / (max - min)
X = (X - np.min(X)) / (np.max(X) - np.min(X))


"""
3. Para utilização dos modelos implementados, faça uma
organização no conjunto de dados para que se tenha a nova
dimensão, XeR^(p+1)xN
"""
# X já está no formato.


"""
4. Para ambos os modelos, faça a definição do η (passo
de aprendizagem) e precisão, conforme as discussões
realizadas em sala e escrita nos slides.
"""
lr = 0.001
epochs = 10


"""
5. Para o modelo MLP, faça uma discussão inicial sobre
overfitting e underfitting. Assim, realize o projeto de
uma rede com poucos neurônios na cadama oculta e produza
resultados de acurácia e Matriz de Confusão. Em seguida,
aumente a quantidade de neurônios e/ou camadas escondidas
até que seja identificado o overfitting. Expresse os
resultados em duas matrizes de confusão.
"""
mlp = MLP(hidden_layers=3,
          hidden_neurons=[2, 4, 8],
          output_layers=c,
          X=X,
          Y=Y,)


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


"""
11. Projete uma rede MLP que seja superdimensionada, contudo,
identifique quando o overfitting acontecer. Nesse caso, a parada
antecipada deve ser operada. Faça discussões, se essa arquitetura
obteve resultados melhores ou piores com relação às topologias
anteriores e a rede RBF.
"""