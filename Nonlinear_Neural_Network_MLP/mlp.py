import numpy as np

from typing import Tuple # Usado para notação de retorno das funções.


# ⊗ -> Produto externo
# ◦ -> Produto de Hadamard
# · -> Produto entre duas matrizes
class MLP:
    """Implementação de um Multilayer Perceptron."""

    def __init__(self, hidden_layers: int, hidden_neurons: list[int], output_layers: int, X: np.ndarray, Y: np.ndarray) -> None:
        """Construtor do MLP.

        Parameters
        ----------
        hidden_layers : int
            A quantidade L de camadas ocultas.
        hidden_neurons : list[int]
            A quantidade de neurônios em cada uma das L camadas ocultas.
        output_layers : int
            A quantidade de neurônios m na camada de saída.
        X : np.ndarray
            As variáveis independentes.
        Y : np.ndarray
            As variáveis dependentes.
        """
        # Qntd. de amostras (N) e Qntd. de variáveis (p)
        self.N, self.p = X.shape[::-1]
        # Qntd. de classes (c)
        self.c = Y.shape[0]

        # Estrutura do MLP.
        self.input_layers = self.p
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.output_layers = output_layers

        # Pesos (L + 1 camadas com q neurônios ocultos cada com valores aleatórios [-.5, .5]).
        self.initialize_weights()
        # Entradas (L + 1 camadas com q neurônios ocultos cada).
        # [[Entrada ], [Camadas Ocultas]]
        # [[p + 1, N], [q1, N], ..., [qL, N]]
        self.i = [np.zeros((self.p + 1, self.N))] + [np.zeros((q, self.N)) for q in self.hidden_neurons]
        # Saídas. (L + 1 camadas com q neurônios ocultos cada).
        # [[Camadas Ocultas    ], [Saída]]
        # [[q1, N], ..., [qL, N], [m, 1]]
        self.y = [np.zeros((q, self.N)) for q in self.hidden_neurons] + [np.zeros((self.output_layers, 1))]
        # Erro. (L + 1 camadas com q neurônios ocultos cada).
        # [[Camadas Ocultas    ], [Saída]]
        # [[q1, N], ..., [qL, N], [C, N]]
        self.delta = [np.zeros((q, self.N)) for q in self.hidden_neurons] + [np.zeros((self.output_layers, self.N))]

        # Adiciona um vetor linha -1 em X.
        self.X = np.vstack((-np.ones((1, self.N)), X)) # XeR^(p+1)xN
        self.Y = Y # YeR^cxN


    def logistic_sigmoid(self, u) -> float:
        """Função de ativação sigmóide logística."""
        return 1 / (1 + np.exp(-u))


    def logistic_sigmoid_derivative(self, u) -> float:
        """Função de ativação sigmóide logística derivada."""
        sig = self.logistic_sigmoid(u)
        return sig * (1 - sig)


    def split_train_test(self, train_percentage: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Realiza a divisão do conjunto de dados
        em conjunto de dados de treino e teste.

        Parameters
        ----------
        train_percentage : float, optional
            O percentual dos dados atribuído ao conjunto de treinamento,
            por padrão é 0.8

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Os conjuntos de treino (X e Y) e os conjuntos de teste (X e Y)
        """
        seed = np.random.permutation(self.N)
        train_indexes = seed[:int(self.N * train_percentage)]
        test_indexes = seed[int(self.N * train_percentage):]

        Xtrain = self.X[:, train_indexes]
        Ytrain = self.Y[:, train_indexes]

        Xtest = self.X[:, test_indexes]
        Ytest = self.Y[:, test_indexes]
        
        return Xtrain, Ytrain, Xtest, Ytest


    def initialize_weights(self) -> None:
        """Inicializa os pesos dos neurônios das camadas ocultas."""
        # W(1)eR^q1x(p+1) - Dimensão pesos da camada de entrada.
        # W(L)eR^qLx(qL-1 +1) - Dimensão pesos da camada oculta.
        # W(L+1)eR^mx(qL +1) - Dimensão pesos da camada de saída.
        # q1 x (p+1) .. qL x (qL-1 + 1) .. m x (qL + 1) 
        # Adiciona os pesos às camadas/neurônios. [-0.5, 0.5]
        self.W = [np.random.uniform(-0.5, 0.5,
                                    (self.hidden_neurons[i], (self.p)
                                     if i == 0 else (self.hidden_neurons[i - 1]))
                                     ) for i in range(self.hidden_layers)]
        # Adiciona os viés às camadas/neurônios. (-1)
        for l in range(self.hidden_layers):
            self.W[l] = np.hstack((self.W[l], -np.ones((self.hidden_neurons[l], 1))))

        # Adiciona os pesos às camadas/neurônios de saída.
        self.W.append(np.random.uniform(-0.5, 0.5, (self.output_layers, self.hidden_neurons[-1] + 1)))


    def EQM(self, Xtrain: np.ndarray, Ytrain: np.ndarray) -> float:
        """Calcula o Erro Quadrático Médio (EQM)

        Parameters
        ----------
        Xtrain : np.ndarray
            Os dados do conjunto de treino (X).
        Ytrain : np.ndarray
            Os dados do conjunto de treino (Y).

        Returns
        -------
        float
            O EQM calculado.
        """
        eqm = 0
        for t in range(Xtrain.shape[1]):
            x_t = Xtrain[:, t]
            self.forward(x_t)
            d_t = Ytrain[:, t]
            eqi = 0
            for j in range(self.output_layers):
                # EQI ← EQI + (d[j] − y[QTD_L − 1][j])²
                eqi += (d_t[j] - self.y[self.hidden_layers][j]) ** 2
            # EQM ← EQM + EQI
            eqm += eqi
        # EQM ← EQM/(2 ∗ QtdAmostrasTreino)
        return eqm / (2 * Xtrain.shape[1])


    # TODO: Adicionar termo do momento (alpha [0, 0.9])
    def fit(self, epochs: int, lr: float, min_eqm: float):
        """Realiza o treinamento do modelo.

        Parameters
        ----------
        epochs : int
            A quantidade máxima de épocas.
        lr : float
            A taxa de aprendizado.
        min_eqm : float
            O critério de parada em função do erro (EQM).
        """
        current_lr = lr
        current_eqm = 1
        current_epoch = 0
        while current_eqm > min_eqm and current_epoch < epochs:
            print(f'Epoch # {current_epoch} - lr: {current_lr} - ', end='')

            # Separa o conjunto de dados em treino e teste.
            Xtrain, Ytrain, Xtest, Ytest = self.split_train_test(train_percentage=0.9)

            for t in range(Xtrain.shape[1]):
                x_t = Xtrain[:, t]
                self.forward(x_t)
                d_t = Ytrain[:, t]
                self.backward(x_t, d_t, current_lr)
            
            # Calcula o EQM para esta época.
            current_eqm = self.EQM(Xtrain=Xtrain, Ytrain=Ytrain)

            # Taxa de aprendizagem variável. (Decaimento exponencial)
            current_lr = lr / (1 + current_epoch)

            # Testa a acurácia do modelo para esta época.
            self.eval(Xtest=Xtest, Ytest=Ytest)
            current_epoch += 1
        print('Treinamento concluído.')
        self.eval(Xtest=Xtest, Ytest=Ytest)


    def eval(self, Xtest: np.ndarray, Ytest: np.ndarray) -> None:
        """Realiza o teste do modelo.

        Parameters
        ----------
        Xtest : np.ndarray
            O conjunto de teste (X).
        Ytest : np.ndarray
            O conjunto de teste (Y).
        """
        corrects = 0
        for t in range(Xtest.shape[1]):
            x_t = Xtest[:, t]
            self.forward(x_t)
            predicted_class = np.argmax(self.y[-1])
            if predicted_class == np.argmax(Ytest[:, t]):
                corrects += 1
        print(f'Accuracy: {corrects * 100 / Ytest.shape[1]:.2f}%')


    def forward(self, x: np.ndarray) -> None:
        """Forward da rede MLP.
        Passagem de informações da camada de entrada
        à camada de saída, propagação direta.

        Parameters
        ----------
        x : np.ndarray
            Uma amostra xeR^(p+1)x1
        """
        for j, w in enumerate(self.W):
            if j == 0:
                # i[j] ← W[j] · x_amostra
                self.i[j] = np.dot(w, x)
                # y[j] ← g(i[j])
                self.y[j] = self.logistic_sigmoid(self.i[j])
            else:
                # y_bias ← y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = np.insert(self.y[j - 1], 0, -1, axis=0)
                # i[j] ← W[j] · y_bias
                self.i[j] = np.dot(w, y_bias)
                # y[j] ← g(i[j])
                self.y[j] = self.logistic_sigmoid(self.i[j])


    def backward(self, x: np.ndarray, d: np.ndarray, lr: float) -> None:
        """Backward da rede MLP.
        Passagem de informações da camada de saída
        às camadas ocultas, retropropagação.

        Parameters
        ----------
        x : np.ndarray
            Uma amostra xeR^(p+1)x1
        d : np.ndarray
            O rótulo da amostra deR^cx1
        lr : float
            Taxa de aprendizado.
        """
        j = len(self.W) - 1
        while j >= 0:
            if j + 1 == len(self.W):
                # δ[j] ← g′ (i[j]) ◦ (d − y[j])
                self.delta[j] = self.logistic_sigmoid_derivative(self.i[j]) * (d - self.y[j])
                # y_bias ← y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = np.insert(self.y[j - 1], 0, -1, axis=0)
                # W[j] ← W[j] + η(δ[j] ⊗ y_bias)
                self.W[j] += lr * np.outer(self.delta[j], y_bias)
            elif j == 0:
                # Wb[j + 1] ← W[j + 1] transposto sem o víes/bias (-1).
                wb = self.W[j + 1][:, :-1].T
                # δ[j] ← g′(i[j]) ◦ (Wb[j + 1] · δ[j + 1])
                self.delta[j] = self.logistic_sigmoid_derivative(self.i[j]) * np.dot(wb, self.delta[j + 1])
                # W[j] ← W[j] + η(δ[j] ⊗ y_bias)
                self.W[j] += lr * np.outer(self.delta[j], x)
            else:
                # Wb[j + 1] ← W[j + 1] transposto sem o víes/bias (-1).
                wb = self.W[j + 1][:, :-1].T
                # δ[j] ← g′(i[j]) ◦ (Wb[j + 1] · δ[j + 1])
                self.delta[j] = self.logistic_sigmoid_derivative(self.i[j]) * np.dot(wb, self.delta[j + 1])
                # y_bias ← y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = np.insert(self.y[j - 1], 0, -1, axis=0)
                # W[j] ← W[j] + η(δ[j] ⊗ y_bias)
                self.W[j] += lr * np.outer(self.delta[j], y_bias)
            j -= 1
