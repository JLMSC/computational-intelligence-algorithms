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
        # Acurácia no conjunto de teste do modelo no treinamento por época.
        self.accuracies_per_epoch = []
        # EQMs do modelo no treinamento por época.
        self.eqms_per_epoch = []

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
        # Camadas Ocultas       , Saída
        # [[q1, 1], ..., [qL, 1], [m, 1]]
        self.i = [np.zeros((q, 1)) for q in self.hidden_neurons] + [np.zeros((self.c, 1))]
        # Saídas (L + 1 camadas com q neurônios ocultos cada).
        # Camadas Ocultas       , Saída
        # [[q1, 1], ..., [qL, 1], [m, 1]]
        self.y = [np.zeros((q, 1)) for q in self.hidden_neurons] + [np.zeros((self.c, 1))]
        # Erros. (L + 1 camadas com q neurônios ocultos cada).
        # Camadas Ocultas       , Saída
        # [[q1, 1], ..., [qL, 1], [m, 1]]
        self.delta = [np.zeros((q, 1)) for q in self.hidden_neurons] + [np.zeros((self.c, 1))]

        # Adiciona um vetor linha -1 em X.
        self.X = np.vstack((-np.ones((1, self.N)), X)) # XeR^(p+1)xN
        self.Y = Y # YeR^cxN


    def activate(self, u) -> float:
        """Função de ativação."""
        return (1 - np.exp(-u)) / (1 + np.exp(-u)) # Tangente Hiperbólica
        # return 1 / (1 + np.exp(-u)) # Sigmóide Logística


    def activate_derivative(self, u) -> float:
        """Função de ativação derivada."""
        activate = self.activate(u)
        return 0.8 * (1 - (activate ** 2)) # Tangente Hiperbólica Derivada.
        # return activate * (1 - activate) # Sigmóide Logística Derivada


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
        # W(1)eR^q1x(p + 1) - Dimensão pesos da camada de entrada.
        # W(L-1)eR^qL-1x(q1 + 1) // W(L)eR^qLx(qL-1 + 1) - Dimensão pesos das camadas ocultas.
        # W(L+1)eR^mx(qL + 1) - Dimensão pesos da camada de saída.
        # q1x(p + 1) .. qL-1x(q1 + 1) .. mx(qL + 1) 
        self.W = [
            # Entrada.
            *[np.random.uniform(-0.5, 0.5, (self.hidden_neurons[0], self.p))],
            # Camadas Ocultas.
            *[np.random.uniform(-0.5, 0.5, (self.hidden_neurons[i + 1], self.hidden_neurons[i]))
              for i in range(self.hidden_layers - 1)],
            # Saída.
            *[np.random.uniform(-0.5, 0.5, (self.c, self.hidden_neurons[-1]))],
        ]
        # Adiciona os viés (-1) em todas as camadas.
        for i, w in enumerate(self.W):
            # self.W[i] = np.hstack((w, -np.ones((w.shape[0], 1))))
            self.W[i] = np.hstack((-np.ones((w.shape[0], 1)), w))


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
            x_t = Xtrain[:, t].reshape((Xtrain.shape[0], 1))
            self.forward(x_t)
            d_t = Ytrain[:, t].reshape((Ytrain.shape[0], 1))
            eqi = 0
            for j in range(self.output_layers):
                # EQI ← EQI + (d[j] − y[QTD_L − 1][j])²
                eqi = eqi + (d_t[j] - self.y[self.hidden_layers][j]) ** 2
            # EQM ← EQM + EQI
            eqm = eqm + eqi
        # EQM ← EQM/(2 ∗ QtdAmostrasTreino)
        return (eqm / (2 * Xtrain.shape[1]))[0]


    def fit(self, epochs: int, lr: float, criterion: float, momentum: float):
        """Realiza o treinamento do modelo.

        Parameters
        ----------
        epochs : int
            A quantidade máxima de épocas.
        lr : float
            A taxa de aprendizado.
        criterion : float
            O critério de parada em função do erro (EQM).
        momentum : float
            O termo do momento.
        """
        current_lr = lr
        current_eqm = 1
        current_epoch = 0
        momentums = [np.zeros_like(w) for w in self.W]

        while current_eqm > criterion and current_epoch < epochs:
            print(f'Epoch # {current_epoch} - lr: {current_lr} - EQM: {current_eqm} - ', end='')

            # Separa o conjunto de dados em treino e teste.
            Xtrain, Ytrain, Xtest, Ytest = self.split_train_test()

            for t in range(Xtrain.shape[1]):
                x_t = Xtrain[:, t].reshape((Xtrain.shape[0], 1))
                self.forward(x_t)
                d_t = Ytrain[:, t].reshape((Ytrain.shape[0], 1))
                self.backward(x_t, d_t, current_lr)

            #     for i in range(len(self.W)):
            #         momentums[i] = momentum * momentums[i] + current_lr * self.W[i]

            # for i in range(len(self.W)):
            #     self.W[i] += momentums[i]

            # Calcula o EQM para esta época.
            current_eqm = self.EQM(Xtrain=Xtrain, Ytrain=Ytrain)
            self.eqms_per_epoch.append(current_eqm)

            # Taxa de aprendizagem variável.
            current_lr = lr / (1 + current_epoch) # Decaimento Exponencial
            # current_lr = lr * (1 - (1 / epochs)) # Decaimento Linear

            # Testa a acurácia do modelo para esta época.
            acc = self.eval(Xtest=Xtest, Ytest=Ytest)
            self.accuracies_per_epoch.append(acc)

            print(f'Accuracy: {acc:.4f}%')

            current_epoch += 1
        print('Treinamento concluído! ', end='')
        print(f'Accuracy: {self.eval(Xtest=Xtest, Ytest=Ytest):.4f}%')


    def eval(self, Xtest: np.ndarray, Ytest: np.ndarray) -> float:
        """Realiza o teste do modelo.

        Parameters
        ----------
        Xtest : np.ndarray
            O conjunto de teste (X).
        Ytest : np.ndarray
            O conjunto de teste (Y).
        
        Returns
        -------
        float
            A acurácia do modelo.
        """
        corrects = 0
        for t in range(Xtest.shape[1]):
            x_t = Xtest[:, t].reshape((Xtest.shape[0], 1))
            self.forward(x_t)
            predicted_class = np.argmax(self.y[self.hidden_layers])
            if predicted_class == np.argmax(Ytest[:, t]):
                corrects += 1
        return (corrects * 100) / Ytest.shape[1]


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
                self.y[j] = self.activate(self.i[j])
            else:
                # y_bias ← y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = np.insert(self.y[j - 1], 0, -1, axis=0)
                # i[j] ← W[j] · y_bias
                self.i[j] = np.dot(w, y_bias)
                # y[j] ← g(i[j])
                self.y[j] = self.activate(self.i[j])


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
        # Mesma coisa que W - 1
        j = self.hidden_layers
        while j >= 0:
            if j + 1 == len(self.W):
                # δ[j] ← g′(i[j]) ◦ (d − y[j])
                self.delta[j] = np.multiply(self.activate_derivative(self.i[j]), d - self.y[j])
                # y_bias ← y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = np.insert(self.y[j - 1], 0, -1, axis=0)
                # W[j] ← W[j] + η(δ[j] ⊗ y_bias)
                self.W[j] = self.W[j] + lr * (np.outer(self.delta[j], y_bias))
            elif j == 0:
                # Wb[j + 1] ← W[j + 1] transposto sem o víes/bias (-1).
                Wb = self.W[j + 1][:, :-1].T[j + 1]
                # δ[j] ← g′(i[j]) ◦ (Wb[j + 1] · δ[j + 1])
                self.delta[j] = np.multiply(self.activate_derivative(self.i[j]), np.dot(Wb, self.delta[j + 1]))
                # W[j] ← W[j] + η(δ[j] ⊗ y_bias)
                self.W[j] = self.W[j] + lr * (np.outer(self.delta[j], x))
            else:
                # Wb[j + 1] ← W[j + 1] transposto sem o víes/bias (-1).
                Wb = self.W[j + 1][:, :-1].T[j + 1]
                # δ[j] ← g′(i[j]) ◦ (Wb[j + 1] · δ[j + 1])
                self.delta[j] = np.multiply(self.activate_derivative(self.i[j]), np.dot(Wb, self.delta[j + 1]))
                # y_bias ← y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = np.insert(self.y[j - 1], 0, -1, axis=0)
                # W[j] ← W[j] + η(δ[j] ⊗ y_bias)
                self.W[j] = self.W[j] + lr * (np.outer(self.delta[j], y_bias))
            j -= 1
