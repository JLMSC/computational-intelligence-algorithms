import numpy as np

from typing import Tuple # Usado para notação de retorno das funções.


class PerceptronSimples:
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Construtor do Perceptron.

        Parameters
        ----------
        X : np.ndarray
            As variáveis independentes.
            Obs:. XeR^(p+1)xN
        Y : np.ndarray
            As variáveis dependentes.
            Obs:. YeR^Nx1
        """
        # Variáveis independentes.
        self.X = X
        # Variáveis dependentes.
        self.Y = Y
        # Quantidade de amostras.
        self.N = X.shape[1]
        # Quantidade de variáveis.
        self.p = X.shape[0]
        # Vetor de pesos, inicializado com valores nulos.
        self.W = np.zeros(self.p)
        # Verdadeiro/Falso Positivo/Negativo.
        self.VP, self.VN, self.FP, self.FN = 0, 0, 0, 0
        # Acurácias do modelo, de cada época.
        self.accuracies = []
        # Sensibilidades do modelo, de cada época.
        self.sensitivities = []
        # Especificidades do modelo, de cada época.
        self.specificities = []
        # VPS, VNS, FPS, FNS
        self.VPS, self.VNS = [], []
        self.FPS, self.FNS = [], []


    def split_train_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Faz a divisão das variáveis independentes (X) e dependentes (Y)
        em dados, aleatórios, de treino (80%) e teste (20%).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Os dados de treino e teste, de X e Y, respectivamente.
        """
        seed = np.random.permutation(self.N)
        train_indexes = seed[:int(self.N * 0.8)]
        test_indexes = seed[int(self.N * 0.8):]

        X_train = self.X[:, train_indexes]
        Y_train = self.Y[train_indexes]

        X_test = self.X[:, test_indexes]
        Y_test = self.Y[test_indexes]

        return X_train, Y_train, X_test, Y_test


    def sign(self, u: float) -> int:
        """Função de ativação binária.

        Parameters
        ----------
        u : float
            Passo de ativação

        Returns
        -------
        int
            +1 ou -1, conforme a normalização.
        """
        return 1 if u >= 0 else -1


    def fit(self, epochs: int, lr: float) -> None:
        """Realiza o treinamento do modelo.

        Parameters
        ----------
        epochs : int
            Quantidade de épocas a ser realizado o treinamento.
        lr : float
            Taxa de aprendizado.
        """
        # Reinicia o vetor de pesos com valores nulos.
        self.W = np.zeros(self.p)
        current_epoch = 0
        err = True
        while err:
            if current_epoch > epochs:
                break

            print(f'Epoch # {current_epoch} - lr: {lr} - ', end='')

            # Gera novos dados aleatórios de treino (80%) a cada época.
            Xtrain, Ytrain, Xtest, Ytest = self.split_train_test()

            err = False
            for t in range(Xtrain.shape[1]):
                x_t = Xtrain[:, t]
                u_t = (self.W.T @ x_t)
                y_t = self.sign(u_t)
                d_t = Ytrain[t, 0]
                self.W = self.W + lr * (d_t - y_t) * x_t
                if d_t != y_t:
                    err = True

            current_epoch += 1

            # Testa a acurácia do modelo na época atual.
            self.eval(Xtest=Xtest, Ytest=Ytest)
        print('Treinamento completo. ', end='')
        self.eval(Xtest=Xtest, Ytest=Ytest)


    def eval(self, Xtest: np.ndarray, Ytest: np.ndarray) -> None:
        """Realiza o teste do modelo.

        Parameters
        ----------
        Xtest : np.ndarray
            Variáveis independentes do conjunto de teste.
        Ytest : np.ndarray
            Variáveis dependentes do conjunto de teste.
        """ 
        self.VP, self.VN, self.FP, self.FN = 0, 0, 0, 0
        for t in range(Xtest.shape[1]):
            x_t = Xtest[:, t]
            u_t = self.W.T @ x_t
            y_t = self.sign(u_t)
            if y_t == Ytest[t]:
                if y_t == 1:
                    self.VP += 1
                else:
                    self.VN += 1
            else:
                if y_t == -1:
                    self.FN += 1
                else:
                    self.FP += 1
        self.accuracies.append(((self.VP + self.VN) / (self.VP + self.VN + self.FP + self.FN)) * 100)
        self.sensitivities.append((self.VP / (self.VP + self.FN)) * 100)
        self.specificities.append((self.VN / (self.VN + self.FP)) * 100)

        self.VPS.append(self.VP)
        self.VNS.append(self.VN)
        self.FPS.append(self.FP)
        self.FNS.append(self.FN)

        print(f'Accuracy: {self.accuracies[-1]:.2f}% - '
              f'Sensitivity: {self.sensitivities[-1]:.2f}% - '
              f'Specificity: {self.specificities[-1]:.2f}%')


