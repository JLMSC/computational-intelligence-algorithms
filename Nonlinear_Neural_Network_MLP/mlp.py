import numpy as np


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

        # Pesos (valores aleatórios [-.5, .5]).
        self.initialize_weights()
        # Entradas.
        self.i = np.zeros((self.hidden_layers + 1, self.N))
        # Saídas.
        self.y = np.zeros((self.hidden_layers + 1, self.N))
        # Erro.
        self.delta = np.zeros((self.hidden_layers + 1, self.N))

        # Adiciona um vetor linha -1 em X.
        self.X = np.vstack((-np.ones((1, self.N)), X)) # XeR^(p+1)xN
        self.Y = Y # YeR^cxN


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

    # def 