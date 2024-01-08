import numpy as np

from typing import Tuple


class SimplePerceptron:
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        """A simple perceptron.

        Parameters
        ----------
        X : np.ndarray
            It's the input vector.
            NOTE: XeR^(p+1)xN
        Y : np.ndarray
            It's the output of the model.
            NOTE: YeR^Nx1
        """
        self.X = X
        self.Y = Y
        self.N = X.shape[1] # The amount of samples/inputs.
        self.p = X.shape[0] # The amount of inputs. (refers to the input vector dimensions)
        self.W = np.zeros(shape=self.p) # The weight vector.


    def split_train_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits the dataset (both input and output vectors) into
        training and testing sets.

        Training set uses 80% of data from the dataset.
        Testing set uses the remaining 20%.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_test, Y_train, Y_test respectively.
        """
        seed = np.random.permutation(x=self.N)
        train_indexes = seed[:int(self.N * 0.8)]
        test_indexes = seed[int(self.N * 0.8):]

        X_train, X_test = self.X[:, train_indexes], self.X[:, test_indexes]
        Y_train, Y_test = self.Y[train_indexes], self.Y[test_indexes]

        return X_train, X_test, Y_train, Y_test


    def sign(self, u: float) -> int:
        """Activation function.

        Parameters
        ----------
        u: Any
            Activation step.
        
        Returns
        -------
        int
            +1 or -1, refers to normalization method used.
        """
        return 1 if u >= 0 else -1


    def fit(self, epochs: int, lr: float) -> None:
        """Trains the model.

        Parameters
        ----------
        epochs : int
            The maximum amount of epochs.
        lr : float
            The learning rate.
        """
        # Reset the wieghts vector.
        self.W = np.zeros(shape=self.p)

        error = True
        current_epoch = 0
        while error:
            if current_epoch > epochs:
                break

            print(f'Epoch # {current_epoch} - lr: {lr} - ', end='')

            X_train, X_test, Y_train, Y_test = self.split_train_test()

            error = False
            for t in range(X_train.shape[1]):
                x_t = X_train[:, t]
                u_t = (self.W.T @ x_t)
                y_t = self.sign(u=u_t)
                d_t = Y_train[t, 0]
                self.W = self.W + lr * (d_t - y_t) * x_t
                if d_t != y_t:
                    error = True

            current_epoch += 1

            # Test model's current accuracy.
            self.eval(X_test=X_test, Y_test=Y_test)
        print('Done. ', end='')


    def eval(self, X_test: np.ndarray, Y_test: np.ndarray) -> None:
        """Test the model's output vector accuracy.

        Parameters
        ----------
        X_test : np.ndarray
            The input vector used on testing set.
        Y_test : np.ndarray
            The output vector used on testing set.
        """
        correct = 0
        for t in range(X_test.shape[1]):
            x_t = X_test[:, t]
            u_t = self.W.T @ x_t
            y_t = self.sign(u=u_t)
            if y_t == Y_test[t]:
                correct += 1
        print(f'Accuracy: {(correct / Y_test.shape[0]) * 100}')

