import numpy as np

from typing import Tuple


class Adaline:
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        """A single-layer artificial neural network.

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

        Training set used 80% of data from the dataset.
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


    def sign(self, u) -> int:
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


    def MSE(self, X_train: np.ndarray, Y_train: np.ndarray) -> float:
        """Calculates the Mean Squared Error (MSE)

        Parameters
        ----------
        X_train : np.ndarray
            The input vector.
        Y_train : np.ndarray
            The output vector.

        Returns
        -------
        float
            The MSE calculated value.
        """
        mse = 0
        for t in range(X_train.shape[1]):
            x_t = X_train[:, t]
            u_t = self.W.T @ x_t
            d_t = Y_train[t, 0]
            mse += (d_t - u_t) ** 2
        return mse / (2 * X_train.shape[1])


    def fit(self, epochs: int, lr: float, pr: float) -> None:
        """Trains the model.

        Parameters
        ----------
        epochs : int
            The maximum amount of epochs.
        lr : float
            The learning rate.
        pr : float
            The precision rate.
        """
        # Reset the weights vector.
        self.W = np.zeros(shape=self.p)

        curr_epoch = 0
        prev_mse, next_mse = 0, 0
        while abs(prev_mse - next_mse) > pr or curr_epoch < epochs:
            print(f'Epoch # {curr_epoch} - lr: {lr} - ', end='') 

            X_train, X_test, Y_train, Y_test = self.split_train_test()
            prev_mse = self.MSE(X_train=X_train, Y_train=Y_train)

            for t in range(X_train.shape[1]):
                x_t = X_train[:, t]
                u_t = (self.W.T @ x_t)
                d_t = Y_train[t, 0]
                self.W = self.W + lr * (d_t - u_t) * x_t

            curr_epoch += 1
            next_mse = self.MSE(X_train=X_train, Y_train=Y_train)

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