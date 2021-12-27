import numpy as np


class NNet:
    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def tanh(self, x, deriv=False):
        if deriv:
            return 1 - x ** 2
        return np.tanh(x)

    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.W = []  # Weights
        self.W += [np.random.rand(self.input_size + 1, self.hidden_sizes[0])]
        for i in range(1, len(self.hidden_sizes)):
            self.W += [np.random.rand(self.hidden_sizes[i - 1] + 1, self.hidden_sizes[i])]
        self.W += [np.random.rand(self.hidden_sizes[len(self.hidden_sizes) - 1] + 1, self.output_size)]

    def train(self, X, y, activation='sigmoid', epochs=1000, eta=1):
        if activation == 'sigmoid':
            self.activation = self.sigmoid
        if activation == 'tanh':
            self.activation = self.tanh
        self.bias = np.ones((len(X), 1))
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print(f"epoch {epoch}")
            L = []
            Z = [X]  # [Z0]
            delta = []
            for i in range(len(self.W)):
                L += [np.append(Z[i], self.bias, axis=1)]
                Z += [self.activation(np.dot(L[i], self.W[i]))]
            L += [Z[len(Z) - 1]]
            delta += [y - Z[len(Z) - 1]]
            for i in range(1, len(self.W)):
                delta += [
                    (np.dot(delta[i - 1] * self.activation(Z[len(Z) - i], True), self.W[len(self.W) - i].T))[:, :-1]]
            for i in range(len(self.W)):
                self.W[len(self.W) - 1 - i] += eta * np.dot(L[len(self.W) - 1 - i].T,
                                                      delta[i] * self.activation(Z[len(self.W) - i], True)) / (len(X))

    def predict(self, X):
        bias = np.ones((len(X), 1))
        Z = X
        for i in range(len(self.W)):
            L = np.append(Z, bias, axis=1)
            Z = self.activation(np.dot(L, self.W[i]))
        L = Z
        return np.argmax(Z, axis=1)
