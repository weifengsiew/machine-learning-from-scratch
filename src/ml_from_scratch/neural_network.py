"""A small fully-connected neural network trained with backpropagation."""

import numpy as np


def sigmoid(values):
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def initialize_weights(input_size, hidden_layers, hidden_size, output_size, random_state=None):
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be at least 1")
    generator = np.random.default_rng(random_state)
    shapes = [(hidden_size, input_size + 1)]
    shapes += [(hidden_size, hidden_size + 1)] * (hidden_layers - 1)
    shapes.append((output_size, hidden_size + 1))
    return [generator.normal(size=shape) for shape in shapes]


class NeuralNetwork:
    """Dense sigmoid network with optional L2 regularization."""

    def __init__(self, weights):
        self.weights = [np.asarray(weight, dtype=float) for weight in weights]

    def forward(self, x):
        activation = np.insert(np.asarray(x, dtype=float), 0, 1.0)
        activations = [activation]
        preactivations = []
        for weights in self.weights:
            preactivation = weights @ activation
            preactivations.append(preactivation)
            activation = sigmoid(preactivation)
            if weights is not self.weights[-1]:
                activation = np.insert(activation, 0, 1.0)
            activations.append(activation)
        return preactivations, activations, activation

    def predict_proba(self, X):
        return np.asarray([self.forward(row)[2] for row in X])

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int).squeeze()

    def fit(self, X, y, learning_rate=0.1, epochs=100, batch_size=None,
            regularization=0.0, random_state=None, shuffle=True):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        y = y.reshape(len(y), -1)
        batch_size = batch_size or len(X)
        generator = np.random.default_rng(random_state)
        history = {"loss": []}

        for _ in range(epochs):
            indices = generator.permutation(len(X)) if shuffle else np.arange(len(X))
            for start in range(0, len(X), batch_size):
                batch = indices[start:start + batch_size]
                gradient_totals = [np.zeros_like(weight) for weight in self.weights]
                for x_i, y_i in zip(X[batch], y[batch]):
                    _, activations, prediction = self.forward(x_i)
                    deltas = [prediction - y_i]
                    for layer in range(len(self.weights) - 2, -1, -1):
                        next_delta = self.weights[layer + 1].T @ deltas[0]
                        next_delta = next_delta[1:] * activations[layer + 1][1:] * (1 - activations[layer + 1][1:])
                        deltas.insert(0, next_delta)
                    for index, delta in enumerate(deltas):
                        gradient_totals[index] += np.outer(delta, activations[index])
                for index, (weight, gradient) in enumerate(zip(self.weights, gradient_totals)):
                    penalty = regularization * weight
                    penalty[:, 0] = 0.0
                    self.weights[index] -= learning_rate * (gradient / len(batch) + penalty / len(batch))
            predictions = np.clip(self.predict_proba(X), 1e-12, 1 - 1e-12)
            history["loss"].append(float(-np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))))
        return history
