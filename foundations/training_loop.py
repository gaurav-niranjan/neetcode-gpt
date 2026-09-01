import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import torch
import torch.nn as nn


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        #print(X.shape)

        N, D = X.shape
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

        w = torch.zeros(D, 1, dtype=torch.float32, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

        for e in range(epochs):
            pred = X @ w + b
            loss = ((pred - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
            w.grad = None
            b.grad = None

        w_out = np.round(w.detach().numpy().reshape(-1), 5)
        b_out = round(float(b.item()), 5)
        return (w_out, b_out)

        
