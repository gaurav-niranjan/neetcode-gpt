import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        
        y_zeros_idx = np.where(y_true == 0)
        y_pred_vals = y_pred[y_zeros_idx]
        y_pred[y_zeros_idx] = 1 - y_pred_vals

        loss = (-1/float(y_true.size))*np.sum(np.log(y_pred + 1e-7))
        return np.round(loss, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_true_c_row , y_true_c_col = np.where(y_true == 1)

        p_selected = y_pred[y_true_c_row, y_true_c_col]

        loss = (-1/float(len(y_true)))*np.sum(np.log(p_selected + 1e-7))
        return np.round(loss, 4)


        
