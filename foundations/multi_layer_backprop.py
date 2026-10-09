import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)   # (hidden, in)
        b1 = np.array(b1, dtype=float)   # (hidden,)
        W2 = np.array(W2, dtype=float)   # (out, hidden)
        b2 = np.array(b2, dtype=float)   # (out,)
        y = np.array(y_true, dtype=float)

        # ---- forward ----
        z1 = W1 @ x + b1                 # (hidden,)
        a1 = np.maximum(0, z1)           # ReLU
        pred = W2 @ a1 + b2              # (out,)
        loss = np.mean((pred - y) ** 2)

        # ---- backward ----
        dpred = 2 * (pred - y) / pred.size
        dW2 = np.outer(dpred, a1)        # (out, hidden)
        db2 = dpred

        da1 = W2.T @ dpred               # (hidden,)
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)           # (hidden, in)
        db1 = dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }
