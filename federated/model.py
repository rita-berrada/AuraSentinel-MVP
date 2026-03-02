import numpy as np


class TensionClassifier:
    """
    Lightweight logistic regression for binary tension classification.

    Trained via federated learning across store nodes. Each store
    contributes local model updates (weight gradients) without ever
    sending raw data to the central server.

    Input features (3):
      - visual_score      : pose-based tension score from VisualTensionScorer
      - audio_score       : audio-based tension score from AudioTensionAnalyzer
      - movement_intensity: normalized body movement speed

    Output:
      - Probability of genuine tension [0.0, 1.0]
      - Threshold at 0.65 triggers a staff alert
    """

    ALERT_THRESHOLD = 0.65

    def __init__(self):
        np.random.seed(0)
        self.weights = np.random.randn(3) * 0.01
        self.bias = 0.0

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return tension probabilities for each sample in X."""
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (1 = genuine tension)."""
        return (self.predict_proba(X) >= self.ALERT_THRESHOLD).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 15):
        """
        Train locally on store feedback data using gradient descent.

        Args:
            X:      Feature matrix (n_samples, 3).
            y:      Labels — 1 = accurate alert, 0 = false alarm.
            lr:     Learning rate.
            epochs: Number of local training epochs per FL round.
        """
        for _ in range(epochs):
            preds = self.predict_proba(X)
            error = preds - y
            self.weights -= lr * (X.T @ error) / len(y)
            self.bias -= lr * float(error.mean())

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def get_parameters(self) -> list[np.ndarray]:
        """Serialize model state as a list of numpy arrays for Flower."""
        return [self.weights.copy(), np.array([self.bias])]

    def set_parameters(self, params: list[np.ndarray]):
        """Restore model state from Flower parameter arrays."""
        self.weights = params[0].copy()
        self.bias = float(params[1][0])
