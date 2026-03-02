import numpy as np
import flwr as fl

from federated.model import TensionClassifier


def _generate_store_data(store_id: int, n_samples: int = 120):
    """
    Generate synthetic local feedback data for a store.

    Simulates staff confirming or rejecting AuraSentinel alerts via the PDA app.
    In a production deployment, this would be real feedback collected over time.

    Features: [visual_score, audio_score, movement_intensity]
    Label:    1 = genuine tension (alert was accurate)
              0 = false alarm (staff dismissed the alert)

    Each store uses a different random seed and noise level to simulate
    the variability across real-world environments (lighting, crowd, layout).
    """
    rng = np.random.RandomState(store_id * 7)
    X = rng.rand(n_samples, 3).astype(np.float32)

    # Ground truth: alerts are genuine when visual + audio both show high tension
    y = ((X[:, 0] > 0.55) & (X[:, 1] > 0.50)).astype(np.int64)

    # Store-specific noise (different ambient conditions per location)
    noise_scale = 0.04 + store_id * 0.015
    X = np.clip(X + rng.normal(0, noise_scale, X.shape).astype(np.float32), 0, 1)

    return X, y


class StoreClient(fl.client.NumPyClient):
    """
    Flower FL client representing a single retail store.

    Behaviour per training round:
      1. Receive the current global model weights from the server.
      2. Fine-tune locally on this store's feedback data.
      3. Return updated weights to the server.

    Privacy guarantee: only weight arrays (gradients) are sent to the server.
    Raw feedback data — staff responses, audio features, pose data — never
    leaves the local edge device.
    """

    def __init__(self, store_id: int):
        self.store_id = store_id
        self.model = TensionClassifier()
        self.X, self.y = _generate_store_data(store_id)
        n_genuine = int(self.y.sum())
        n_false = int((self.y == 0).sum())
        print(
            f"  [Store {store_id}] {len(self.X)} feedback samples — "
            f"{n_genuine} genuine alerts, {n_false} false alarms"
        )

    def get_parameters(self, config):
        return self.model.get_parameters()

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        self.model.fit(self.X, self.y)
        acc = self.model.accuracy(self.X, self.y)
        print(f"  [Store {self.store_id}] Local training — accuracy: {acc:.2%}")
        return self.model.get_parameters(), len(self.X), {"local_accuracy": float(acc)}

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        acc = self.model.accuracy(self.X, self.y)
        return float(1.0 - acc), len(self.X), {"accuracy": float(acc)}
