import numpy as np
from federated.model import TensionClassifier


def fed_avg(
    updates: list[tuple[list[np.ndarray], int]]
) -> list[np.ndarray]:
    """
    Federated Averaging (FedAvg) aggregation.

    Computes a weighted average of client weight updates,
    weighted by each client's number of training samples.

    No raw data is involved — only the weight arrays sent by each store.

    Args:
        updates: List of (parameters, n_samples) from each store client.

    Returns:
        Aggregated global model parameters.
    """
    total_samples = sum(n for _, n in updates)
    n_params = len(updates[0][0])

    aggregated = []
    for i in range(n_params):
        weighted = sum(params[i] * (n / total_samples) for params, n in updates)
        aggregated.append(weighted)

    return aggregated


def get_initial_parameters() -> list[np.ndarray]:
    """Return initial global model weights before any training."""
    return TensionClassifier().get_parameters()
