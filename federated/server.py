import flwr as fl

from federated.model import TensionClassifier


def _get_initial_parameters() -> fl.common.Parameters:
    """Return initial global model weights before any training begins."""
    model = TensionClassifier()
    return fl.common.ndarrays_to_parameters(model.get_parameters())


def build_strategy(num_stores: int = 3) -> fl.server.strategy.FedAvg:
    """
    Build the FedAvg aggregation strategy for the global model.

    FedAvg (Federated Averaging):
      Each round, all store clients train locally and return weight updates.
      The server computes a weighted average of those updates — weighted by
      each store's number of training samples — to update the global model.

    No raw data leaves any store. Only weight arrays are aggregated.

    Args:
        num_stores: Number of store clients participating in training.
    """
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_stores,
        min_evaluate_clients=num_stores,
        min_available_clients=num_stores,
        initial_parameters=_get_initial_parameters(),
    )
