"""
AuraSentinel — Federated Learning Simulation

Simulates the privacy-preserving model improvement loop described in the
AuraSentinel system design. Multiple retail stores (clients) train the
tension classifier locally on staff feedback data. Only model weight
updates are shared with the central server; raw data never leaves the edge.

Each store client represents a real Jetson Nano edge device that:
  - Receives the current global model from the server
  - Trains locally on staff feedback (alert accurate / false alarm)
  - Returns only weight updates — no raw data transmitted

The server runs Federated Averaging (FedAvg) to merge all updates into
an improved global model that benefits every store.

Usage:
    python federated/simulate.py
    python -m federated.simulate
"""

import flwr as fl

from federated.client import StoreClient
from federated.server import build_strategy

NUM_STORES = 3
NUM_ROUNDS = 5


def client_fn(cid: str) -> fl.client.NumPyClient:
    """Instantiate a store client by its integer ID."""
    return StoreClient(store_id=int(cid))


if __name__ == "__main__":
    print("=" * 55)
    print("  AuraSentinel — Federated Learning Simulation")
    print("=" * 55)
    print(f"\nConfiguration:")
    print(f"  Stores (clients) : {NUM_STORES}")
    print(f"  Training rounds  : {NUM_ROUNDS}")
    print(f"  Strategy         : FedAvg (Federated Averaging)")
    print(f"  Privacy          : Raw data stays on each store node")
    print(f"\nInitializing store clients...\n")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_STORES,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=build_strategy(NUM_STORES),
    )

    print("\n" + "=" * 55)
    print("  Federated Learning complete")
    print("=" * 55)

    if history.metrics_distributed:
        for metric, values in history.metrics_distributed.items():
            print(f"\n  {metric} per round:")
            for rnd, val in values:
                print(f"    Round {rnd}: {val:.2%}")

    print(
        "\nThe global model has been updated using aggregated feedback"
        "\nfrom all stores — no raw data left any edge device.\n"
    )
