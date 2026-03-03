"""
AuraSentinel — Federated Learning Simulation

Simulates the privacy-preserving model improvement loop described in the
AuraSentinel system design. Multiple retail stores (clients) train the
tension classifier locally on staff feedback data. Only model weight
updates are shared with the central server; raw data never leaves the edge.

Each round:
  1. Server broadcasts the current global model weights to all stores.
  2. Each store trains locally on its own staff feedback data.
  3. Stores send back weight updates (no raw data).
  4. Server aggregates updates via FedAvg → improved global model.

Usage:
    python federated/simulate.py
"""

import numpy as np
from federated.client import StoreClient
from federated.server import fed_avg, get_initial_parameters

NUM_STORES = 3
NUM_ROUNDS = 5


if __name__ == "__main__":
    print("=" * 55)
    print("  AuraSentinel — Federated Learning Simulation")
    print("=" * 55)
    print(f"\n  Stores  : {NUM_STORES}")
    print(f"  Rounds  : {NUM_ROUNDS}")
    print(f"  Strategy: FedAvg (Federated Averaging)")
    print(f"  Privacy : Raw data stays on each store node\n")

    print("Initializing store clients...")
    clients = [StoreClient(store_id=i) for i in range(NUM_STORES)]

    global_params = get_initial_parameters()

    print()
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"Round {round_num}/{NUM_ROUNDS}")

        # Each store trains locally and returns weight updates
        updates = []
        for client in clients:
            local_params, n_samples, _ = client.fit(global_params, {})
            updates.append((local_params, n_samples))

        # Server aggregates with FedAvg — no raw data involved
        global_params = fed_avg(updates)

        # Evaluate the new global model on each store's local data
        accs = [metrics["accuracy"] for client in clients for _, _, metrics in [client.evaluate(global_params, {})]]
        print(f"  Global accuracy after round {round_num}: {np.mean(accs):.2%}\n")

    print("=" * 55)
    print("  Federated Learning complete")
    print("=" * 55)
    print("\nThe global model has been updated using aggregated feedback")
    print("from all stores — no raw data left any edge device.\n")
