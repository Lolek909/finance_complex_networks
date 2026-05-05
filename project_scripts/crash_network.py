import os
import numpy as np
import networkx as nx

from src.data import get_data, sector_map
from src.impact_reduction import sector_impact_reduction
from src.metrics import evaluate_pair_unified


CRASH_DATE = "2022-04-29"


def build_network_unified(price_window, vol_window=None, threshold=0.4):
    G = nx.DiGraph()
    nodes = price_window.columns
    G.add_nodes_from(nodes)

    for i, t_a in enumerate(nodes):
        for j, t_b in enumerate(nodes):
            if i == j:
                continue

            vol_a = vol_window[t_a] if vol_window is not None else None

            weight, lag, granger, mi, vp = evaluate_pair_unified(
                price_window[t_a],
                price_window[t_b],
                vol_a
            )

            if abs(weight) > threshold:
                G.add_edge(t_a, t_b, weight=weight, lag=lag, granger=granger, mutual_info=mi)

    return G


def process_data(data):
    log_returns = np.log(data / data.shift(1)).dropna()

    log_returns = log_returns.clip(
        log_returns.quantile(0.01),
        log_returns.quantile(0.99),
        axis=1
    )

    return log_returns


def load_crash_window(period="10y", window=60):
    data = get_data(period=period, interval="1d", volume=False)
    data.index = data.index = __import__("pandas").to_datetime(data.index)

    event = __import__("pandas").to_datetime(CRASH_DATE)

    start = event - __import__("pandas").Timedelta(days=window)
    end = event - __import__("pandas").Timedelta(days=1)

    return data[(data.index >= start) & (data.index <= end)]


def main():
    import pandas as pd

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    out_dir = os.path.join(project_root, "results", "crash")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading crash window (BEFORE AMZN crash)...")

    prices = load_crash_window(period="10y", window=60)

    print("Processing returns...")
    log_returns = process_data(prices)

    print("Applying sector reduction...")
    log_returns = sector_impact_reduction(log_returns, sector_map)

    print("Building network (threshold=0.4)...")
    G = build_network_unified(log_returns, threshold=0.4)

    # --- save graph ---
    graph_path = os.path.join(out_dir, "crash_network_before_amzn.pkl")

    import pickle
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    print(f"Saved graph -> {graph_path}")

    # --- basic stats ---
    print("\nGraph stats:")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())


if __name__ == "__main__":
    main()