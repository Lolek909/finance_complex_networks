import os
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict

from src.utils import save_all

OUTPUT_DIR = "results/grid_search"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# METRICS
# =========================
def compute_metrics(G):
    degrees = [d for _, d in G.degree()]

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": float(np.mean(degrees)) if len(degrees) > 0 else 0,
        "max_degree": int(np.max(degrees)) if len(degrees) > 0 else 0,
        "components": nx.number_weakly_connected_components(G) if G.is_directed()
                      else nx.number_connected_components(G),
    }

def grid_search(log_returns, build_network, calculate_lead_lag_corr,
                sector_map, thresholds, window_sizes, step):

    nodes = log_returns.columns

    for window_size in window_sizes:
        for threshold in thresholds:

            print(f"\n=== WINDOW={window_size}, THRESH={threshold} ===")

            historical_links = defaultdict(list)
            total_windows = 0

            for start in range(0, len(log_returns) - window_size, step):
                window = log_returns.iloc[start:start + window_size]
                total_windows += 1

                for i, a in enumerate(nodes):
                    for j, b in enumerate(nodes):
                        if i == j:
                            continue

                        corr, lag = calculate_lead_lag_corr(window[a], window[b])

                        if corr > threshold:
                            historical_links[(a, b)].append(corr)

            ranking = []

            for (u, v), weights in historical_links.items():
                if len(weights) == 0:
                    continue

                avg_weight = np.mean(weights)
                freq = len(weights) / total_windows * 100

                ranking.append({
                    "Lead": u,
                    "Lag": v,
                    "Freq": freq,
                    "AvgWeight": avg_weight,
                    "Score": freq * avg_weight
                })

            ranking_df = pd.DataFrame(ranking).sort_values(
                by=["Freq", "AvgWeight"],
                ascending=False
            )

            # G_final = build_network(log_returns.tail(window_size), threshold)
            window = log_returns.iloc[-window_size:]
            G_final = build_network(window, threshold)
            metrics = compute_metrics(G_final)

            metrics.update({
                "window_size": window_size,
                "threshold": threshold,
                "n_windows": total_windows
            })

            name = f"w{window_size}_t{threshold}"
            out_dir = os.path.join(OUTPUT_DIR, name)

            save_all(G_final, ranking_df, metrics, out_dir, name)