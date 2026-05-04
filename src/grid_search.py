import os
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
from src.utils import save_all
from src.epidemic import run_epidemic_simulation
from src.macro import plot_sector_flow_heatmap

OUTPUT_DIR = "results/grid_search"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_topology_metrics(G):
    if G.number_of_nodes() == 0:
        return {}

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    degrees = [d for _, d in G.degree()]

    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": float(np.mean(degrees)),
        "max_out_degree_lead": int(np.max(list(out_degrees.values()))),
        "max_in_degree_lag": int(np.max(list(in_degrees.values()))),
        "components": nx.number_weakly_connected_components(G)
    }
    return metrics

def process_pair(a, b, win_price_a, win_price_b, vol_a, evaluate_pair):
    pearson = win_price_a.corr(win_price_b)
    if abs(pearson) < 0.15:
        return None
    weight, lag, _, _, _ = evaluate_pair(win_price_a, win_price_b, vol_a)
    return (a, b, weight)


def grid_search(log_returns, build_network, evaluate_pair, sector_map, thresholds, window_sizes, step, log_volumes=None, raw_prices=None):
    nodes = log_returns.columns
    n_jobs = -1

    for window_size in window_sizes:
        print(f"\n=== WINDOW={window_size} (Obliczanie raz dla wszystkich progów) ===")

        historical_links = {t: defaultdict(list) for t in thresholds}
        total_windows = 0

        for start in range(0, len(log_returns) - window_size, step):
            win_price = log_returns.iloc[start:start + window_size]
            win_vol = log_volumes.iloc[start:start + window_size] if log_volumes is not None else None
            total_windows += 1

            pairs_to_check = []
            for i, a in enumerate(nodes):
                for j, b in enumerate(nodes):
                    if i == j: continue
                    vol_a = win_vol[a] if win_vol is not None else None
                    pairs_to_check.append((a, b, win_price[a], win_price[b], vol_a))

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_pair)(p[0], p[1], p[2], p[3], p[4], evaluate_pair)
                for p in pairs_to_check
            )

            for res in results:
                if res:
                    u, v, w = res
                    abs_w = abs(w)
                    for threshold in thresholds:
                        if abs_w > threshold:
                            historical_links[threshold][(u, v)].append(w)

        final_win_price = log_returns.iloc[-window_size:]
        final_win_vol = log_volumes.iloc[-window_size:] if log_volumes is not None else None

        for threshold in thresholds:
            ranking = []
            for (u, v), weights in historical_links[threshold].items():
                avg_weight = np.mean(weights)
                freq = len(weights) / total_windows * 100
                ranking.append({
                    "Lead": u,
                    "Lag": v,
                    "Freq": freq,
                    "AvgWeight": avg_weight,
                    "Score": freq * abs(avg_weight)
                })

            ranking_df = pd.DataFrame(ranking)
            if not ranking_df.empty:
                ranking_df = ranking_df.sort_values(by="Score", ascending=False)
            else:
                ranking_df = pd.DataFrame(columns=["Lead", "Lag", "Freq", "AvgWeight", "Score"])

            G_final_wta = build_network(final_win_price, final_win_vol, threshold)

            metrics = compute_topology_metrics(G_final_wta)
            metrics.update({
                "window_size": window_size,
                "threshold": threshold,
                "n_windows": total_windows,
            })

            name = f"w{window_size}_t{threshold}"
            out_dir = os.path.join(OUTPUT_DIR, name)
            os.makedirs(out_dir, exist_ok=True)

            save_all(G_final_wta, ranking_df, metrics, out_dir, name)

            print(f"Zakonczenie {name}. Zapisano w {out_dir}")

            plot_sector_flow_heatmap(
                G=G_final_wta,
                sector_map=sector_map,
                out_dir=out_dir,
                name=name
            )