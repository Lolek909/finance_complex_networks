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


def build_mst_network(G_weighted):
    if G_weighted.number_of_edges() == 0:
        return nx.Graph()

    H = nx.Graph()
    for u, v, d in G_weighted.edges(data=True):
        w = d.get("weight", 0)
        dist = np.sqrt(2 * (1 - min(max(w, -1), 1)))
        H.add_edge(u, v, weight=dist)

    return nx.minimum_spanning_tree(H, weight='weight')


def process_pair(a, b, win_price_a, win_price_b, vol_a, threshold, evaluate_pair):
    pearson = win_price_a.corr(win_price_b)
    if abs(pearson) < 0.1:
        return None

    weight, lag, _, _, _ = evaluate_pair(win_price_a, win_price_b, vol_a)
    if weight > threshold:
        return (a, b, weight)
    return None


def grid_search(log_returns, build_network, evaluate_pair, sector_map, thresholds, window_sizes, step, log_volumes=None, raw_prices=None):
    nodes = log_returns.columns
    n_jobs = -1

    for window_size in window_sizes:
        for threshold in thresholds:
            print(f"\n=== WINDOW={window_size}, THRESH={threshold} ===")

            historical_links = defaultdict(list)
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
                    delayed(process_pair)(p[0], p[1], p[2], p[3], p[4], threshold, evaluate_pair)
                    for p in pairs_to_check
                )

                for res in results:
                    if res:
                        u, v, w = res
                        historical_links[(u, v)].append(w)

            ranking = []
            for (u, v), weights in historical_links.items():
                avg_weight = np.mean(weights)
                freq = len(weights) / total_windows * 100
                ranking.append({
                    "Lead": u,
                    "Lag": v,
                    "Freq": freq,
                    "AvgWeight": avg_weight,
                    "Score": freq * avg_weight
                })

            ranking_df = pd.DataFrame(ranking).sort_values(by="Score", ascending=False)

            final_win_price = log_returns.iloc[-window_size:]
            final_win_vol = log_volumes.iloc[-window_size:] if log_volumes is not None else None

            G_final_wta = build_network(final_win_price, final_win_vol, threshold)
            G_mst = build_mst_network(G_final_wta)

            metrics = compute_topology_metrics(G_final_wta)
            metrics.update({
                "window_size": window_size,
                "threshold": threshold,
                "n_windows": total_windows,
                "mst_edges": G_mst.number_of_edges()
            })

            name = f"w{window_size}_t{threshold}"
            out_dir = os.path.join(OUTPUT_DIR, name)

            save_all(G_final_wta, ranking_df, metrics, out_dir, name)

            nx.write_gexf(G_mst, f"{out_dir}/graph_mst.gexf")
            print(f"Zakonczenie {name}. Zapisano w {out_dir}")

            if raw_prices is not None:
                raw_win_price = raw_prices.iloc[-window_size:]

                run_epidemic_simulation(
                    G=G_final_wta,
                    raw_prices_window=raw_win_price,
                    sector_map=sector_map,
                    out_dir=out_dir,
                    name=name,
                    threshold=-0.05
                )

            plot_sector_flow_heatmap(
                G=G_final_wta,
                sector_map=sector_map,
                out_dir=out_dir,
                name=name
            )