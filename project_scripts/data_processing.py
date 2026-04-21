from data_downloading import data, sector_map
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
import json

# 3. Obliczenia log-zwrotów
log_returns = np.log(data / data.shift(1)).dropna()
# log_returns = log_returns.resample("5min").mean() # resample
log_returns = log_returns.clip( # outliery
    log_returns.quantile(0.01),
    log_returns.quantile(0.99),
    axis=1
)
log_returns = log_returns.ewm(span=10).mean() # smoothing
def safe_corr(a, b, lag):
    a_shifted = a.shift(lag)
    combined = pd.concat([a_shifted, b], axis=1)
    combined.columns = ['a', 'b']
    combined = combined.dropna()
    
    if len(combined) < 10:
        print("brak danych", flush=True)
        return 0
    
    if combined['a'].std() < 1e-9 or combined['b'].std() < 1e-9:
        return 0
        
    return combined['a'].corr(combined['b'])
# 4. Funkcje sieciowe
def calculate_lead_lag_corr(series_a, series_b, max_lag=5):
    corrs = []
    for lag in range(1, max_lag + 1):
        # A(t-lag) wpływa na B(t)
        c = series_a.shift(lag).corr(series_b)
        #c = safe_corr(series_a, series_b, lag)
        corrs.append(c)

    if not corrs or np.all(np.isnan(corrs)):
        return 0, 0

    max_c = np.nanmax(corrs)
    best_lag = np.nanargmax(corrs) + 1
    return (max_c, best_lag) if not np.isnan(max_c) else (0, 0)


def build_network(returns_window, threshold=0.30):
    G = nx.DiGraph()
    nodes = returns_window.columns
    G.add_nodes_from(nodes)

    for i, t_a in enumerate(nodes):
        for j, t_b in enumerate(nodes):
            if i == j: continue
            corr, lag = calculate_lead_lag_corr(returns_window[t_a], returns_window[t_b])
            if corr > threshold:
                G.add_edge(t_a, t_b, weight=corr, lag=lag)
    return G


# 5. ZOPTYMALIZOWANA SYMULACJA CZASU I 
def main():
    window_size = 60
    step = 15


    historical_links = defaultdict(list)
    total_windows = 0
    nodes = log_returns.columns

    print(f"Rozpoczynam analizę czasową. Całkowita liczba wierszy danych: {len(log_returns)}")

    # Pętla symulująca upływ czasu (tylko matematyka, zero grafów)
    for start_idx in range(0, len(log_returns) - window_size, step):
        end_idx = start_idx + window_size
        window_data = log_returns.iloc[start_idx:end_idx]
        total_windows += 1

        for i, t_a in enumerate(nodes):
            for j, t_b in enumerate(nodes):
                if i == j: continue

                # Liczymy korelacje bezpośrednio
                corr, lag = calculate_lead_lag_corr(window_data[t_a], window_data[t_b])
                if corr > 0.20:  # Próg (threshold)
                    historical_links[(t_a, t_b)].append(corr)

    # 6. WYZNACZANIE RANKINGU
    print("\n--- RANKING NAJSILNIEJSZYCH RELACJI LEAD-LAG ---")
    ranking = []
    for (u, v), weights in historical_links.items():
        occurrences = len(weights)
        freq = (occurrences / total_windows) * 100
        avg_weight = np.mean(weights)
        ranking.append({"Lead": u, "Lag": v, "Freq": freq, "AvgWeight": avg_weight, "R-kwadrat": f"{avg_weight**2*100}%"})

    ranking_df = pd.DataFrame(ranking).sort_values(by=['Freq', 'AvgWeight'], ascending=[False, False])
    print(ranking_df.head(15).to_string(index=False))

    # [W tym miejscu zostawiasz stary kod punktu 7 - Wizualizacja grafu dla last_window]

    # 7. WIZUALIZACJA (Dla ostatniego okna, żeby pokazać wagi)
    print("\nGenerowanie grafu dla OSTATNIEGO dostępnego okna czasowego...")
    last_window = log_returns.tail(window_size)

            
    G_final = build_network(last_window, threshold=0.3)  # Wyższy próg dla czytelności grafu

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G_final, k=0.5, iterations=100)

    color_map = {'Tech': '#1f77b4', 'Finance': '#ff7f0e', 'Energy': '#2ca02c', 'Healthcare': '#d62728'}
    node_colors = [color_map.get(sector_map.get(node), 'grey') for node in G_final.nodes()]

    # Rysowanie węzłów
    nx.draw_networkx_nodes(G_final, pos, node_size=700, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G_final, pos, font_size=9, font_weight='bold')

    # Rysowanie krawędzi
    edges = G_final.edges(data=True)
    if edges:
        weights_for_plot = [d['weight'] * 5 for u, v, d in edges]
        nx.draw_networkx_edges(G_final, pos, width=weights_for_plot, edge_color='gray',
                               alpha=0.4, arrowsize=20, connectionstyle="arc3,rad=0.1")

        # NOWOŚĆ: Rysowanie etykiet (wag) na krawędziach
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G_final, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

    plt.title("Sieć Zależności Lead-Lag (Ostatnie 60 min)\nWagi widoczne na krawędziach")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
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


# =========================
# PLOT GRAPH
# =========================
def plot_graph(G, path, title, sector_map):

    plt.figure(figsize=(12, 9))

    if G.number_of_nodes() == 0:
        plt.title("Empty graph")
        plt.savefig(path)
        plt.close()
        return

    pos = nx.spring_layout(G, seed=42)

    color_map = {
        'Tech': '#1f77b4',
        'Finance': '#ff7f0e',
        'Energy': '#2ca02c',
        'Healthcare': '#d62728'
    }

    node_colors = [
        color_map.get(sector_map.get(n, ""), "grey")
        for n in G.nodes()
    ]

    node_sizes = [50 + 50 * G.degree(n) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9)

    nx.draw_networkx_labels(G, pos, font_size=8)

    edges = list(G.edges(data=True))

    if edges:
        widths = [d.get("weight", 0.1) * 3 for _, _, d in edges]

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, _ in edges],
            width=widths,
            alpha=0.4
        )

        edge_labels = {
            (u, v): f"{d.get('weight', 0):.2f}"
            for u, v, d in edges
        }

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=6
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_all(G, ranking_df, metrics, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/graph.pkl", "wb") as f:
        pickle.dump(G, f)

    nx.write_gexf(G, f"{out_dir}/graph.gexf")

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    ranking_df.to_csv(f"{out_dir}/ranking.csv", index=False)

    plot_graph(
        G,
        f"{out_dir}/graph.png",
        name,
        sector_map
    )


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
            
            
if __name__ == '__main__':
    # main()
    grid_search(
    log_returns,
     build_network,
     calculate_lead_lag_corr,
     sector_map,
     thresholds=[0.2, 0.3, 0.4],
     window_sizes=[30, 60, 120],
     step=15)