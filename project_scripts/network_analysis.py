import os
import json
import matplotlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import kendalltau, linregress
from collections import Counter

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community import louvain_communities

tickers = {
    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'ADBE'],
    'Finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA', 'AXP', 'PYPL'],
    'Energy': ['XOM', 'CVX', 'SHEL', 'BP', 'TTE', 'COP', 'SLB', 'PBR', 'EQNR', 'VLO'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'AZN', 'NVO', 'DHR']
}

sector_map = {ticker: sector for sector, t_list in tickers.items() for ticker in t_list}
OUTPUT_DIR = "../results/analysis"

def safe_modularity(G, comms):
    if G.number_of_edges() == 0 or len(comms) == 0:
        return 0.0
    return nx.algorithms.community.modularity(G, comms)


def get_louvain(G):
    G_undirected = G.to_undirected()
    return louvain_communities(G_undirected, seed=42)


def build_labels(G):
    nodes = list(G.nodes())
    true = [sector_map.get(n, "Unknown") for n in nodes]
    return nodes, true


def community_labels(comms, nodes):
    mapping = {}
    for i, c in enumerate(comms):
        for n in c:
            mapping[n] = i
    return [mapping.get(n, -1) for n in nodes]


def compute_node_sizes(G, nodes):
    sizes = []
    for n in nodes:
        wdeg = sum(abs(d.get("weight", 0)) for _, _, d in G.edges(n, data=True))
        deg = G.degree(n)

        # mix: topology + strength
        score = 0.5 * deg + 0.5 * wdeg
        sizes.append(score)

    sizes = np.array(sizes)

    if sizes.max() > 0:
        sizes = 300 * (sizes / sizes.max())
    else:
        sizes = np.ones_like(sizes) * 100

    return sizes


def get_top_edges(G, top_k=40):
    edges = sorted(
        G.edges(data=True),
        key=lambda x: abs(x[2].get("weight", 0)),
        reverse=True
    )
    return edges[:top_k]


def plot_dual_view(G, comms, save_path, title):
    nodes = list(G.nodes())
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=120)

    node_sizes = compute_node_sizes(G, nodes)

    louvain_color = {}
    for i, c in enumerate(comms):
        for n in c:
            louvain_color[n] = i

    louvain_colors = [louvain_color.get(n, 0) for n in nodes]

    sector_colors = {
        "Tech": 0,
        "Finance": 1,
        "Energy": 2,
        "Healthcare": 3,
        "Unknown": -1
    }

    real_colors = [
        sector_colors.get(sector_map.get(n, "Unknown"), -1)
        for n in nodes
    ]

    edges = list(G.edges(data=True))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=louvain_colors,
        cmap=plt.cm.tab10,
        node_size=node_sizes,
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, _ in edges],
        width=[abs(d.get("weight", 0.1)) * 3 for _, _, d in edges],
        alpha=0.4,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    edge_labels = {
        (u, v): f"{d.get('weight', 0):.2f}"
        for u, v, d in edges
    }
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax
    )
    ax.set_title("Louvain communities")
    ax.axis("off")

    ax = axes[1]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=real_colors,
        cmap=plt.cm.Set2,
        node_size=node_sizes,
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, _ in edges],
        width=[abs(d.get("weight", 0.1)) * 3 for _, _, d in edges],
        alpha=0.4,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax
    )
    ax.set_title("Sektory (rzeczywiste)")
    ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze(G, meta):
    comms = get_louvain(G)

    nodes, true_labels = build_labels(G)
    pred_labels = community_labels(comms, nodes)

    mask = [t != "Unknown" for t in true_labels]

    if sum(mask) > 0:
        true_f = np.array(true_labels)[mask]
        pred_f = np.array(pred_labels)[mask]

        nmi = normalized_mutual_info_score(true_f, pred_f)
        ari = adjusted_rand_score(true_f, pred_f)
    else:
        nmi, ari = 0, 0

    metrics = {
        "modularity": safe_modularity(G, comms),
        "nmi": nmi,
        "ari": ari,
        "n_communities": len(comms),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()])
    }

    return comms, metrics


def save_degree_distribution(G, out_dir, name):
    degrees = [d for _, d in G.degree()]
    df = pd.DataFrame({"degree": degrees})
    df.to_csv(f"{out_dir}/degree_distribution.csv", index=False)
    plt.figure(figsize=(7, 5))
    sns.histplot(
        degrees,
        bins=15,
        kde=True,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5
    )

    plt.title(f"Degree distribution: {name}")
    plt.xlabel("Degree")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/degree_distribution.png")
    plt.close()


def plot_betweenness_centrality(G, out_dir, name):
    betweenness = nx.betweenness_centrality(G)
    plt.figure(figsize=(10, 6))
    sns.histplot(list(betweenness.values()), bins=25, kde=True, color="steelblue", edgecolor="white", linewidth=0.5)
    plt.title(f"Betweenness Centrality Distribution: {name}", fontsize=14)
    plt.xlabel("Betweenness")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/betweenness_distribution.png")
    plt.close()


def plot_closeness_centrality(G, out_dir, name):
    closeness = nx.closeness_centrality(G)
    plt.figure(figsize=(10, 6))
    sns.histplot(list(closeness.values()), bins=25, kde=True, color="steelblue", edgecolor="white", linewidth=0.5)
    plt.title(f"Closeness Centrality Distribution: {name}", fontsize=14)
    plt.xlabel("Closeness")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/closeness_distribution.png")
    plt.close()


def get_node_metrics_df(G) -> pd.DataFrame:
    G_simple = G.to_undirected() if G.is_directed() else G

    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G_simple)

    G_pr = G.copy()
    for u, v, d in G_pr.edges(data=True):
        if 'weight' in d:
            try:
                w = abs(float(d['weight']))
                if np.isnan(w) or np.isinf(w):
                    w = 0.0
                d['weight'] = w
            except (ValueError, TypeError):
                d['weight'] = 0.0

    try:
        pagerank = nx.pagerank(G_pr, max_iter=1000, weight='weight')
    except nx.PowerIterationFailedConvergence:
        print("    [Ostrzeżenie] PageRank nie zbiegł się, przypisuję wartości 0.0.")
        pagerank = {n: 0.0 for n in G.nodes()}

    if G.is_directed():
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
    else:
        in_degree = dict(G.degree())
        out_degree = dict(G.degree())

    df = pd.DataFrame({
        "out_degree_Lead": pd.Series(out_degree),
        "in_degree_Lag": pd.Series(in_degree),
        "betweenness": pd.Series(betweenness),
        "closeness": pd.Series(closeness),
        "clustering": pd.Series(clustering),
        "pagerank": pd.Series(pagerank)
    })

    return df.sort_values(by="pagerank", ascending=False)


def prove_scale_free_network(G, out_dir, name):
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return 0.0, 0.0

    deg_counts = Counter(degrees)
    x = np.array(list(deg_counts.keys()))
    y = np.array(list(deg_counts.values()))

    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return 0.0, 0.0

    log_x = np.log10(x)
    log_y = np.log10(y)

    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

    plt.figure(figsize=(8, 6))
    plt.scatter(log_x, log_y, color='steelblue', label='Dane empiryczne', zorder=5)
    plt.plot(log_x, intercept + slope * log_x, color='red', linewidth=2,
             label=f'Regresja liniowa (Nachylenie: {slope:.2f}, R²: {r_value ** 2:.2f})')

    plt.title(f"Dowód na Scale-Free Network (Log-Log) - {name}")
    plt.xlabel("log10(Stopień węzła)")
    plt.ylabel("log10(Częstotliwość)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scale_free_proof.png")
    plt.close()

    return slope, r_value ** 2


def calculate_kendall_tau(G):
    degree = dict(nx.degree(G))
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    nodes = list(G.nodes())
    deg_vals = [degree[n] for n in nodes]
    bet_vals = [betweenness[n] for n in nodes]
    close_vals = [closeness[n] for n in nodes]

    ken_deg_bet, _ = kendalltau(deg_vals, bet_vals)
    ken_deg_close, _ = kendalltau(deg_vals, close_vals)
    ken_bet_close, _ = kendalltau(bet_vals, close_vals)

    return {
        "tau_degree_betweenness": float(ken_deg_bet) if not np.isnan(ken_deg_bet) else 0,
        "tau_degree_closeness": float(ken_deg_close) if not np.isnan(ken_deg_close) else 0,
        "tau_betweenness_closeness": float(ken_bet_close) if not np.isnan(ken_bet_close) else 0
    }


def get_top_nodes(G, n=5):
    if G.is_directed():
        out_degree = dict(G.out_degree())
        in_degree = dict(G.in_degree())
    else:
        out_degree = dict(G.degree())
        in_degree = dict(G.degree())

    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    return {
        "top_out_degree_lead": sorted(out_degree, key=out_degree.get, reverse=True)[:n],
        "top_in_degree_lag": sorted(in_degree, key=in_degree.get, reverse=True)[:n],
        "top_betweenness": sorted(betweenness, key=betweenness.get, reverse=True)[:n],
        "top_closeness": sorted(closeness, key=closeness.get, reverse=True)[:n]
    }


def get_diameter_and_shortest_path(G):
    if G.is_directed():
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)

    G_largest = G.subgraph(largest_cc).copy()

    try:
        avg_shortest_path = nx.average_shortest_path_length(G_largest)
        diameter = nx.diameter(G_largest)
    except nx.NetworkXError:
        avg_shortest_path = 0
        diameter = 0

    return diameter, avg_shortest_path


def main():
    GRID_SEARCH_DIR = "../results/grid_search"
    MASTER_OUTPUT_DIR = "../results/analysis"
    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)

    summary = []

    for dataset_name in os.listdir(GRID_SEARCH_DIR):
        dataset_path = os.path.join(GRID_SEARCH_DIR, dataset_name)

        if not os.path.isdir(dataset_path):
            continue

        print(f"\n=== Rozpoczynam analizę datasetu: {dataset_name} ===")

        for param_folder in os.listdir(dataset_path):
            param_path = os.path.join(dataset_path, param_folder)
            if not os.path.isdir(param_path):
                continue

            graph_file = os.path.join(param_path, "graph.gexf")

            if not os.path.exists(graph_file):
                print(f"  [Pominięto] Brak pliku {graph_file}")
                continue

            print(f"  -> Przetwarzanie: {param_folder}")

            G = nx.read_gexf(graph_file)
            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                print(f"  [Pominięto] {param_folder} - pusty graf")
                continue

            if G.is_directed():
                largest_cc = max(nx.weakly_connected_components(G), key=len)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

            try:
                w_part, t_part = param_folder.split('_')
                window_size = int(w_part.replace('w', ''))
                threshold = float(t_part.replace('t', ''))
            except ValueError:
                window_size = None
                threshold = None

            meta = {
                "window_size": window_size,
                "threshold": threshold,
                "dataset": dataset_name
            }

            out_dir = os.path.join(MASTER_OUTPUT_DIR, dataset_name, param_folder)
            os.makedirs(out_dir, exist_ok=True)

            comms, metrics = analyze(G, meta)
            diameter, avg_shortest_path = get_diameter_and_shortest_path(G)
            kendall_corrs = calculate_kendall_tau(G)
            top_n = get_top_nodes(G, n=5)
            slope, r_squared = prove_scale_free_network(G, out_dir, f"{dataset_name}_{param_folder}")

            node_metrics_df = get_node_metrics_df(G)

            if G.is_directed():
                max_out_deg = max(dict(G.out_degree()).values()) if len(G) > 0 else 0
            else:
                max_out_deg = max(dict(G.degree()).values()) if len(G) > 0 else 0

            top_pr_node = node_metrics_df.index[0] if not node_metrics_df.empty else "Brak"
            max_pr_val = node_metrics_df['pagerank'].iloc[0] if not node_metrics_df.empty else 0.0

            granger_vals = [float(d.get('granger', 0.0)) for _, _, d in G.edges(data=True)]
            mi_vals = [float(d.get('mutual_info', 0.0)) for _, _, d in G.edges(data=True)]

            sum_granger = sum(granger_vals)
            avg_granger = np.mean(granger_vals) if granger_vals else 0.0

            sum_mi = sum(mi_vals)
            avg_mi = np.mean(mi_vals) if mi_vals else 0.0

            metrics["dataset_name"] = dataset_name
            metrics["experiment_params"] = param_folder
            metrics["window_size"] = window_size
            metrics["threshold"] = threshold
            metrics["diameter"] = diameter
            metrics["avg_shortest_path"] = avg_shortest_path
            metrics["scale_free_slope"] = slope
            metrics["scale_free_R2"] = r_squared
            metrics["max_out_degree_lead"] = max_out_deg
            metrics["top_pagerank_node"] = top_pr_node
            metrics["max_pagerank_val"] = max_pr_val
            metrics["sum_granger"] = sum_granger
            metrics["avg_granger"] = avg_granger
            metrics["sum_mutual_info"] = sum_mi
            metrics["avg_mutual_info"] = avg_mi

            metrics.update(kendall_corrs)
            metrics.update({k: ", ".join(v) for k, v in top_n.items()})

            summary.append(metrics)

            pd.DataFrame([metrics]).to_csv(f"{out_dir}/metrics.csv", index=False)
            with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)

            node_metrics_df.to_csv(f"{out_dir}/node_metrics_centrality.csv", index_label="Node")

            with open(f"{out_dir}/communities.txt", "w", encoding="utf-8") as f:
                for i, c in enumerate(comms):
                    f.write(f"Community {i}: {list(c)}\n")

            plot_dual_view(G, comms, f"{out_dir}/graph_dual.png", f"{dataset_name} | {param_folder}")
            save_degree_distribution(G, out_dir, f"{dataset_name}_{param_folder}")
            plot_betweenness_centrality(G, out_dir, f"{dataset_name}_{param_folder}")
            plot_closeness_centrality(G, out_dir, f"{dataset_name}_{param_folder}")

    if summary:
        final_csv_path = os.path.join(MASTER_OUTPUT_DIR, "summary_all.csv")
        df_summary = pd.DataFrame(summary)
        cols = ['dataset_name', 'experiment_params', 'window_size', 'threshold'] + \
               [c for c in df_summary.columns if
                c not in ['dataset_name', 'experiment_params', 'window_size', 'threshold']]
        df_summary = df_summary[cols]

        df_summary.to_csv(final_csv_path, index=False)
        print(f"\nZakończono sukcesem. Utworzono globalny plik z wynikami w: {final_csv_path}")
    else:
        print("\nNie znaleziono żadnych danych do podsumowania.")

if __name__ == "__main__":
    main()