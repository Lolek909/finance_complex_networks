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

from src.utils import load_graphs

tickers = {
    'Tech': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'ADBE', 'CRM', 'CSCO', 'ORCL', 'IBM', 'QCOM', 'TXN', 'AVGO', 'NOW', 'INTU', 'AMAT'
    ],
    'Finance': [
        'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA', 'AXP', 'PYPL', 'SPGI', 'BLK', 'SCHW', 'CB', 'CME', 'PGR', 'USB', 'PNC', 'TFC', 'COF'
    ],
    'Energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'VLO', 'MPC', 'PSX', 'EOG', 'OXY', 'HAL', 'KMI', 'WMB', 'BKR', 'FANG', 'DVN', 'TRGP', 'CTRA', 'EQT', 'APA', 'NOV'
    ],
    'Healthcare': [
        'JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'DHR', 'ISRG', 'CVS', 'MDT', 'SYK', 'CI', 'VRTX', 'REGN', 'BDX', 'BSX', 'HUM', 'AMGN', 'GILD'
    ],
    'Consumer': [
        'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG', 'MAR', 'HLT', 'F', 'GM', 'CMG', 'DG', 'ORLY'
    ]
}

sector_map = {ticker: sector for sector, t_list in tickers.items() for ticker in t_list}


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
        "Consumer": 4,
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
        width=[d.get("weight", 0.1) * 3 for _, _, d in edges],
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
    pagerank = nx.pagerank(G)

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


def compare_wta_mst(G, out_dir, name, percentile=85):
    edges = list(G.edges(data=True))
    if not edges: return 0

    weights = [abs(d.get("weight", 0)) for _, _, d in edges]
    threshold_wta = np.percentile(weights, percentile)

    G_wta = nx.DiGraph()
    G_wta.add_nodes_from(G.nodes())
    wta_edges = [(u, v, d) for u, v, d in edges if abs(d.get("weight", 0)) >= threshold_wta]
    G_wta.add_edges_from(wta_edges)

    G_undirected = G.to_undirected()
    H = nx.Graph()

    for u, v, d in G_undirected.edges(data=True):
        w = d.get("weight", 0)
        dist = np.sqrt(2 * (1 - abs(min(max(w, -1.0), 1.0))))
        H.add_edge(u, v, weight=dist, orig_w=w)

    G_mst = nx.minimum_spanning_tree(H, weight='weight')

    wta_set = set([tuple(sorted((u, v))) for u, v in G_wta.edges()])
    mst_set = set([tuple(sorted((u, v))) for u, v in G_mst.edges()])
    overlap = wta_set.intersection(mst_set)

    with open(f"{out_dir}/wta_mst_comparison.txt", "w", encoding="utf-8") as f:
        f.write(f"WTA Edges (Top {100 - percentile}% threshold >= {threshold_wta:.3f}): {G_wta.number_of_edges()}\n")
        f.write(f"MST Edges (Distance based): {G_mst.number_of_edges()}\n")
        f.write(f"Core Overlap (Wspólne krawędzie w obu strukturach): {len(overlap)}\n")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pos = nx.spring_layout(G_undirected, seed=42)

    nx.draw_networkx_nodes(G_wta, pos, ax=axes[0], node_size=40, node_color="blue", alpha=0.6)
    nx.draw_networkx_edges(G_wta, pos, ax=axes[0], edge_color="blue", alpha=0.3)
    axes[0].set_title(f"Winner-Takes-All (Top {100 - percentile}%)")
    axes[0].axis("off")

    nx.draw_networkx_nodes(G_mst, pos, ax=axes[1], node_size=40, node_color="red", alpha=0.6)
    nx.draw_networkx_edges(G_mst, pos, ax=axes[1], edge_color="red", alpha=0.5)
    axes[1].set_title("Minimum Spanning Tree")
    axes[1].axis("off")

    plt.suptitle(f"WTA vs MST Topology Comparison: {name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/wta_vs_mst_visual.png")
    plt.close()

    return len(overlap)


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
    graphs = load_graphs()
    summary = []

    for name, G, meta in graphs:
        print(f"Processing {name}")

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print(f"Pominięto {name} - pusty graf (brak węzłów lub krawędzi)")
            continue

        if G.is_directed():
            largest_cc = max(nx.weakly_connected_components(G), key=len)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

        from src.utils import OUTPUT_DIR
        out_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        comms, metrics = analyze(G, meta)

        diameter, avg_shortest_path = get_diameter_and_shortest_path(G)
        kendall_corrs = calculate_kendall_tau(G)
        top_n = get_top_nodes(G, n=5)

        slope, r_squared = prove_scale_free_network(G, out_dir, name)
        mst_wta_overlap = compare_wta_mst(G, out_dir, name, percentile=85)

        metrics["experiment"] = name
        metrics["window_size"] = meta.get("window_size")
        metrics["threshold"] = meta.get("threshold")
        metrics["diameter"] = diameter
        metrics["avg_shortest_path"] = avg_shortest_path
        metrics["scale_free_slope"] = slope
        metrics["scale_free_R2"] = r_squared
        metrics["wta_mst_overlap"] = mst_wta_overlap

        metrics.update(kendall_corrs)
        metrics.update({k: ", ".join(v) for k, v in top_n.items()})  # Konwersja list na stringi dla CSV

        summary.append(metrics)

        pd.DataFrame([metrics]).to_csv(f"{out_dir}/metrics.csv", index=False)
        with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        node_metrics_df = get_node_metrics_df(G)
        node_metrics_df.to_csv(f"{out_dir}/node_metrics_centrality.csv", index_label="Node")

        with open(f"{out_dir}/communities.txt", "w", encoding="utf-8") as f:
            for i, c in enumerate(comms):
                f.write(f"Community {i}: {list(c)}\n")

        plot_dual_view(G, comms, f"{out_dir}/graph_dual.png", name)
        save_degree_distribution(G, out_dir, name)

        plot_betweenness_centrality(G, out_dir, name)
        plot_closeness_centrality(G, out_dir, name)

    pd.DataFrame(summary).to_csv(f"{OUTPUT_DIR}/summary_all.csv", index=False)

if __name__ == "__main__":
    main()