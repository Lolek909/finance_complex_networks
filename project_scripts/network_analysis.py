import os
import pickle
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community import louvain_communities


INPUT_DIR = "../results/grid_search"
OUTPUT_DIR = "../results/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


sector_map = {
    'AAPL':'Tech','MSFT':'Tech','GOOGL':'Tech','AMZN':'Tech','NVDA':'Tech','META':'Tech','TSLA':'Tech','AMD':'Tech','INTC':'Tech','ADBE':'Tech',
    'JPM':'Finance','BAC':'Finance','GS':'Finance','MS':'Finance','WFC':'Finance','C':'Finance','V':'Finance','MA':'Finance','AXP':'Finance','PYPL':'Finance',
    'XOM':'Energy','CVX':'Energy','SHEL':'Energy','BP':'Energy','TTE':'Energy','COP':'Energy','SLB':'Energy','PBR':'Energy','EQNR':'Energy','VLO':'Energy',
    'JNJ':'Healthcare','UNH':'Healthcare','PFE':'Healthcare','ABBV':'Healthcare','LLY':'Healthcare','MRK':'Healthcare','TMO':'Healthcare','AZN':'Healthcare','NVO':'Healthcare','DHR':'Healthcare'
}


def load_graphs():
    graphs = []

    for folder in os.listdir(INPUT_DIR):
        path = os.path.join(INPUT_DIR, folder)

        g_path = os.path.join(path, "graph.pkl")
        m_path = os.path.join(path, "metrics.json")

        if not os.path.exists(g_path):
            continue

        with open(g_path, "rb") as f:
            G = pickle.load(f)

        meta = {}
        if os.path.exists(m_path):
            with open(m_path, "r") as f:
                meta = json.load(f)

        graphs.append((folder, G, meta))

    return graphs


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
        wdeg = sum(d.get("weight", 0) for _, _, d in G.edges(n, data=True))
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
        key=lambda x: x[2].get("weight", 0),
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
        width=[d.get("weight", 0.1) * 3 for _, _, d in edges],
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
    plt.figure(figsize=(7,5))
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
    degree = dict(G.degree())

    df = pd.DataFrame({
        "degree": pd.Series(degree),
        "betweenness": pd.Series(betweenness),
        "closeness": pd.Series(closeness),
        "clustering": pd.Series(clustering),
        "pagerank": pd.Series(pagerank)
    })

    return df.sort_values(by="pagerank", ascending=False)


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
    degree = dict(nx.degree(G))
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    return {
        "top_degree": sorted(degree, key=degree.get, reverse=True)[:n],
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

        out_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        comms, metrics = analyze(G, meta)

        diameter, avg_shortest_path = get_diameter_and_shortest_path(G)
        kendall_corrs = calculate_kendall_tau(G)
        top_n = get_top_nodes(G, n=5)

        metrics["experiment"] = name
        metrics["window_size"] = meta.get("window_size")
        metrics["threshold"] = meta.get("threshold")
        metrics["diameter"] = diameter
        metrics["avg_shortest_path"] = avg_shortest_path
        metrics.update(kendall_corrs)
        metrics.update({k: ", ".join(v) for k, v in top_n.items()})  # Konwersja list na stringi dla CSV

        summary.append(metrics)

        pd.DataFrame([metrics]).to_csv(f"{out_dir}/metrics.csv", index=False)
        with open(f"{out_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        node_metrics_df = get_node_metrics_df(G)
        node_metrics_df.to_csv(f"{out_dir}/node_metrics_centrality.csv", index_label="Node")

        with open(f"{out_dir}/communities.txt", "w") as f:
            for i, c in enumerate(comms):
                f.write(f"Community {i}: {list(c)}\n")

        plot_dual_view(G, comms, f"{out_dir}/graph_dual.png", name)
        save_degree_distribution(G, out_dir, name)

        plot_betweenness_centrality(G, out_dir, name)
        plot_closeness_centrality(G, out_dir, name)

    pd.DataFrame(summary).to_csv(f"{OUTPUT_DIR}/summary_all.csv", index=False)

if __name__ == "__main__":
    main()