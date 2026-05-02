import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import pickle
import os
import json

from src.data import sector_map

INPUT_DIR = "./results/grid_search"
OUTPUT_DIR = "./results/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
