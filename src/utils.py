import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import pickle
import os
import json
import numpy as np

from src.data import sector_map

INPUT_DIR = "../results/grid_search"
OUTPUT_DIR = "../results/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_graph(G, path, title, sector_map):
    plt.figure(figsize=(16, 12))

    if G.number_of_nodes() == 0:
        plt.title("Pusty graf - brak powiązań", fontsize=16)
        plt.savefig(path)
        plt.close()
        return

    pos = nx.spring_layout(G, k=0.9, iterations=100, seed=42)

    color_map = {
        'Tech': '#1f77b4', 'Finance': '#ff7f0e',
        'Energy': '#2ca02c', 'Healthcare': '#d62728',
    }

    node_colors = [color_map.get(sector_map.get(n, ""), "grey") for n in G.nodes()]

    degrees = dict(G.degree())
    node_sizes = [400 + 150 * degrees[n] for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors='black',
        linewidths=1.5
    )

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    edges = list(G.edges(data=True))

    if edges:
        widths = [abs(d.get("weight", 0.1)) * 4 for _, _, d in edges]

        edge_colors = ['green' if d.get("weight", 0) > 0 else 'red' for _, _, d in edges]

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, _ in edges],
            width=widths,
            edge_color=edge_colors,
            alpha=0.3,
            arrows=True,
            arrowsize=15
        )

        weights_abs = [abs(d.get('weight', 0)) for _, _, d in edges]
        threshold_labels = np.percentile(weights_abs, 85) if len(weights_abs) > 0 else 0

        top_edges = {
            (u, v): f"{d.get('weight', 0):.2f}"
            for u, v, d in edges if abs(d.get('weight', 0)) >= threshold_labels
        }

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=top_edges,
            font_size=8,
            font_color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')  # Białe tło pod tekstem
        )

    plt.title(title, fontsize=20, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
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
