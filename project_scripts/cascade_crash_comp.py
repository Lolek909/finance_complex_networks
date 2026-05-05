import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from src.cascade_logic import cascade_fire_sales, initial_shock


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CRASH_FILE = os.path.join(
    PROJECT_ROOT,
    "data_finance",
    "comparison",
    "after_amzn_crash_2022-04-29.csv"
)

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "crash",
    "crash_network_before_amzn.pkl"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "crash",
    "compare"
)



def compute_real_drops_pct(df):
    returns = df.pct_change().dropna()

    last_move = returns.iloc[-1]

    return last_move.values * 100, list(df.columns)


def plot_compare(G, nodes, cascade_state, real_pct, out_path):
    pos = nx.spring_layout(G, seed=42)

    cascade_nodes = [nodes[i] for i in range(len(nodes)) if cascade_state[i] == 1]

    cascade_healthy = [n for n in nodes if n not in cascade_nodes]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))


    ax = axes[0]

    nx.draw_networkx_edges(
        G, pos,
        alpha=0.25,
        width=0.8,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=cascade_healthy,
        node_color="royalblue",
        node_size=120,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=cascade_nodes,
        node_color="red",
        node_size=180,
        ax=ax
    )

    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    ax.set_title("CASCADE (fire-sales / contagion model)")
    ax.axis("off")

    ax = axes[1]

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_color="lightgrey",
        node_size=110,
        ax=ax,
        alpha=0.4
    )

    down_nodes = []
    up_nodes = []

    for i, n in enumerate(nodes):
        if real_pct[i] < 0:
            down_nodes.append(n)
        else:
            up_nodes.append(n)

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=down_nodes,
        node_color="red",
        node_size=180,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=up_nodes,
        node_color="royalblue",
        node_size=120,
        ax=ax
    )

    labels = {
        n: f"{n}\n{real_pct[i]:.1f}%"
        for i, n in enumerate(nodes)
    }

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=6,
        ax=ax
    )

    ax.set_title("REAL market reaction (1-day move %)")
    ax.axis("off")

    plt.suptitle("Cascade vs Real Financial Contagion", fontsize=15)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading graph...")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    nodes = list(G.nodes)

    print("Loading real market data...")
    df = pd.read_csv(CRASH_FILE, index_col=0, parse_dates=True)

    real_pct, nodes_real = compute_real_drops_pct(df)

    real_map = dict(zip(nodes_real, real_pct))
    real_vector = np.array([real_map.get(n, 0) for n in nodes])

    print("Running cascade...")
    shock, _ = initial_shock(G, fixed_nodes=["AMZN"])

    history, _ = cascade_fire_sales(
        G,
        nodes,
        shock,
        theta=0.4,
        alpha=0.6,
        max_iter=20
    )

    cascade_state = history[-1]

    out_path = os.path.join(OUT_DIR, "cascade_vs_real.png")

    plot_compare(G, nodes, cascade_state, real_vector, out_path)

    real_binary = (real_vector < -1).astype(int)

    intersection = np.sum((cascade_state == 1) & (real_binary == 1))

    union = np.sum((cascade_state == 1) | (real_binary == 1))

    iou = intersection / (union + 1e-8)

    dice = (2 * intersection) / (cascade_state.sum() + real_binary.sum() + 1e-8)

    print("\nRESULTS:")
    print("Cascade size:", cascade_state.sum())
    print("Real negative moves:", np.sum(real_vector < 0))
    print("dice (cascade vs real drops):", dice)
    print("iou (cascade vs real drops):", iou)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()