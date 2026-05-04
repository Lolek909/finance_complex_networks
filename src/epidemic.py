import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

matplotlib.use('Agg')


BASE_COLORS = {
    'Tech': '#1f77b4',
    'Finance': '#ff7f0e',
    'Energy': '#2ca02c',
    'Healthcare': '#d62728',
}


def run_epidemic_simulation(G, raw_prices_window, sector_map, out_dir, name, threshold=-0.05):
    """
    Symuluje kaskadę spadków na podstawie rzeczywistych cen i grafu powiązań.
    Tworzy animację GIF przy użyciu matplotlib.animation (bez zapisu klatek PNG).
    """
    sim_dir = os.path.join(out_dir, f"epidemic_{name}")
    os.makedirs(sim_dir, exist_ok=True)

    V_ts = raw_prices_window.iloc[0]
    W = (raw_prices_window - V_ts) / V_ts

    min_return = W.min().min()
    print(f" -> Max spadek w oknie: {min_return*100:.2f}%. Wymagany próg: {threshold*100:.2f}%")
    print(f" -> Start symulacji epidemicznej: {name} (Klatek: {len(W)})")

    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=100)
    edge_widths = [abs(d.get("weight", 0.1)) * 3 for _, _, d in edges]
    edgelist = [(u, v) for u, v, _ in edges]

    fig, ax = plt.subplots(figsize=(12, 9))

    def update(t_idx):
        ax.clear()
        current_time = W.index[t_idx]
        current_returns = W.iloc[t_idx]

        node_colors = []
        infected_count = 0
        infected_labels = {}

        for n in nodes:
            if n in current_returns and pd.notna(current_returns[n]) and current_returns[n] <= threshold:
                node_colors.append('red')
                infected_count += 1
                infected_labels[n] = n
            else:
                sector = sector_map.get(n, "Unknown")
                node_colors.append(BASE_COLORS.get(sector, 'lightgrey'))

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=150,
                               edgecolors='black', linewidths=0.5)
        if edgelist:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edgelist, width=edge_widths,
                                   alpha=0.3, edge_color='grey', arrows=True, arrowsize=10)
        nx.draw_networkx_labels(G, pos, ax=ax, labels=infected_labels, font_size=9, font_weight="bold")

        ax.set_title(
            f"Epidemic Model: {name} | Threshold: {threshold * 100:.1f}%\n"
            f"Czas: {current_time} | Zainfekowane: {infected_count}/{len(nodes)}",
            fontsize=14)
        ax.axis('off')

    anim = FuncAnimation(fig, update, frames=len(W), interval=200)

    gif_path = os.path.join(sim_dir, f"animation_cascade_{name}.gif")
    try:
        anim.save(gif_path, writer="pillow", dpi=100)
        print(f" -> Zapisano animację GIF w: {gif_path}")
    except Exception as e:
        print(f" -> Nie udało się zapisać animacji: {e}")
    finally:
        plt.close(fig)

    return sim_dir