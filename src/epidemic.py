import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from PIL import Image


def run_epidemic_simulation(G, raw_prices_window, sector_map, out_dir, name, threshold=-0.05):
    """
    Symuluje kaskadę spadków na podstawie rzeczywistych cen i grafu powiązań.
    Tworzy klatki (.png) i animację (.gif).
    """
    sim_dir = os.path.join(out_dir, f"epidemic_{name}")
    os.makedirs(sim_dir, exist_ok=True)

    V_ts = raw_prices_window.iloc[0]
    W = (raw_prices_window - V_ts) / V_ts

    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=100)

    base_colors = {
        'Tech': '#1f77b4',
        'Finance': '#ff7f0e',
        'Energy': '#2ca02c',
        'Healthcare': '#d62728',
        'Consumer': '#9467bd'
    }

    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    frame_paths = []

    print(f"Start symulacji epidemicznej: {name} (Klatek: {len(W)})")

    for t_idx in range(len(W)):
        current_time = W.index[t_idx]
        current_returns = W.iloc[t_idx]

        plt.figure(figsize=(12, 9))

        node_colors = []
        infected_count = 0

        for n in nodes:
            if n in current_returns and current_returns[n] <= threshold:
                node_colors.append('red')  # Zainfekowana
                infected_count += 1
            else:
                sector = sector_map.get(n, "Unknown")
                node_colors.append(base_colors.get(sector, 'lightgrey'))

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=150, edgecolors='black', linewidths=0.5)

        if edges:
            widths = [abs(d.get("weight", 0.1)) * 3 for _, _, d in edges]
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges],
                                   width=widths, alpha=0.3, edge_color='grey', arrows=True, arrowsize=10)

        labels = {n: n for n in nodes if n in current_returns and current_returns[n] <= threshold}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")

        plt.title(
            f"Epidemic Model: {name} | Threshold: {threshold * 100}%\nCzas: {current_time} | Zainfekowane: {infected_count}/{len(nodes)}",
            fontsize=14)
        plt.axis('off')

        # Zapis klatki
        frame_path = os.path.join(sim_dir, f"frame_{t_idx:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100)
        plt.close()
        frame_paths.append(frame_path)

    try:
        images = [Image.open(f) for f in frame_paths]
        gif_path = os.path.join(sim_dir, f"animation_cascade_{name}.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
        print(f"Zapisano animację GIF w: {gif_path}")
    except Exception as e:
        print(f"Nie udało się utworzyć GIFa: {e}")

    return sim_dir