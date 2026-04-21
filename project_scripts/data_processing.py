from data_downloading import data, raw_data, sector_map
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import os

# 3. Obliczenia log-zwrotów
log_returns = np.log(data / data.shift(1)).dropna()


# 4. Funkcje sieciowe
def calculate_lead_lag_corr(series_a, series_b, max_lag=5):
    corrs = []
    for lag in range(1, max_lag + 1):
        # A(t-lag) wpływa na B(t)
        c = series_a.shift(lag).corr(series_b)
        corrs.append(c)

    if not corrs or np.all(np.isnan(corrs)):
        return 0, 0

    max_c = np.nanmax(corrs)
    best_lag = np.nanargmax(corrs) + 1
    return (max_c, best_lag) if not np.isnan(max_c) else (0, 0)


def build_network(returns_window, threshold=0.20):
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


# 5. Analiza okna (przykładowo ostatnie 60 minut dostępnych danych)
window_size = 60
if len(log_returns) > window_size:
    window_data = log_returns.tail(window_size)
    print(f"Analiza okna: {window_data.index[0]} do {window_data.index[-1]}")

    G = build_network(window_data, threshold=0.25)  # Zwiększ próg, by graf był czytelniejszy

    # Wizualizacja
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.4, iterations=100)

    color_map = {'Tech': '#1f77b4', 'Finance': '#ff7f0e', 'Energy': '#2ca02c', 'Healthcare': '#d62728'}
    node_colors = [color_map.get(sector_map.get(node), 'grey') for node in G.nodes()]

    # Rysowanie węzłów i etykiet
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    # Krawędzie
    edges = G.edges(data=True)
    if edges:
        weights = [d['weight'] * 5 for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray',
                               alpha=0.4, arrowsize=20, connectionstyle="arc3,rad=0.1")

    plt.title(
        "Sieć Zależności Lead-Lag (Dane 1m)\nSektory: Niebieski(Tech), Pomarańcz(Fin), Zielony(Energia), Czerwony(Health)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Zbyt mało danych do utworzenia okna.")