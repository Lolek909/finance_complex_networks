from data_downloading import data, sector_map
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

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


# 5. ZOPTYMALIZOWANA SYMULACJA CZASU I RANKING
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
G_final = build_network(last_window, threshold=0.25)  # Wyższy próg dla czytelności grafu

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