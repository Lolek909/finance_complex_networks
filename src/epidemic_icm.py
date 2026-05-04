import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

matplotlib.use('Agg')

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics.IndependentCascadesModel as icm

BASE_COLORS = {
    'Tech': '#1f77b4',
    'Finance': '#ff7f0e',
    'Energy': '#2ca02c',
    'Healthcare': '#d62728',
}

# ICM status codes: 0 = susceptible, 1 = infected, 2 = recovered (trwale, nie zaraża)
STATUS_COLORS = {
    0: None,   # kolor sektorowy
    1: 'red',
    2: '#8B0000',  # ciemnoczerwony - odporny/recovered
}


def _build_icm_model(G, seeds, infection_prob):
    """
    Buduje i inicjalizuje model ICM z podanymi seedami.
    infection_prob: float (globalny) lub dict {(u,v): prob} per krawędź.
    """
    cfg = mc.Configuration()

    if isinstance(infection_prob, dict):
        for (u, v), p in infection_prob.items():
            if G.has_edge(u, v):
                cfg.add_edge_configuration("threshold", (u, v), float(p))
        # dla krawędzi bez podanego prob ustaw domyślne 0.1
        for u, v in G.edges():
            if (u, v) not in infection_prob:
                cfg.add_edge_configuration("threshold", (u, v), 0.1)
    else:
        for u, v in G.edges():
            cfg.add_edge_configuration("threshold", (u, v), float(infection_prob))

    cfg.add_model_parameter("fraction_infected", 0.0)

    model = icm(G)
    model.set_initial_status(cfg)

    # Ręczne ustawienie seedów (status 1 = infected)
    initial_status = {n: 0 for n in G.nodes()}
    for s in seeds:
        if s in initial_status:
            initial_status[s] = 1
    model.status = initial_status
    model.initial_status = initial_status.copy()

    return model


def _weights_to_probs(G, max_prob=0.5):
    """
    Konwertuje wagi krawędzi grafu Grangera na prawdopodobieństwa infekcji ICM.
    Normalizuje do zakresu [0.05, max_prob].
    """
    weights = {(u, v): abs(d.get("weight", 0.0)) for u, v, d in G.edges(data=True)}
    if not weights:
        return {}
    w_max = max(weights.values()) or 1.0
    return {
        (u, v): max(0.05, (w / w_max) * max_prob)
        for (u, v), w in weights.items()
    }


def run_epidemic_simulation_icm(G, raw_prices_window, sector_map, out_dir, name,
                                 threshold=-0.05, infection_prob=None, max_icm_steps=5):
    """
    Symuluje kaskadę spadków łącząc ICM (ndlib) z rzeczywistymi danymi cenowymi.

    Mechanizm:
    - t=0: węzły z ceną <= threshold stają się seedami ICM
    - t=1..T: jeden krok ICM propaguje infekcję po krawędziach grafu;
              węzły których zwrot przekroczy próg w danym t są dodatkowo
              wstrzykiwane jako nowe źródła (re-seeding)
    - Wagi krawędzi z grafu Grangera są normalizowane do prawdopodobieństw infekcji

    Parametry
    ----------
    G : nx.DiGraph
        Graf z wagami krawędzi (np. z analizy Grangera).
    raw_prices_window : pd.DataFrame
        Okno cen (wiersze = czas, kolumny = tickery).
    sector_map : dict
        Mapa ticker -> sektor.
    out_dir : str
        Katalog wyjściowy.
    name : str
        Nazwa konfiguracji (używana w nazwach plików).
    threshold : float
        Próg zwrotu poniżej którego węzeł jest zainfekowany (np. -0.05 = -5%).
    infection_prob : float | None
        Globalne prawdopodobieństwo infekcji per krawędź. None = użyj wag grafu.
    max_icm_steps : int | None
        Maksymalna liczba kroków ICM. None = len(raw_prices_window).
    """
    sim_dir = os.path.join(out_dir, f"epidemic_icm_{name}")
    os.makedirs(sim_dir, exist_ok=True)

    V_ts = raw_prices_window.iloc[0]
    W = (raw_prices_window - V_ts) / V_ts
    T = max_icm_steps if max_icm_steps is not None else len(W)
    T = min(T, len(W))

    min_return = W.min().min()
    print(f" -> Max spadek w oknie: {min_return*100:.2f}%. Próg: {threshold*100:.2f}%")

    if infection_prob is None:
        edge_probs = _weights_to_probs(G)
    else:
        edge_probs = float(infection_prob)

    # Seedy: węzły które jako pierwsze przekraczają próg (tak jak w modelu deterministycznym)
    seeds = []
    first_infected_t = 0
    for t_idx in range(len(W)):
        current_returns = W.iloc[t_idx]
        infected_at_t = [
            n for n in G.nodes()
            if n in current_returns and pd.notna(current_returns[n]) and current_returns[n] <= threshold
        ]
        if infected_at_t:
            seeds = infected_at_t
            first_infected_t = t_idx
            break
    print(f" -> Seedów ICM (t={first_infected_t}): {len(seeds)} węzłów: {seeds}")

    model = _build_icm_model(G, seeds, edge_probs)

    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=100)
    edge_widths = [abs(d.get("weight", 0.1)) * 3 for _, _, d in edges]
    edgelist = [(u, v) for u, v, _ in edges]

    # Zbieramy stan statusów dla każdej klatki z góry (ICM jest jednokierunkowy)
    # Klatki przed first_infected_t: wszystkie węzły zdrowe
    frames_status = [{n: 0 for n in nodes} for _ in range(first_infected_t)]
    for t_idx in range(first_infected_t, T):
        # Re-seeding: węzły które właśnie przekroczyły próg cenowy w tym kroku
        current_returns = W.iloc[t_idx]
        for n in nodes:
            if (n in current_returns and pd.notna(current_returns[n])
                    and current_returns[n] <= threshold
                    and model.status.get(n, 0) == 0):
                model.status[n] = 1

        frames_status.append(model.status.copy())

        if t_idx < T - 1:
            model.iteration()

    infected_series = [sum(1 for s in fs.values() if s == 1) for fs in frames_status]
    recovered_series = [sum(1 for s in fs.values() if s == 2) for fs in frames_status]
    print(f" -> Max zainfekowanych: {max(infected_series)}, max recovered: {max(recovered_series)}")
    print(f" -> Budowanie animacji ({T} klatek)...")

    fig, ax = plt.subplots(figsize=(12, 9))

    def update(t_idx):
        ax.clear()
        current_time = W.index[t_idx]
        status = frames_status[t_idx]

        node_colors = []
        infected_labels = {}
        inf_count = 0
        rec_count = 0

        for n in nodes:
            s = status.get(n, 0)
            if s == 1:
                node_colors.append('red')
                infected_labels[n] = n
                inf_count += 1
            elif s == 2:
                node_colors.append('#8B0000')
                rec_count += 1
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
            f"ICM Epidemic: {name} | Próg: {threshold*100:.1f}%\n"
            f"Czas: {current_time} | Zainfekowane: {inf_count} | Recovered: {rec_count} | Zdrowe: {len(nodes)-inf_count-rec_count}",
            fontsize=13)
        ax.axis('off')

    anim = FuncAnimation(fig, update, frames=T, interval=200)

    gif_path = os.path.join(sim_dir, f"animation_icm_{name}.gif")
    try:
        anim.save(gif_path, writer="pillow", dpi=100)
        print(f" -> Zapisano animację ICM GIF w: {gif_path}")
    except Exception as e:
        print(f" -> Nie udało się zapisać animacji: {e}")
    finally:
        plt.close(fig)

    return sim_dir
