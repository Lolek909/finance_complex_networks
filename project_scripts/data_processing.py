import os
import numpy as np
import pandas as pd
import networkx as nx
from src.data import get_data, sector_map
from src.impact_reduction import sector_impact_reduction
from src.grid_search import grid_search


def safe_corr(a, b, lag):
    a_shifted = a.shift(lag)
    combined = pd.concat([a_shifted, b], axis=1)
    combined.columns = ['a', 'b']
    combined = combined.dropna()

    if len(combined) < 10:
        print("brak danych", flush=True)
        return 0

    if combined['a'].std() < 1e-9 or combined['b'].std() < 1e-9:
        return 0

    return combined['a'].corr(combined['b'])

def calculate_lead_lag_corr(series_a, series_b, max_lag=5):
    corrs = []
    for lag in range(1, max_lag + 1):
        c = series_a.shift(lag).corr(series_b)
        # c = safe_corr(series_a, series_b, lag)
        corrs.append(c)

    if not corrs or np.all(np.isnan(corrs)):
        return 0, 0

    max_c = np.nanmax(corrs)
    best_lag = np.nanargmax(corrs) + 1
    return (max_c, best_lag) if not np.isnan(max_c) else (0, 0)


def build_network(returns_window, threshold=0.30):
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


def process_data(data):
    log_returns = np.log(data / data.shift(1)).dropna()
    log_returns = log_returns.clip(
        log_returns.quantile(0.01),
        log_returns.quantile(0.99),
        axis=1
    )
    log_returns = log_returns.ewm(span=10).mean()
    return log_returns


def main():
    datasets = [
        {"period": "5d", "interval": "5m"},
        {"period": "1mo", "interval": "1h"},
        {"period": "1y", "interval": "1d"}
    ]

    for config in datasets:
        period = config["period"]
        interval = config["interval"]
        dataset_name = f"{interval}_{period}"

        print(f"Przetwarzanie danych: {dataset_name.upper()}")

        try:
            data = get_data(period=period, interval=interval, volume=False)
        except Exception as e:
            print(f"Nie udało się wczytać danych dla {dataset_name}: {e}")
            continue

        print("Obliczanie log-zwrotów i czyszczenie...")
        log_returns = process_data(data)

        print("Redukcja wpływu całego sektora na korelacje...")
        log_returns = sector_impact_reduction(log_returns, sector_map)

        new_out_dir = f"../results/grid_search/{dataset_name}"
        os.makedirs(new_out_dir, exist_ok=True)

        grid_search.OUTPUT_DIR = new_out_dir

        print(f"Uruchamianie Grid Search... (Zapis do {new_out_dir})")

        grid_search(
            log_returns,
            build_network,
            calculate_lead_lag_corr,
            sector_map,
            thresholds=[0.2, 0.3, 0.4],
            window_sizes=[30, 60, 120],
            step=15
        )

if __name__ == '__main__':
    main()