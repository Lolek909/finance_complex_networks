import os
import numpy as np
import networkx as nx
from src.data import get_data, sector_map
from src.impact_reduction import sector_impact_reduction
import src.grid_search
from src.metrics import evaluate_pair_unified


def build_network_unified(price_window, vol_window=None, threshold=0.30):
    G = nx.DiGraph()
    nodes = price_window.columns
    G.add_nodes_from(nodes)

    temp_edges = []

    for i, t_a in enumerate(nodes):
        for j, t_b in enumerate(nodes):
            if i == j: continue

            vol_a = vol_window[t_a] if vol_window is not None else None
            weight, lag, granger, mi, vp = evaluate_pair_unified(price_window[t_a], price_window[t_b], vol_a)

            if weight > threshold:
                temp_edges.append({
                    "u": t_a, "v": t_b, "weight": weight,
                    "lag": lag, "granger": granger, "mutual_info": mi, "vol_price": vp
                })

    if not temp_edges:
        return G

    raw_weights = [edge["weight"] for edge in temp_edges]
    min_w = min(raw_weights)
    max_w = max(raw_weights)
    range_w = max_w - min_w if max_w > min_w else 1e-9

    for edge in temp_edges:
        scaled_weight = (edge["weight"] - min_w) / range_w

        if vol_window is not None:
            G.add_edge(edge["u"], edge["v"],
                       weight=scaled_weight,
                       raw_weight=edge["weight"],
                       lag=edge["lag"],
                       granger=edge["granger"],
                       mutual_info=edge["mutual_info"],
                       vol_price=edge["vol_price"])
        else:
            G.add_edge(edge["u"], edge["v"],
                       weight=scaled_weight,
                       raw_weight=edge["weight"],
                       lag=edge["lag"])
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    datasets = [
        {"period": "5d", "interval": "5m"},
        {"period": "1mo", "interval": "1h"},
        {"period": "1y", "interval": "1d"}
    ]

    for config in datasets:
        period = config["period"]
        interval = config["interval"]
        dataset_name = f"{interval}_{period}"

        print(f"\nPrzetwarzanie danych: {dataset_name.upper()}")

        has_volume = True
        try:
            raw_data = get_data(period=period, interval=interval, volume=True)
            prices = raw_data['Close']
            volumes = raw_data['Volume']
        except Exception as e:
            print(f"Brak wolumenu (lub błąd: {e}). Przełączanie na klasyczną korelację...")
            has_volume = False
            try:
                prices = get_data(period=period, interval=interval, volume=False)
                volumes = None
            except Exception as e2:
                print(f"Całkowity błąd wczytywania danych dla {dataset_name}: {e2}")
                continue

        print("Obliczanie przekształceń...")
        log_returns = process_data(prices)
        log_volumes = process_data(volumes)

        print("Redukcja wpływu całego sektora na korelacje...")
        log_returns = sector_impact_reduction(log_returns, sector_map)

        new_out_dir = os.path.join(project_root, "results", "grid_search", dataset_name)
        os.makedirs(new_out_dir, exist_ok=True)
        src.grid_search.OUTPUT_DIR = new_out_dir

        current_thresholds = [0.5, 1.0, 1.5] if has_volume else [0.2, 0.3, 0.4]

        print(f"Uruchamianie Grid Search. Tryb Wolumenu: {has_volume}")

        src.grid_search.grid_search(
            log_returns,
            build_network_unified,
            evaluate_pair_unified,
            sector_map,
            thresholds=current_thresholds,
            window_sizes=[30, 60, 120],
            step=15,
            log_volumes=log_volumes,
            raw_prices=prices
        )


if __name__ == '__main__':
    main()