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

    for i, t_a in enumerate(nodes):
        for j, t_b in enumerate(nodes):
            if i == j: continue

            vol_a = vol_window[t_a] if vol_window is not None else None
            weight, lag, granger, mi, vp = evaluate_pair_unified(price_window[t_a], price_window[t_b], vol_a)

            if abs(weight) > threshold:
                if vol_window is not None:
                    G.add_edge(t_a, t_b, weight=weight, lag=lag, granger=granger, mutual_info=mi, vol_price=vp)
                else:
                    G.add_edge(t_a, t_b, weight=weight, lag=lag, granger=granger)
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

    volume_modes = [True, False]

    for config in datasets:
        for has_volume in volume_modes:
            period = config["period"]
            interval = config["interval"]

            mode_suffix = "with_vol" if has_volume else "no_vol"
            dataset_name = f"{interval}_{period}_{mode_suffix}"

            print(f"\nPrzetwarzanie danych: {dataset_name.upper()}")

            try:
                if has_volume:
                    raw_data = get_data(period=period, interval=interval, volume=True)
                    prices = raw_data['Close']
                    volumes = raw_data['Volume']
                else:
                    prices = get_data(period=period, interval=interval, volume=False)
                    volumes = None
            except Exception as e:
                print(f"Błąd wczytywania danych dla {dataset_name}: {e}. Pomijam.")
                continue

            print("Obliczanie przekształceń...")
            log_returns = process_data(prices)
            log_volumes = process_data(volumes) if volumes is not None else None

            print("Redukcja wpływu całego sektora na korelacje...")
            log_returns = sector_impact_reduction(log_returns, sector_map)
            if log_volumes is not None:
                log_volumes = sector_impact_reduction(log_volumes, sector_map)

            new_out_dir = os.path.join(project_root, "results", "grid_search", dataset_name)
            os.makedirs(new_out_dir, exist_ok=True)
            src.grid_search.OUTPUT_DIR = new_out_dir

            current_thresholds = [0.2, 0.3, 0.4]

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