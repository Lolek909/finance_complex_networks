import os
import re
import networkx as nx
from src.data import get_data, sector_map
from src.epidemic import run_epidemic_simulation
from src.epidemic_icm import run_epidemic_simulation_icm




def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results", "grid_search")

    if not os.path.exists(results_dir):
        print(f"Katalog wyników nie istnieje: {results_dir}")
        return

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
            dataset_path = os.path.join(results_dir, dataset_name)

            if not os.path.exists(dataset_path):
                continue

            print(f"\nSzukanie grafów dla: {dataset_name.upper()}")

            try:
                if has_volume:
                    raw_data = get_data(period=period, interval=interval, volume=True)
                    prices = raw_data['Close']
                else:
                    prices = get_data(period=period, interval=interval, volume=False)
            except Exception as e:
                print(f"Błąd pobierania danych dla {dataset_name}: {e}. Pomijam.")
                continue

            for config_folder in os.listdir(dataset_path):
                config_path = os.path.join(dataset_path, config_folder)
                if not os.path.isdir(config_path):
                    continue

                match = re.search(r'w(\d+)_t', config_folder)
                if not match:
                    continue
                window_size = int(match.group(1))

                graph_file = None
                for file in os.listdir(config_path):
                    if file.endswith(".gexf") and "mst" not in file.lower():
                        graph_file = os.path.join(config_path, file)
                        break

                if not graph_file:
                    print(f"Nie znaleziono pliku .gexf dla {config_folder}. Pomijam.")
                    continue

                print(f"Wczytywanie grafu: {graph_file}")
                try:
                    G = nx.read_gexf(graph_file)

                    for u, v, data in G.edges(data=True):
                        if 'weight' in data:
                            data['weight'] = float(data['weight'])

                    raw_win_price = prices.iloc[-window_size:]

                    if interval == "5m":
                        dynamic_threshold = -0.002
                    elif interval == "1h":
                        dynamic_threshold = -0.01
                    else:
                        dynamic_threshold = -0.04

                    # run_epidemic_simulation(
                    #     G=G,
                    #     raw_prices_window=raw_win_price,
                    #     sector_map=sector_map,
                    #     out_dir=config_path,
                    #     name=config_folder,
                    #     threshold=dynamic_threshold
                    # )
                    run_epidemic_simulation_icm(
                        G=G,
                        raw_prices_window=raw_win_price,
                        sector_map=sector_map,
                        out_dir=config_path,
                        name=config_folder,
                        threshold=dynamic_threshold,
                        # infection_prob=None  →  użyje wag z grafu automatycznie
                    )

                except Exception as e:
                    print(f"Błąd podczas symulacji dla {config_folder}: {e}")


if __name__ == '__main__':
    main()