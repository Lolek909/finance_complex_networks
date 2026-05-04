import matplotlib
matplotlib.use("Agg")
import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
script_dir = os.path.dirname(os.path.abspath(__file__))
from src.cascade_logic import (
    load_graph, initial_shock, cascade_fire_sales, 
    compute_metrics, save_gif
)
BASE_DIR = os.path.join(script_dir, "../results/grid_search")
OUT_DIR = os.path.join(script_dir, "../results/analysis")

os.makedirs(OUT_DIR, exist_ok=True)


def process_one(graph_path, rel_folder):
    print(f"\nProcessing: {rel_folder}", flush=True)

    G = load_graph(graph_path)
    nodes = list(G.nodes)

    shock, nodes = initial_shock(G, method="fixed")

    history, A = cascade_fire_sales(G, nodes, shock)

    metrics = compute_metrics(history, A)

    out_folder = os.path.join(OUT_DIR, rel_folder, "cascade")
    os.makedirs(out_folder, exist_ok=True)

    gif_path = os.path.join(out_folder, "cascade.gif")

    save_gif(history, G, nodes, shock, gif_path)

    np.save(os.path.join(out_folder, "history.npy"), history)
    np.save(os.path.join(out_folder, "adj.npy"), A)

    with open(os.path.join(out_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return rel_folder, metrics
    
def run():

    tasks = []

    for root, dirs, files in os.walk(BASE_DIR):

        if "graph.pkl" not in files:
            continue

        graph_path = os.path.join(root, "graph.pkl")
        rel_folder = os.path.relpath(root, BASE_DIR)

        tasks.append((graph_path, rel_folder))

    print(f"Total tasks: {len(tasks)}", flush=True)

    with ProcessPoolExecutor() as executor:

        futures = [
            executor.submit(process_one, gp, rf)
            for gp, rf in tasks
        ]

        for f in as_completed(futures):
            rel_folder, metrics = f.result()
            print(f"Done: {rel_folder}", flush=True)
            print(metrics, flush=True)
if __name__ == "__main__":
    run()