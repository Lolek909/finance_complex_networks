import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import json

BASE_COLORS = {
    'Tech': '#1f77b4',
    'Finance': '#ff7f0e',
    'Energy': '#bcbd22',
    'Healthcare': '#9467bd',
}

def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def initial_shock(G, shock_ratio=0.1, method="fixed", *, fixed_nodes = None):
    nodes = list(G.nodes)
    n = len(nodes)

    shock = np.zeros(n)

    if method == "random":
        idx = np.random.choice(n, int(n * shock_ratio), replace=False)

    elif method == "degree":
        deg = np.array([G.degree(n) for n in nodes])
        idx = np.argsort(deg)[-int(n * shock_ratio):]
    elif method == "fixed":
        fixed_nodes = [
            "XOM",
            "SHEL",
            "BP"
        ] if not fixed_nodes else fixed_nodes

        idx = [i for i, n in enumerate(nodes) if n in fixed_nodes]
    shock[idx] = 1
    return shock, nodes

def cascade_fire_sales(G, nodes, initial_state, theta=0.5, alpha=0.6, max_iter=20):
    # Default Cascade model (Amini et al.)
    # λ_ij - adj matrix
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")

    # X_i(t) ∈ {0,1}
    state = initial_state.copy()

    history = [state.copy()]

    for t in range(max_iter):

        # (Eq. loss propagation, 2.2)
        # loss_i(t) = Σ_j λ_ji X_j(t)
        # D* = { i : c_i < Σ_j (1 - R_ji) λ_ji }
        impact = A.T @ state

        # Fire sales price impact (2.1, inverse demand g)
        # p(t) ≈ g(Γ(t)/n)
        # Γ(t)/n ~ mean(defaulted exposure)
        price_impact = alpha * impact.mean()

        # c_i(t) < losses + fire-sale stress
        # i.e. Θ_i(t) = θ + market stress
        adjusted_threshold = theta + price_impact

        # X_i(t+1) = 1{ loss_i(t) > Θ_i(t) } OR X_i(t)
        new_state = np.maximum(
            state,
            (impact > adjusted_threshold).astype(int)
        )

        history.append(new_state.copy())

        # D_t = D_{t-1}
        if np.all(new_state == state):
            break

        state = new_state

    return np.array(history), A


def compute_metrics(history, A):

    final_state = history[-1]

    cascade_size = final_state.sum()
    cascade_ratio = cascade_size / len(final_state)

    steps = len(history)

    density_init = np.mean(A > 0)
    density_final = density_init * (1 - cascade_ratio)

    return {
        "cascade_size": int(cascade_size),
        "cascade_ratio": float(cascade_ratio),
        "steps_to_convergence": int(steps),
        "initial_density": float(density_init),
        "final_density_proxy": float(density_final),
    }

def save_gif(history, G, nodes, shock_vector, out_path):
    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=100)

    tickers = {
        'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'ADBE'],
        'Finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA', 'AXP', 'PYPL'],
        'Energy': ['XOM', 'CVX', 'SHEL', 'BP', 'TTE', 'COP', 'SLB', 'PBR', 'EQNR', 'VLO'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'AZN', 'NVO', 'DHR']
    }
    sector_map = {ticker: sector for sector, t_list in tickers.items() for ticker in t_list}

    base_node_colors = np.array([
        BASE_COLORS.get(sector_map.get(n, "Unknown"), "lightgrey") for n in nodes
    ])

    shocked_nodes = [nodes[i] for i in range(len(nodes)) if shock_vector[i] == 1]
    edge_weights = nx.get_edge_attributes(G, "weight")
    frames = []

    for t, state in enumerate(history):
        fig = plt.figure(figsize=(7, 7))
        
        state_bool = state.astype(bool)
        infected_nodelist = [nodes[i] for i, defaulted in enumerate(state_bool) if defaulted]
        healthy_nodelist = [nodes[i] for i, defaulted in enumerate(state_bool) if not defaulted]
        
        nx.draw_networkx_edges(
            G, pos, alpha=0.15,
            width=[2 * edge_weights.get((u, v), 0) for u, v in G.edges()]
        )

        if healthy_nodelist:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=healthy_nodelist,
                node_color=base_node_colors[~state_bool],
                node_size=150,
                alpha=0.8
            )

        if infected_nodelist:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=infected_nodelist,
                node_color="red",
                node_size=280,
                edgecolors="black",
                linewidths=1.5
            )

        nx.draw_networkx_labels(G, pos, font_size=7)

        if t == 0:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=shocked_nodes,
                node_color="#00FF00",
                node_size=350,
                edgecolors="black",
                linewidths=2.0
            )
            plt.suptitle("Initial shock ε_i (green)", fontsize=10)

        plt.text(
            0.02, 0.98, f"step = {t}",
            transform=plt.gca().transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        plt.text(
            0.02, 0.92, f"defaults = {int(np.sum(state))}/{len(state)}",
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )

        plt.axis("off")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=2)