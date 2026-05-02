import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_sector_flow_heatmap(G, sector_map, out_dir, name):
    sectors = sorted(list(set(sector_map.values())))
    if "Unknown" in sectors:
        sectors.remove("Unknown")
        sectors.append("Unknown")

    flow_matrix = pd.DataFrame(0.0, index=sectors, columns=sectors)

    for u, v, data in G.edges(data=True):
        sec_u = sector_map.get(u, "Unknown")
        sec_v = sector_map.get(v, "Unknown")
        weight = data.get('weight', 0.0)

        if sec_u in flow_matrix.index and sec_v in flow_matrix.columns:
            flow_matrix.loc[sec_u, sec_v] += weight

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        flow_matrix,
        annot=True,
        cmap="YlOrRd",
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Zsumowana waga powiązań (Lead -> Lag)'}
    )

    plt.title(f"Sector Flow Matrix (Agregacja Makro) - {name}", fontsize=14, pad=15)
    plt.ylabel("Od Sektora (Lead)", fontsize=12, fontweight='bold')
    plt.xlabel("Do Sektora (Lag)", fontsize=12, fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(out_dir, f"sector_heatmap_{name}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    csv_path = os.path.join(out_dir, f"sector_matrix_{name}.csv")
    flow_matrix.to_csv(csv_path)

    print(f"Zapisano Sector Flow Heatmap w: {save_path}")

    return flow_matrix