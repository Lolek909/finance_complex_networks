import os
import itertools

BASE_PATH = "./results/analysis"

sectors = {
    'Tech': set(['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','AMD','INTC','ADBE']),
    'Finance': set(['JPM','BAC','GS','MS','WFC','C','V','MA','AXP','PYPL']),
    'Energy': set(['XOM','CVX','SHEL','BP','TTE','COP','SLB','PBR','EQNR','VLO']),
    'Healthcare': set(['JNJ','UNH','PFE','ABBV','LLY','MRK','TMO','AZN','NVO','DHR'])
}


def jaccard(a, b):
    return len(a & b) / len(a | b)


def load_communities(path):
    communities = {}
    with open(path, "r") as f:
        for line in f:
            if "Community" in line:
                parts = line.split(":")
                idx = int(parts[0].split()[1])
                nodes = eval(parts[1].strip())
                communities[idx] = set(nodes)
    return communities


def best_matching(communities, sectors):
    comm_items = list(communities.items())
    sector_items = list(sectors.items())

    n_comm = len(comm_items)
    n_sec = len(sector_items)

    k = min(n_comm, n_sec)

    best_score = -1
    best_mapping = {}

    for perm in itertools.permutations(sector_items, k):
        score = 0
        mapping = {}

        for i in range(k):
            comm_id, comm_nodes = comm_items[i]
            sector_name, sector_nodes = perm[i]

            score += jaccard(comm_nodes, sector_nodes)
            mapping[comm_id] = sector_name

        if score > best_score:
            best_score = score
            best_mapping = mapping

    return best_score, best_mapping


def analyze_missing(mapping, communities, sectors):
    used_sectors = set(mapping.values())
    all_sectors = set(sectors.keys())

    missing_sectors = all_sectors - used_sectors
    extra_communities = set(communities.keys()) - set(mapping.keys())

    return missing_sectors, extra_communities


results = []

for f1 in os.listdir(BASE_PATH):
    p1 = os.path.join(BASE_PATH, f1)
    if not os.path.isdir(p1):
        continue

    for f2 in os.listdir(p1):
        p2 = os.path.join(p1, f2)

        comm_file = os.path.join(p2, "communities.txt")
        if not os.path.exists(comm_file):
            continue

        communities = load_communities(comm_file)

        score, mapping = best_matching(communities, sectors)
        missing, extra = analyze_missing(mapping, communities, sectors)

        results.append({
            "folder1": f1,
            "folder2": f2,
            "score": score,
            "mapping": mapping,
            "missing_sectors": missing,
            "extra_communities": list(extra)
        })


results.sort(key=lambda x: x["score"], reverse=True)


for r in results:
    print("\n====================")
    print(r["folder1"], r["folder2"])
    print("Score:", r["score"])
    print("Mapping:", r["mapping"])
    print("Missing sectors:", r["missing_sectors"])
    print("Extra communities:", r["extra_communities"])