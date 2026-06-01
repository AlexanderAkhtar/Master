#  multi-agent road evacuation simulation
# can be run standalone by settings or called from sim_configs.py to generate test set of datasets

import random
import math
import os
from collections import defaultdict
import json

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# simulation parameters (used when running standalone) -
RANDOM_SEED    = 1
SIM_NAME       = "few"
N_NODES        = 1000
PLANE          = (1000, 1000)
K_NN           = 4
N_GROUPS       = (50, 10, 6, 3)     # (individuals, pairs, trios, quads) number of each group
TIMESTEPS      = 40

P_IDLE         = 0.03               # do-nothing pause prob, independent of health
MAX_EXTRA_IDLE = 0.35               # max health pause prob at health = 0

CHILD_AGE = (0, 17)
ADULT_AGE = (18, 83)
P_CHILD   = 0.3                     # prob any group member is a child

HEALTH_SLOPE = 0.7                  # health lost per year of age
HEALTH_NOISE = 20.0                 # individual variation std
HI_THRESH    = 65                   # threshold values for high and medium health bracket
MED_THRESH   = 35


def jfmt(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(type(o))

def assign_health(age):
    # assigns a health score (0-100) and a corresponding health bracket for agent
    h = max(0, min(100, 100 - age*HEALTH_SLOPE +  random.gauss(0, HEALTH_NOISE)))
    br = "high" if h >= HI_THRESH else ("medium" if h >= MED_THRESH else "low")
    return h, br


def plot_network(G, pos, exits, path):
    # generates and saves a 2d graph of the road network
    fig, ax = plt.subplots(figsize=(8, 8))
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="gray", lw=0.5, alpha=0.6)
    non = [n for n in G.nodes if n not in exits]
    ax.scatter([pos[n][0] for n in non], [pos[n][1] for n in non],
               s=8, color="steelblue", alpha=0.7)
    ax.scatter([pos[n][0] for n in exits], [pos[n][1] for n in exits],
               s=120, color="red", label="exits")
    ax.legend(); ax.axis("equal"); ax.grid(False); ax.set_title("Road Network")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def road_network_sim(cfg=None, seed=None, out_dir=None):
    # runs one instance of the simulation
    # dfg = parameters, defualt parameters run if none given
    # seed = random seed
    # returns the trajectories.csv of the agent paths throughout the sim

    # override defualt parameters if cfg given
    n_groups   = cfg["n_groups"]    if cfg else N_GROUPS
    timesteps  = cfg["timesteps"]   if cfg else TIMESTEPS
    max_idle   = cfg["health_idle"] if cfg else MAX_EXTRA_IDLE
    k_nn       = cfg["k_nn"]        if cfg else K_NN
    seed       = seed if seed is not None else RANDOM_SEED
    out_dir    = out_dir if out_dir else f"sim_dataset/{SIM_NAME}"

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    # health pause prob calculation
    def idle_prob(health):
        return ((100 - health) / 100) * max_idle

    # road network
    pos = {i: (random.uniform(0, PLANE[0]), random.uniform(0, PLANE[1])) for i in range(N_NODES)}
    G = nx.Graph()
    for i, p in pos.items(): G.add_node(i, pos=p)

    nodes = list(pos.items())
    for i, pi in nodes:
        dists = sorted([(math.hypot(pi[0]-pj[0], pi[1]-pj[1]), j)
                        for j, pj in nodes if i != j])
        for _, j in dists[:k_nn]:
            if not G.has_edge(i, j): G.add_edge(i, j)

    # connects isolated components until fully connected
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        ca, cb = comps[0], comps[1]
        b = min([(math.hypot(pos[i][0]-pos[j][0], pos[i][1]-pos[j][1]), i, j)
                 for i in ca for j in cb], key=lambda x: x[0])
        G.add_edge(b[1], b[2])

    # assignes the exit nodes and buildes the exit next hop look up s
    exits = random.sample(list(G.nodes()), 2)
    pe = {e: nx.single_source_shortest_path(G, e) for e in exits}
    le = {e: nx.single_source_shortest_path_length(G, e) for e in exits}
    hop = {}
    for nd in G.nodes():
        ne = min(exits, key=lambda e: le[e].get(nd, float("inf")))
        fp = pe[ne].get(nd)
        full = nx.shortest_path(G, nd, ne) if fp is None else list(reversed(fp))
        hop[nd] = full[1] if len(full) >= 2 else None

    # agents
    agents, aid = {}, 0

    # build individual and group agents
    for _ in range(n_groups[0]):
        age = random.randint(*ADULT_AGE); h, br = assign_health(age)
        agents[aid] = {"id": aid, "group_id": None, "age": age,
                       "gender": random.choice("MF"), "health": h,
                       "health_bracket": br, "behavior": "evacuation",
                       "node": random.choice(list(G.nodes()))}
        aid += 1

    gid = 0
    for size, count in enumerate(n_groups[1:]):
        for _ in range(count):
            mbs = list(range(aid, aid + size + 2))
            gi  = random.randrange(size + 2)
            for i, a in enumerate(mbs):
                #ensures at least one adult per group
                adult = (i == gi) or (random.random() >= P_CHILD)
                age   = random.randint(*(ADULT_AGE if adult else CHILD_AGE))
                h, br = assign_health(age)
                agents[a] = {"id": a, "group_id": gid, "age": age,
                             "gender": random.choice("MF"), "health": h,
                             "health_bracket": br,
                             "behavior": "stay" if age < 18 else "rendezvous",
                             "node": random.choice(list(G.nodes()))}
                aid += 1
            gid += 1

    groups = defaultdict(list)
    for a in agents.values():
        if a["group_id"] is not None:
            groups[a["group_id"]].append(a["id"])

   
    # simulation

    # helper function to record agant values for given timestep
    rows = []
    def record(t):
        for a in agents.values():
            xy = pos[a["node"]]
            rows.append({"agent_id": a["id"], 
                         "group_id": a["group_id"], 
                         "behavior": a["behavior"],
                         "health_bracket": a["health_bracket"],
                         "health": round(a["health"], 2), 
                         "timestep": t,
                         "node_id": int(a["node"]), 
                         "x": xy[0], 
                         "y": xy[1],
                         "age": a["age"], 
                         "gender": a["gender"]})

    record(0)
    spl = dict(nx.all_pairs_shortest_path_length(G))


    # behavior state machine
    for t in range(1, timesteps + 1):
        nxt = {}
        for ai, a in agents.items():
            if a["behavior"] == "exited":
                nxt[ai] = a["node"]; continue

            if a["behavior"] in ("rendezvous", "evacuation"):
                if random.random() < P_IDLE:
                    nxt[ai] = a["node"]; continue
                if random.random() < idle_prob(a["health"]):
                    nxt[ai] = a["node"]; continue

            if a["behavior"] == "stay":
                if any(agents[m]["node"] == a["node"] and agents[m]["age"] >= 18
                       for m in groups.get(a["group_id"], []) if m != ai):
                    a["behavior"] = "rendezvous"
                else:
                    nxt[ai] = a["node"]; continue

            if a["behavior"] == "rendezvous":
                sep = [(spl[a["node"]].get(agents[m]["node"], float("inf")),
                        agents[m]["node"])
                       for m in groups[a["group_id"]]
                       if agents[m]["node"] != a["node"]]
                if not sep:
                    for m in groups[a["group_id"]]: agents[m]["behavior"] = "evacuation"
                    nxt[ai] = a["node"]
                else:
                    path = nx.shortest_path(G, a["node"], min(sep)[1])
                    nxt[ai] = path[1] if len(path) >= 2 else a["node"]

            if a["behavior"] == "evacuation":
                h_ = hop[a["node"]]
                if h_ is None:
                    a["behavior"] = "exited"; nxt[ai] = a["node"]
                else:
                    nxt[ai] = h_

        for ai, nd in nxt.items():
            agents[ai]["node"] = nd
            if agents[ai]["node"] in exits and agents[ai]["behavior"] == "evacuation":
                agents[ai]["behavior"] = "exited"

        for g, mbs in groups.items():
            if len(set(agents[m]["node"] for m in mbs)) == 1:
                for m in mbs:
                    if agents[m]["behavior"] != "exited":
                        agents[m]["behavior"] = "evacuation"
        record(t)

    # Data recording
    cols = ["agent_id","group_id","behavior","health_bracket","health", "timestep","node_id","x","y","age","gender"]
    csv_path = os.path.join(out_dir, "trajectories.csv")
    graph_path = os.path.join(out_dir, "geodesic.json")

    pd.DataFrame(rows)[cols].to_csv(csv_path, index=False)
    plot_network(G, pos, exits, os.path.join(out_dir, "road_network.png"))

    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    with open(graph_path, "w") as f:
        json.dump(spl, f, indent=4, default=jfmt)

    bk = pd.DataFrame(rows).drop_duplicates("agent_id")["health_bracket"].value_counts()
    n  = df_agents = len(agents)
    print(f"Done {n} agents  " + "  ".join(f"{b}={c}" for b, c in sorted(bk.items())))

    return csv_path


if __name__ == "__main__":
    road_network_sim()
