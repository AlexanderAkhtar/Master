# time variant mapper implementation

import collections
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
import networkx as nx
import os
import pickle
import json

from data_vis import visualize_trajectories, plot_graph_3d

# manual parameters for singular run

OUT = "./software_test/tda"   
DATA = "sim_dataset/few"   
TDA_PARAMS = {"k": 3, "p": 2.0, "cluster_threshold": 3, "overlap": True}

geodesic = {}

#datahelpers 

# load the  geodesic distance lookup table/matrix
def set_geodesic(geo_dict):

    global geodesic
    geodesic = geo_dict
 
def load_and_set_geodesic(path):
    with open(path) as f:
        raw = json.load(f)
    set_geodesic({int(k): {int(kk): v for kk, v in vv.items()} for k, vv in raw.items()})


def load_data(path):  
    df = pd.read_csv(path) 
    req = {"agent_id", "timestep","node_id", "x", "y"} 
    if not req.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {req}")
    return df.sort_values(["agent_id", "timestep"]).reset_index(drop=True)  

# builds the agent paths dict:  agent_id: (t, nid, x, y) 
def build_paths(df):  
    
    paths = {}
    for aid, g in df.groupby("agent_id"): 
        paths[aid] = list(zip(g["timestep"].astype(int), g["node_id"].astype(int), g["x"].astype(float), g["y"].astype(float)))
    return paths



# point by point distance up to length of shorter path
def lp_path_distance(p1, p2, p):

    if not p1 or not p2:  
        return float("inf") 
    
    """ euclidian distance metric implemented but not tested so not mentioned in thesis
    c1 = np.array([(x, y) for (_, x, y) in p1]) 
    c2 = np.array([(x, y) for (_, x, y) in p2]) 
    n = min(len(c1), len(c2)) 
    d = np.linalg.norm(c1[:n] - c2[:n], axis=1) 
    """
    nodes1 = [node for (_, node, _, _) in p1]
    nodes2 = [node for (_, node, _, _) in p2]
    n = min(len(nodes1), len(nodes2))
    
    # look up the distance from geo table
    d = np.array([geodesic[nodes1[i]][nodes2[i]] for i in range(n)], dtype=np.float64)

    if p == float("inf"):           # special case  for p = inf
        return float(np.max(d))
    
    return float(np.sum(d ** p) ** (1.0 / p))


  
# creates a matrix containing the lp distance metric of every pair of trajectory segments
def pairwise_lp_distance_matrix(segs, p):
    n = len(segs)
    if n <= 1:
        return np.zeros((n, n))

    # run pairwise
    raw = [lp_path_distance(segs[i], segs[j], p) for i in range(n) for j in range(i+1, n)]
    
    # data cleanup
    D = squareform(raw)
    fin = D[np.isfinite(D)]
    if fin.size:
        D = np.where(np.isinf(D), fin.max() * 10, D)
    return D

# creates the set of time intervals
def make_intervals(T_min, T_max, k, overlap=False):

    if k == 0:
        return [(t, t) for t in range(T_min, T_max + 1)]
    stride = max(1, k // 2) if overlap else k + 1
    ivs, start = [], T_min
    
    while start <= T_max:
        end = min(start + k, T_max)
        ivs.append((start, end))
        if end == T_max:
            break
        start += stride
    return ivs


#creates the agents restricted trajectories
def restrict_path_to_interval(path, iv):
    return [pt for pt in path if iv[0] <= pt[0] <= iv[1]]


# clusters the restricked paths within the intervals
def cluster_interval(aids, segs, cluster_threshold, p):
 
    # Agents with no data in this interval get no membership
    if not aids:
        return {}

    # generate mask for which aids have paths
    valid = [i for i, s in enumerate(segs) if s]
    empty = [i for i, s in enumerate(segs) if not s]
    res = {aids[i]: -1 for i in empty}

    if not valid:
        return res

    #e extract only the valid agents and paths
    v_aids = [aids[i] for i in valid]
    v_segs = [segs[i] for i in valid]  

 
    #build distance metric and clean data
    D = pairwise_lp_distance_matrix(v_segs, p)
    fin = D[np.isfinite(D)]
    if fin.size:
        D = np.where(np.isinf(D), fin.max() * 10, D)  
    else:
        D = np.zeros_like(D)

    # run clustering and return results
    lbls = DBSCAN(eps=cluster_threshold, min_samples=1, metric="precomputed").fit_predict(D)
    for a, l in zip(v_aids, lbls.tolist()):
        res[a] = l
    return res


# builds the child parent dict with edge weight
def build_children(edges):

    cm = collections.defaultdict(list)  
    for e in edges:
        w = 1
        if len(e) == 3 and isinstance(e[2], dict):  # e[2] attribute dict 
            w = e[2].get("weight", len(e[2].get("agents", [])))   
        cm[tuple(e[0])].append({"node": tuple(e[1]), "weight": w}) 
    return cm

# build the taxonomy cluster  per interval, then connect over intervals
def tda_build_graph(paths, k, p, cluster_threshold, overlap=False):
    
    #checks or timesteps t
    ts = [t for path in paths.values() for (t, _, _, _) in path]
    if not ts:
        raise ValueError("No timesteps in paths")

    ivs = make_intervals(min(ts), max(ts), k, overlap=overlap)
    lbl_maps = []   # one dict per interval  agent_id: cluster label 
    meta = {}       # (iv_idx, label): node metadata 

    # Runs for each interval 
    for idx, iv in enumerate(ivs):
        aids = list(paths.keys())
        segs = [restrict_path_to_interval(paths[a], iv) for a in aids]
        lm = cluster_interval(aids, segs, cluster_threshold, p)
        lbl_maps.append(lm)

        # data collection
        cl_agents = collections.defaultdict(list)
        cl_coords = collections.defaultdict(list)
        for a in aids:
            l = lm.get(a)
            if l is None or l == -1:
                continue
            cl_agents[l].append(a)
            r = restrict_path_to_interval(paths[a], iv)
            if r:
                cl_coords[l].append(np.mean([[x, y] for _, _, x, y in r], axis=0))

        for l, agents in cl_agents.items():
            nk = (idx, int(l))
            c = np.mean(np.vstack(cl_coords[l]), axis=0) if cl_coords[l] else None
            meta[nk] = {"interval_index": idx, "interval": iv,"cluster_label": int(l), "agents": agents, "centroid": c}
  
    # build graph  
    G = nx.DiGraph() 
    G.add_nodes_from((nk, m) for nk, m in meta.items()) 

    # follows each agent through their path recording the previous nodes and adds to its weigth 
    for aid in paths:
        prev = None
        for idx, lm in enumerate(lbl_maps):
            l = lm.get(aid)
            if l is None or l == -1:
                prev = None; continue
            nk = (idx, int(l))
            if nk not in G.nodes:
                prev = None; continue
            if prev is not None:
                if G.has_edge(prev, nk):
                    G[prev][nk]["weight"] += 1
                    G[prev][nk]["agents"].append(aid)
                else:
                    G.add_edge(prev, nk, weight=1, agents=[aid])
            prev = nk

    cm = build_children(G.edges(data=True))
    for nk in meta:
        meta[nk]["children"] = cm.get(nk, [])

    return G, ivs, meta


def map_test_paths_to_tda(test_paths, ivs, meta):
    # project test agent into the pre built tda graph nearest centroid matching
    out = collections.defaultdict(list)
    for aid, path in test_paths.items():
        for idx, iv in enumerate(ivs):
            seg = restrict_path_to_interval(path, iv)
            if not seg:
                continue
            pos = np.mean([[x, y] for _, _, x, y in seg], axis=0)
            iv_nodes = {k: v for k, v in meta.items() if v["interval_index"] == idx}
            if not iv_nodes:
                continue
            best, bd = None, float("inf")
            for nk, m in iv_nodes.items():
                if m["centroid"] is not None:
                    d = np.linalg.norm(pos - m["centroid"])
                    if d < bd:
                        bd, best = d, nk
            if best is not None:
                out[aid].append(best)
    return out





if __name__ == "__main__":
    # single manual run test
    os.makedirs(OUT, exist_ok=True)
    traj_path = os.path.join(DATA, "trajectories.csv")
    df = load_data(traj_path)

    geo_path = os.path.join(DATA, "geodesic.json")
    with open(geo_path) as f:
        geodesic = json.load(f)
    geodesic = {int(k): {int(kk): v for kk, v in vv.items()} for k, vv in geodesic.items()}
    
    paths = build_paths(df)

    G, ivs, meta = tda_build_graph(paths, **TDA_PARAMS)
    print(f"tda: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


    with open(os.path.join(OUT, "tda_graph.pkl"), "wb") as f: pickle.dump(G, f)
    with open(os.path.join(OUT, "tda_meta.pkl"), "wb") as f: pickle.dump(meta, f)

    visualize_trajectories(df, filter_mode="all", save_path=f"{OUT}/traj_all.png")
    plot_graph_3d(G, meta, title="Taxonomy", save_path=f"{OUT}/tda_3d.png")

    print("Done")
