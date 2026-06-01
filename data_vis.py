# data visualisation  for trajectories and taxonomy graphs

import numpy as np
import matplotlib.pyplot as plt



def visualize_trajectories(df, filter_mode="all", save_path=None):
    if filter_mode == "individuals":
        data = df[df["group_id"].isna()]
        title = "Individual Trajectories"
    elif filter_mode == "groups":
        data = df[df["group_id"].notna()]
        title = "Group Trajectories"
    else:
        data = df
        title = "All Trajectories"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for _, traj in data.groupby("agent_id"):
        ax.plot(traj["x"].values, traj["y"].values, -traj["timestep"].values, alpha=0.7)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Time")
    ax.set_title(title)
    if save_path: plt.savefig(save_path)
    else: plt.show()
    plt.close()




def plot_graph_3d(G, meta, title="Taxonomy", save_path=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.invert_zaxis()

    nodes = [n for n, m in meta.items() if m.get("centroid") is not None]
    if not nodes: return

    # spatial color maping
    xs = [meta[n]["centroid"][0] for n in nodes]
    ys = [meta[n]["centroid"][1] for n in nodes]
    zs = [meta[n]["interval_index"] for n in nodes]
    ss = [20 + 5 * len(meta[n]["agents"]) for n in nodes]

    x_arr = np.array(xs)
    y_arr = np.array(ys)
    x_range = max(x_arr.max() - x_arr.min(), 1e-6)
    y_range = max(y_arr.max() - y_arr.min(), 1e-6)
    xn = (x_arr - x_arr.min()) / x_range
    yn = (y_arr - y_arr.min()) / y_range

    # agent count affects node darkness
    max_agents = max((len(meta[n]["agents"]) for n in nodes), default=1)
    alphas = [0.2 + 0.8 * (len(meta[n]["agents"]) / max_agents) for n in nodes]

    # assign color
    for x, y, z, s, xni, yni, a in zip(xs, ys, zs, ss, xn, yn, alphas):
        r = float(xni)
        g = float(0.3 * (1.0 - abs(xni - yni)))
        b = float(yni)
        color = [[np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1), a]]
        ax.scatter(x, y, z, s=s, c=color, depthshade=True)

    # edge thickness scales with transitions
    edge_weights = [(u, v, d.get("weight", 1)) for u, v, d in G.edges(data=True)]
    max_w = max((w for _, _, w in edge_weights), default=1)
    for u, v, w in edge_weights:
        if (u in meta and v in meta and meta[u]["centroid"] is not None and meta[v]["centroid"] is not None):
            lw = 0.3 + 2.7 * (w / max_w)
            ax.plot([meta[u]["centroid"][0], meta[v]["centroid"][0]], [meta[u]["centroid"][1], meta[v]["centroid"][1]], [meta[u]["interval_index"], meta[v]["interval_index"]], alpha=0.4, linewidth=lw, color="steelblue")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Interval")
    ax.set_title(title)
    if save_path: plt.savefig(save_path)
    else: plt.show()
    plt.close()

