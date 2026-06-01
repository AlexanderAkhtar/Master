# experiment 1 baseline clustring 
# baseline = flatten trajectories, k-means cluster
 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
 
 
def pad_trajectories(df, max_t):
    # pads out agents trajectories to uniform length
 
    mats, aids = [], []
    for aid, g in df.groupby("agent_id"):
        xy = g.sort_values("timestep")[["x", "y"]].values
        if len(xy) < max_t:
            xy = np.vstack([xy, np.tile(xy[-1], (max_t - len(xy), 1))])
        else:
            xy = xy[:max_t]
        mats.append(xy.flatten()) # flatten to 1d
        aids.append(aid)
    return np.array(mats), np.array(aids)
 
 
def chance_baselines(targets):
    # computes the two chance baselines one random and one majority class 
 
    from collections import Counter
 
    counts = Counter(targets)
    total  = len(targets)
    probs  = {c: n/total for c, n in counts.items()}
    p_vals = list(probs.values())
 
    n_classes  = len(counts)
    # uniform random macro F1
    uniform_f1 = sum(
        (2 * p * (1.0 / n_classes)) / (p + 1.0 / n_classes) for p in p_vals
    ) / n_classes
 
    # majority class macro F1
    majority_f1  = 1.0 / n_classes
    majority_acc = max(p_vals)
 
    return {
        "class_distribution": dict(probs),
        "uniform_random_f1":  round(uniform_f1, 4),
        "majority_class_f1":  round(majority_f1, 4),
        "majority_class_acc": round(majority_acc, 4),
    }
 
 
def run_baseline_comparison(df, target_col="health_bracket", n_clusters=10, k_folds=5):
    # evaluate standard k-means clustering predictive power to set baseline
    
    targets = df.sort_values("timestep").groupby("agent_id").last()[target_col]
    X, _ = pad_trajectories(df, int(df["timestep"].max()) + 1)
 
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = []
 
    for tr, te in skf.split(X, targets.values):
        
        # cluster training data  
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        tr_cl = km.fit_predict(X[tr])
        
        #map data to clusters
        te_cl = km.predict(X[te])
 
        # predict target using randomforest with cluster id
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        clf.fit(tr_cl.reshape(-1, 1), targets.iloc[tr])
        scores.append(f1_score(targets.iloc[te], clf.predict(te_cl.reshape(-1, 1)), average="macro"))
 
    avg = float(np.mean(scores))
    chance = chance_baselines(targets.values)
    margin = avg - chance["uniform_random_f1"]
 
    print(f"K-Means baseline F1: {avg:.4f}  (chance={chance['uniform_random_f1']:.4f}, margin={margin:+.4f})")
    return {"baseline_kmeans_f1": avg, **chance}