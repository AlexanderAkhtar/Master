# Experiment 2 mixed_model
# compares three randomforest models
 
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_rel
 
from dp_tda import map_test_paths_to_tda
 
# collects the last label of the target column
def build_target(df, col):
    return (df.sort_values("timestep").groupby("agent_id").last().reset_index()[["agent_id", col]])
 

# builds the raw features
def build_raw_features(df):
    rows = []
    for aid, g in df.groupby("agent_id"):
        g = g.sort_values("timestep")
        xy_all = g[["x", "y"]].values
 
        # all features except displacement use active timesteps
        active = g[g["behavior"] != "exited"]
        if len(active) < 2:
            continue
 
        xy_act  = active[["x", "y"]].values
        diffs   = np.diff(xy_act, axis=0)
        dists   = np.linalg.norm(diffs, axis=1)   #per step distance
 
        total_dist = float(dists.sum())
        mean_speed = float(dists.mean())
        std_speed  = float(dists.std())
        speed_cv   = float(std_speed / (mean_speed + 1e-8))
 
        # displacement uses all timesteps - the start and stop positions
        displacement = float(np.linalg.norm(xy_all[-1] - xy_all[0]))
        straightness_ratio = float(displacement / (total_dist + 1e-8))
 
        # pause characterisation
        stopped = dists < 1e-6    # boolean mask for checks distance traveled
 
        frac_stopped = float(stopped.mean())
 
        if stopped.any():
            padded      = np.concatenate([[False], stopped, [False]])
            edges       = np.diff(padded.astype(int))
            run_starts  = np.where(edges ==  1)[0]
            run_ends    = np.where(edges == -1)[0]
            run_lengths = run_ends - run_starts
 
            longest_stop      = int(run_lengths.max())
            pause_count       = int(len(run_lengths))
            mean_stop_duration = float(run_lengths.mean())
        else:
            longest_stop       = 0
            pause_count        = 0
            mean_stop_duration = 0.0
 
        rows.append({
            "agent_id":          aid,
            "total_dist":        total_dist,
            "mean_speed":        mean_speed,
            "speed_cv":          speed_cv,
            "displacement":      displacement,
            "straightness_ratio": straightness_ratio,
            "frac_stopped":      frac_stopped,
            "longest_stop":      float(longest_stop),
            "pause_count":       float(pause_count),
            "mean_stop_duration": mean_stop_duration,
        })
    return pd.DataFrame(rows)
 

#builds the topology features
def build_taxonomy_features(agent_nodes, meta=None, G=None):
    # compute total agents per interval, population fraction features
    interval_totals = defaultdict(int)
    if meta:
        for nk, m in meta.items():
            interval_totals[m["interval_index"]] += len(m["agents"])
 
    # stationarity threshold for centroid drift 50 units is 5 per cent of the 1000-unit simulation space.
    STATIONARY_THRESHOLD = 50.0
 
    def get_centroid(nk):
        if meta and nk in meta and meta[nk].get("centroid") is not None:
            return np.array(meta[nk]["centroid"], dtype=float)
        return None
 
    # there are multiple features commented out, to keep them for posterity
    rows = []
    for aid, seq in agent_nodes.items():
        if not seq:
            continue
 
        centroids = [get_centroid(nk) for nk in seq]
        valid_c   = [c for c in centroids if c is not None]
        n_valid   = len(valid_c)
 
        if n_valid == 0:
            continue
 
        c_arr = np.array(valid_c)   # shape (n_valid, 2)
 
        # interval drift values
        drifts = []
        for i in range(len(seq) - 1):
            c1, c2 = get_centroid(seq[i]), get_centroid(seq[i+1])
            if c1 is not None and c2 is not None:
                drifts.append(float(np.linalg.norm(c2 - c1)))
 
        drift_arr          = np.array(drifts) if drifts else np.array([0.0])
        spatial_path_len   = float(drift_arr.sum())
        mean_drift         = float(drift_arr.mean())
        std_drift          = float(drift_arr.std())
        drift_cv           = float(std_drift / (mean_drift + 1e-8))
 
        spatial_displace   = float(np.linalg.norm(c_arr[-1] - c_arr[0]))
        
        # path straightness, 1 = straight line 0 = circled back
        path_straightness  = float(spatial_displace / (spatial_path_len + 1e-8))
 
        centroid_x_std     = float(np.std(c_arr[:, 0]))
        centroid_y_std     = float(np.std(c_arr[:, 1]))
 
        
        #initial_cx, initial_cy = (float(c_arr[0, 0]),  float(c_arr[0, 1]))
        #final_cx,   final_cy   = (float(c_arr[-1, 0]), float(c_arr[-1, 1]))
 
        if len(drifts) > 0:
            stationary_mask  = drift_arr < STATIONARY_THRESHOLD
            #stationarity_cnt = int(stationary_mask.sum())
 
            if stationary_mask.any():
                padded      = np.concatenate([[False], stationary_mask, [False]])
                edges       = np.diff(padded.astype(int))
                run_lengths = np.where(edges == -1)[0] - np.where(edges == 1)[0]
                #max_stat_run = int(run_lengths.max())
            else:
                max_stat_run = 0
 
            # Lag-1 autocorrelation of the drift sequence
            a, b = drift_arr[:-1], drift_arr[1:]
            if (len(drift_arr) >= 3
                    and a.std() > 1e-8
                    and b.std() > 1e-8):
                drift_autocorr = float(np.corrcoef(a, b)[0, 1])
                if np.isnan(drift_autocorr):
                    drift_autocorr = 0.0
            else:
                drift_autocorr = 0.0
        else:
            #stationarity_cnt = 0
            #max_stat_run     = 0
            drift_autocorr   = 0.0
 
        pop_fracs = []
        if meta:
            for nk in seq:
                if nk in meta:
                    iv_total = interval_totals.get(meta[nk]["interval_index"], 1)
                    pop_fracs.append(len(meta[nk]["agents"]) / max(iv_total, 1))
 
        mean_pop_frac = float(np.mean(pop_fracs)) if pop_fracs else 0.0
        min_pop_frac  = float(np.min(pop_fracs))  if pop_fracs else 0.0
 
        # the slope of population frac 
        if len(pop_fracs) >= 3:
            x = np.arange(len(pop_fracs), dtype=float)
            pop_frac_trend = float(np.polyfit(x, pop_fracs, 1)[0])
        else:
            pop_frac_trend = 0.0
 
        mean_out_degree  = 0.0
        mean_in_degree   = 0.0
        mean_edge_weight = 0.0
        #unique_edge_ratio = 0.0
        n_returns         = 0
        max_excursion_ret = 0.0
 
        #node degrees
        if G is not None and len(seq) > 0:
            out_degs = [G.out_degree(nk) for nk in seq if nk in G]
            in_degs  = [G.in_degree(nk)  for nk in seq if nk in G]
            if out_degs:
                mean_out_degree = float(np.mean(out_degs))
                mean_in_degree  = float(np.mean(in_degs))
 
            if len(seq) > 1:
                edge_weights = []
                edges_used   = set()
                for i in range(len(seq) - 1):
                    u, v = seq[i], seq[i+1]
                    edges_used.add((u, v))
                    edge_weights.append(
                        G[u][v].get("weight", 1) if G.has_edge(u, v) else 0)
 
                if edge_weights:
                    mean_edge_weight = float(np.mean(edge_weights))
                #unique_edge_ratio = len(edges_used) / max(len(seq) - 1, 1)
 
        # Spatial returns count
        if n_valid >= 3:
            start = c_arr[0]
            dists_from_start = np.linalg.norm(c_arr - start, axis=1)
            n_returns = int(np.sum(dists_from_start[1:] < dists_from_start[:-1]))
 
        # Maximum excursion and return
        if n_valid >= 2:
            start = c_arr[0]
            dists = np.linalg.norm(c_arr - start, axis=1)
            max_excursion     = float(dists.max())
            final_dist        = float(dists[-1])
            max_excursion_ret = max(0.0, max_excursion - final_dist)
 
        # tax feature set 
        rows.append({
            "agent_id": aid,
            # Group 1: centroid path geometry
            #"spatial_path_length":  spatial_path_len,
            #"spatial_displacement": spatial_displace,
            "path_straightness":    path_straightness,
            "centroid_x_std":       centroid_x_std,
            "centroid_y_std":       centroid_y_std,
            #"mean_drift":           mean_drift,
            "drift_cv":             drift_cv,
            #"initial_centroid_x":   initial_cx,
            #"initial_centroid_y":   initial_cy,
            #"final_centroid_x":     final_cx,
            #"final_centroid_y":     final_cy,
            # Group 2: temporal pause patterns
            #"stationarity_count":   float(stationarity_cnt),
            #"max_stationarity_run": float(max_stat_run),
            "drift_autocorr":       drift_autocorr,
            # Group 3: population context
            "mean_pop_frac":        mean_pop_frac,
            "min_pop_frac":         min_pop_frac,
            "pop_frac_trend":       pop_frac_trend,
            # Group 4: graph topology
            "mean_out_degree":      mean_out_degree,
            "mean_in_degree":       mean_in_degree,
            "mean_edge_weight":     mean_edge_weight,
            #"unique_edge_ratio":    unique_edge_ratio,
            "n_returns":            float(n_returns),
            "max_excursion_return": max_excursion_ret,
        })
 
    return pd.DataFrame(rows)
 
 
# main test 2 pipeline
def run_predictive_comparison(df, target_col, tda_build_graph, build_paths, k_folds=5, tda_params=(5, 2.0, 5, True), min_movement_threshold=0.0, precomputed_raw_scores=None):
    tgt = build_target(df, target_col)
    raw = build_raw_features(df)
    raw_cols = [c for c in raw.columns if c != "agent_id"]
 
    data = tgt.merge(raw, on="agent_id")
 
    # filter the agents with very low movement
    if min_movement_threshold > 0:
        before = len(data)
        data = data[data["total_dist"] > min_movement_threshold].copy()
        removed = before - len(data)
        if removed:
            print(f" Removed {removed} stationary agents ({len(data)} remain)")
 
    y = data[target_col].reset_index(drop=True) 
    agents = data["agent_id"].reset_index(drop=True)
 
    # handle precomputed raw score cache
    if isinstance(precomputed_raw_scores, dict):
        _cached_scores = precomputed_raw_scores.get("fold_scores", [])
        _cached_raw_cm = precomputed_raw_scores.get("cm")
        use_cached_raw = len(_cached_scores) == k_folds
    elif precomputed_raw_scores is not None:
        _cached_scores = precomputed_raw_scores
        _cached_raw_cm = None
        use_cached_raw = len(_cached_scores) == k_folds
    else:
        _cached_scores = []
        _cached_raw_cm = None
        use_cached_raw = False
 
    if use_cached_raw:
        print(f" raw: using cached scores (skipping RF)")
        raw_sc = [tuple(s) for s in _cached_scores]
    else:
        raw_sc = []
 
    # importance and cm collection
    classes = sorted(y.unique())
    n_cls   = len(classes)
    tax_cm  = np.zeros((n_cls, n_cls), dtype=int)
    raw_cm  = np.zeros((n_cls, n_cls), dtype=int)
    comb_cm = np.zeros((n_cls, n_cls), dtype=int)
 
    tax_imp_folds  = []
    raw_imp_folds  = []
    comb_imp_folds = []
 
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    tax_sc, comb_sc = [], []
 
    for fold, (tr_idx, te_idx) in enumerate(skf.split(agents, y)):
        print(f" Fold {fold+1}/{k_folds}...")
        tr_aids = set(agents.iloc[tr_idx])
        te_aids = set(agents.iloc[te_idx])
 
        # build taxonomy
        G, ivs, meta = tda_build_graph(build_paths(df[df["agent_id"].isin(tr_aids)]), *tda_params)
 
        tr_nodes = defaultdict(list)
        for nk, m in meta.items():
            for a in m["agents"]:
                tr_nodes[a].append(nk)

        # project test agents to taxonom
        te_nodes = map_test_paths_to_tda(build_paths(df[df["agent_id"].isin(te_aids)]), ivs, meta)
 
        all_nodes = {**tr_nodes, **te_nodes}
        taxon_df  = build_taxonomy_features(all_nodes, meta=meta, G=G)
        tax_cols  = [c for c in taxon_df.columns if c != "agent_id"]
 
        full = data.merge(taxon_df, on="agent_id", how="left").fillna(0)
        tr   = full[full["agent_id"].isin(tr_aids)]
        te   = full[full["agent_id"].isin(te_aids)]
 
        y_tr, y_te = tr[target_col], te[target_col]
 
        Xr_tr, Xr_te = tr[raw_cols], te[raw_cols]
        Xt_tr, Xt_te = tr[tax_cols], te[tax_cols]
        Xc_tr = tr[raw_cols + tax_cols]
        Xc_te = te[raw_cols + tax_cols]
 
        # train and score each regime (tax, raw, comb)
        regimes = [(Xt_tr, Xt_te, tax_sc, tax_cm, tax_imp_folds), (Xc_tr, Xc_te, comb_sc, comb_cm, comb_imp_folds)]
        if not use_cached_raw:
            regimes.insert(1, (Xr_tr, Xr_te, raw_sc, raw_cm, raw_imp_folds))
 
        for Xtr, Xte, sc, cm_acc, imp_folds in regimes:
            m = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
            m.fit(Xtr, y_tr)
            preds = m.predict(Xte)
            sc.append((float(accuracy_score(y_te, preds)), float(f1_score(y_te, preds, average="macro"))))
            cm_acc += confusion_matrix(y_te, preds, labels=classes)
            imp_folds.append((list(Xtr.columns), m.feature_importances_.tolist()))
 
    if use_cached_raw and _cached_raw_cm is not None:
        raw_cm = np.array(_cached_raw_cm, dtype=int)
 
    #generates the summary files
    def avg_importances(folds):
        if not folds:
            return {}
        names = folds[0][0]
        arr   = np.array([f[1] for f in folds])
        return {n: float(v) for n, v in zip(names, arr.mean(axis=0))}
 
    raw_importances  = avg_importances(raw_imp_folds)
    tax_importances  = avg_importances(tax_imp_folds)
    comb_importances = avg_importances(comb_imp_folds)
 
    if use_cached_raw and isinstance(precomputed_raw_scores, dict):
        cached_imp = precomputed_raw_scores.get("importances")
        if cached_imp:
            raw_importances = cached_imp
 
    def summ(sc):
        accs = [s[0] for s in sc]
        f1s  = [s[1] for s in sc]
        return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)), "f1_mean":  float(np.mean(f1s)),  "f1_std":  float(np.std(f1s))}
 
    comb_f1s = [s[1] for s in comb_sc]
    raw_f1s  = [s[1] for s in raw_sc]
    tax_f1s  = [s[1] for s in tax_sc]
 
    t_cr, pv_cr_2 = ttest_rel(comb_f1s, raw_f1s)
    pv_cr_1 = float(pv_cr_2 / 2 if t_cr > 0 else 1 - pv_cr_2 / 2)
    _, pv_tr = ttest_rel(tax_f1s, raw_f1s)
 
    raw_cache = {
        "fold_scores":  raw_sc,
        "cm":           raw_cm.tolist(),
        "classes":      [str(c) for c in classes],
        "importances":  raw_importances,
    }
 
    return {
        "taxonomy":                summ(tax_sc),
        "raw":                     summ(raw_sc),
        "combined":                summ(comb_sc),
        "classes":                 [str(c) for c in classes],
        "taxonomy_cm":             tax_cm.tolist(),
        "raw_cm":                  raw_cm.tolist(),
        "combined_cm":             comb_cm.tolist(),
        "raw_importances":         raw_importances,
        "taxonomy_importances":    tax_importances,
        "combined_importances":    comb_importances,
        "p_combined_vs_raw":       float(pv_cr_2),
        "p_combined_vs_raw_1tail": float(pv_cr_1),
        "p_taxonomy_vs_raw":       float(pv_tr),
        "raw_cache":               raw_cache,
    }