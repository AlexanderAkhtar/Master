# experiment framework, runs all 3 experiments on every combination of simulation config x random seed x TDA config  
# 
#
 
import os
import json
import time
import traceback
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_1samp
 
from dp_tda import load_data, build_paths, tda_build_graph, load_and_set_geodesic
from test_1_baseline_clustering import run_baseline_comparison
from test_2_mixed_model         import run_predictive_comparison
from test_3_early_prediction    import run_early_prediction
from sim_configs import SIMULATION_CONFIGS, test_rest_seed, n_agents, generate_all_datasets
 
 
  
 
RES    = "./results"
DATA   = "./test_rest_dataset"
TARGET = "behavior"
STATIC_TARGET   = TARGET not in {"behavior"}
STAGE2_TDA_JSON = "./stage2_tda.json"
MMT    = 0.0   # holdover, not used
KF_EXP2 = 5     # cross validation k fold counts
KF_EXP3 = 5 
 
# mapper configurations (k: timstep interval width, p: lp norm, clusters_threshold: cluster distance threshold, overlap)

# manual input from sweep results
TDA = {
    "k8_lp2_c3_nov":    (8, 2.0, 3, False),
    "k15_lp2_c3_nov":   (15, 2.0, 3, False),
    "k17_lp2_c3_nov":  (17, 2.0, 3, False),
    "k5_lp2_c3_ov":  (5, 2.0, 3, True),
    "k3_lp2_c3_ov":    (3, 2.0, 3, True),
}
 
ALL_CONFIGS = list(SIMULATION_CONFIGS.keys())
ALL_TDA     = list(TDA.keys())
 
# data helpers
 
def jfmt(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(type(o))
 
def jsave(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=jfmt)
 
def jload(path):
    with open(path) as f: return json.load(f)
 
def exists(p): return os.path.exists(p)
 
def safe_k(df, col, k):
    mn = df.drop_duplicates("agent_id")[col].value_counts().min()
    if mn < k:
        print(f" failed min class={mn} is less than k={k}, reducing k_folds to {mn}")
        return int(mn)
    return k
 
def res_dir(cfg, seed, tda):
    return os.path.join(RES, cfg, f"seed_{seed}", tda)
 
 
 
# single run of tests  
def run_one(cfg_name, seed, tda_name, tda_params):
    out  = res_dir(cfg_name, seed, tda_name)
    os.makedirs(out, exist_ok=True)
 
    csv = os.path.join(DATA, cfg_name, f"seed_{seed}", "trajectories.csv")
    if not exists(csv):
        print(f" failed missing dataset: {csv}"); return {}
 
    # load geodesic for this dataset
    geo_path = os.path.join(DATA, cfg_name, f"seed_{seed}", "geodesic.json")
    if not exists(geo_path):
        print(f" failed missing geodesic: {geo_path}"); return {}
    load_and_set_geodesic(geo_path)
 
    df    = load_data(csv)
 
    if TARGET not in df.columns:
        print(f" failed '{TARGET}' not in columns"); return {}
 
    na = df["agent_id"].nunique()
    nt = df["timestep"].nunique()
    bc = df.drop_duplicates("agent_id")[TARGET].value_counts()
    print(f" {na} agents x {nt} timesteps " + "  ".join(f"{c}={n}" for c, n in sorted(bc.items())))
 
    #ensure supported amount of folds
    k2 = safe_k(df, TARGET, KF_EXP2)
    k3 = safe_k(df, TARGET, KF_EXP3)
    paths = build_paths(df)
    res = {}
 
    # visualisation only, commented out for less compute  
    """
    vf = os.path.join(out, "vis.flag")  
    if not exists(vf):
        try:
            Gv, _, mv = tda_build_graph(paths, *tda_p)
            visualize_trajectories(df, save_path=os.path.join(out, "traj.png"))
            plot_graph_3d(Gv, mv, title=f"{cfg_name}|s{seed}|{tda_name}", save_path=os.path.join(out, "tda_3d_raw.png"))
            open(vf, "w").write("done")
        
        except Exception as e:
            print(f" failed vis error: {e}")
    """
 
    # experiment 1 baseline clusterinng   
    exp1_path = os.path.join(RES, cfg_name, f"seed_{seed}", "exp1.json")
    if exists(exp1_path):
        print(" exp1: skip")
        res["exp1"] = jload(exp1_path)
    else:
        print(" exp1 ...", end=" ", flush=True)
        
        t0 = time.time()
        r  = run_baseline_comparison(df, target_col=TARGET, n_clusters=10, k_folds=k2)
        jsave(r, exp1_path)
        res["exp1"] = r
        print(f"F1={r['baseline_kmeans_f1']:.4f}  ({time.time()-t0:.1f}s)")
 
    # experiment 2 Mixed model
 
    # to avoid recomputing raw features cache is used to store the raw fold scores raw features dont depend on mapper params so can be shared across all tda configs for a seed
    raw_cache_path = os.path.join(RES, cfg_name, f"seed_{seed}", "raw_scores_cache.json")
    cached_raw = jload(raw_cache_path) if exists(raw_cache_path) else None
 
    exp2_path = os.path.join(out, "exp2.json")
    if exists(exp2_path):  
        print(" exp2: skip")
        res["exp2"] = jload(exp2_path)
    else:
        print(f" exp2 (k_folds={k2}) ...", flush=True) 
        
        t0 = time.time()
        r  = run_predictive_comparison(df=df, target_col=TARGET, tda_build_graph=tda_build_graph, build_paths=build_paths, k_folds=k2, tda_params=tda_params, min_movement_threshold=MMT, precomputed_raw_scores=cached_raw)  
        
        if cached_raw is None and "raw_cache" in r:
            jsave(r["raw_cache"], raw_cache_path)
        
        jsave(r, exp2_path); res["exp2"] = r
        stat_pvalue = r.get("p_combined_vs_raw_1tail", r.get("p_value", 1.0))
        print(f"Tax={r['taxonomy']['f1_mean']:.4f}  Raw={r['raw']['f1_mean']:.4f}  Comb={r['combined']['f1_mean']:.4f}  stat_pvalue(comb>raw)={stat_pvalue:.4f}  ({time.time()-t0:.1f}s)")
 
    # experiment 3 early prediction 
    exp3_path = os.path.join(out, "exp3.json") 
    if exists(exp3_path): 
        print(" exp3: skip")   
        res["exp3"] = jload(exp3_path)
    else:
        print(f" exp3 (k_folds={k3})")   
        
        t0 = time.time()
        raw_cache_dir = os.path.join(RES, cfg_name, f"seed_{seed}")
        r  = run_early_prediction( df=df, target_col=TARGET, tda_build_graph=tda_build_graph, build_paths=build_paths, time_slices=(0.25, 0.5, 0.75, 1.0), k_folds=k3, tda_params=tda_params, min_movement_threshold=MMT, raw_cache_dir=raw_cache_dir, static_target=STATIC_TARGET)
        
        jsave(r, exp3_path); 
        res["exp3"] = r
        
        for sk, m in r.items():
            print(f" {int(m['pct_trajectory_used']*100)}%  Tax={m['taxonomy_f1']:.4f}  Raw={m['raw_f1']:.4f}  Comb={m['combined_f1']:.4f}")
        print(f" done ({time.time()-t0:.1f}s)")
 
    return res
 

# gather all (cfg, seed, tda) results into a single CSV, one row per mapper config
def write_seed_csv(cfg_name, seed, tda_dict):
    
    rows = []
 
    exp1_data = None
    exp1_path = os.path.join(RES, cfg_name, f"seed_{seed}", "exp1.json")
    if exists(exp1_path):
        exp1_data = jload(exp1_path)
 
    for tda_name, tda_params in tda_dict.items():
        k, lp_norm, c_threshold, overlap = tda_params
 
        row = {
            "cfg":        cfg_name,
            "seed":       seed,
            "tda":        tda_name,
            "k":          k,
            "lp_norm":    lp_norm,   
            "c_threshold": c_threshold,
            "overlap":    overlap,
        }

        # exp1
        if exp1_data:
            row["exp1_kmeans_f1"]  = exp1_data.get("baseline_kmeans_f1")
            row["exp1_chance_f1"]  = exp1_data.get("uniform_random_f1")
            row["exp1_maj_f1"]     = exp1_data.get("majority_class_f1")
 
        # exp2
        exp2_path = os.path.join(res_dir(cfg_name, seed, tda_name), "exp2.json")
        if exists(exp2_path):
            d = jload(exp2_path)
            row["exp2_tax_f1"]   = d["taxonomy"]["f1_mean"]
            row["exp2_tax_acc"]  = d["taxonomy"]["acc_mean"]
            row["exp2_raw_f1"]   = d["raw"]["f1_mean"]
            row["exp2_raw_acc"]  = d["raw"]["acc_mean"]
            row["exp2_comb_f1"]  = d["combined"]["f1_mean"]
            row["exp2_comb_acc"]  = d["combined"]["acc_mean"]
            row["exp2_margin_f1"]      = d["combined"]["f1_mean"] - d["raw"]["f1_mean"]
            row["exp2_margin_acc"]      = d["combined"]["acc_mean"] - d["raw"]["acc_mean"]
            
            row["exp2_p_value"]        = d["p_combined_vs_raw"]  
            row["exp2_p_value_1tail"]   = d["p_combined_vs_raw_1tail"]
 

        # exp3
        exp3_path = os.path.join(res_dir(cfg_name, seed, tda_name), "exp3.json")
        if exists(exp3_path):
            d = jload(exp3_path)
            for sk, m in d.items():
                pct = int(float(sk.replace("slice_", "")) * 100)
                row[f"exp3_tax_f1_{pct}"]   = m.get("taxonomy_f1")
                row[f"exp3_raw_f1_{pct}"]   = m.get("raw_f1")
                row[f"exp3_comb_f1_{pct}"]  = m.get("combined_f1")
                row[f"exp3_margin_f1_{pct}"] = ( m["combined_f1"] - m["raw_f1"] if m.get("combined_f1") is not None and m.get("raw_f1") is not None else None)
                raw_acc  = m.get("raw_acc")
                comb_acc = m.get("combined_acc")
                row[f"exp3_raw_acc_{pct}"]      = raw_acc
                row[f"exp3_comb_acc_{pct}"]     = comb_acc
                row[f"exp3_margin_acc_{pct}"]   = (comb_acc - raw_acc if comb_acc is not None and raw_acc is not None else None)
                row[f"exp3_p_value_{pct}"] = m.get("p_value")
 
        rows.append(row)
 
    df = pd.DataFrame(rows)
    out = os.path.join(RES, cfg_name, f"seed_{seed}", f"results_s{seed}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"seed csv written: {out}")
 

#runs the statistical tests returns (sign_count, wilcoxon_p, ttest_p) for a list of margin values across seeds tests are one-tailed (H1  margin > 0)
def stat_tests(values):
    n = len(values)
    sign_count = sum(1 for v in values if v > 0)
 
    wilcoxon_p = 1.0
    if n >= 3 and any(v != 0 for v in values):
        try:
            _, wilcoxon_p = wilcoxon(values, alternative="greater", zero_method="zsplit")
            wilcoxon_p = float(wilcoxon_p)
        except Exception:
            pass
 
    ttest_p = 1.0
    if n >= 3:
        try:
            ttest_p = float(ttest_1samp(values, 0.0, alternative="greater").pvalue)
        except Exception:
            pass
 
    return sign_count, wilcoxon_p, ttest_p
 
# aggregate seed level CSVs into a config level summary 
def write_config_csv(cfg_name):
     
    frames = []   
    for s in test_rest_seed:
        seed_csv_path = os.path.join(RES, cfg_name, f"seed_{s}", f"results_s{s}.csv")  
        if exists(seed_csv_path):
            frames.append(pd.read_csv(seed_csv_path))
 
    if not frames:
        print(f" no seed csvs found for {cfg_name}, skip")
        return
 
    all_seeds = pd.concat(frames, ignore_index=True)
 
    # save the full per seed data as well for reference  
    full_out = os.path.join(RES, cfg_name, f"results_{cfg_name}_full.csv")   
    all_seeds.to_csv(full_out, index=False) 
       
    grp_cols = ["tda", "k", "lp_norm", "c_threshold", "overlap"]
    metric_cols = [c for c in all_seeds.columns if c not in ["cfg", "seed"] + grp_cols and pd.api.types.is_numeric_dtype(all_seeds[c])]  
 
    agg = {}
    for col in metric_cols:
        if "p_value" in col:
            continue
        agg[col + "_mean"] = (col, "mean")
        agg[col + "_std"]  = (col, "std")
 
    summary = all_seeds.groupby(grp_cols, as_index=False).agg(**agg)  
    summary.insert(0, "cfg", cfg_name)    
    summary.insert(1, "n_seeds", len(frames))
 
    stat_rows = []
    for _, grp in all_seeds.groupby(grp_cols):
        key = {c: grp[c].iloc[0] for c in grp_cols}
 
        margins_f1  = grp["exp2_margin_f1"].dropna().tolist()  if "exp2_margin_f1"  in grp.columns else []
        margins_acc = grp["exp2_margin_acc"].dropna().tolist() if "exp2_margin_acc" in grp.columns else []
 
        sc_f1,  wp_f1,  tp_f1  = stat_tests(margins_f1)  if margins_f1  else (0, 1.0, 1.0)
        sc_acc, wp_acc, tp_acc = stat_tests(margins_acc) if margins_acc else (0, 1.0, 1.0)
 
        stat_rows.append({
            **key,
            "sign_count_f1":   sc_f1,
            "wilcoxon_p_f1":   wp_f1,
            "ttest_p_f1":      tp_f1,
            "sign_count_acc":  sc_acc,
            "wilcoxon_p_acc":  wp_acc,
            "ttest_p_acc":     tp_acc,
        })
 
    stats_df = pd.DataFrame(stat_rows)
    summary  = summary.merge(stats_df, on=grp_cols, how="left")
 
    out = os.path.join(RES, cfg_name, f"results_{cfg_name}.csv")
    summary.to_csv(out, index=False)
    n_seeds_str = f"{len(frames)} seeds"
    print(f" config csv written: {out}  ({n_seeds_str}, {len(summary)} tda configs) [sign_count = X/{len(frames)} seeds where combined > raw]"
    )
 
 
#aggregate the seed variants json from early implementation and backwards compatability.
def build_summaries():

    # concat config level summary
    cfg_rows = []
    for cn in SIMULATION_CONFIGS:
        p = os.path.join(RES, cn, f"results_{cn}.csv")
        if exists(p):
            df = pd.read_csv(p)
            cfg = SIMULATION_CONFIGS.get(cn, {})
            df.insert(2, "n_agents",  n_agents(cfg) if "n_groups" in cfg else None)
            df.insert(3, "timesteps", cfg.get("timesteps"))
            df.insert(4, "idle",      cfg.get("health_idle"))
            cfg_rows.append(df)
 
    if cfg_rows:
        summary = pd.concat(cfg_rows, ignore_index=True)
        summary.to_csv(os.path.join(RES, "summary.csv"), index=False)
        print(f"summary.csv written ({len(cfg_rows)} configs, {len(summary)} rows)")
    else:
        print("summary.csv: no config CSVs found")
 
    # pool individual
    all_rows = []
    for cfg_name in SIMULATION_CONFIGS:
        for seed in test_rest_seed:
            p = os.path.join(RES, cfg_name, f"seed_{seed}", f"results_s{seed}.csv")
            if exists(p):
                chunk = pd.read_csv(p)
                chunk["cfg"]  = cfg_name
                chunk["seed"] = seed
                all_rows.append(chunk)
 
    if not all_rows:
        print("pooled_summary.csv: no seed CSVs found")
        return
 
    all_data = pd.concat(all_rows, ignore_index=True)
    n_total  = len(all_data)   # total (cfg, seed, tda) observations
 

    # compute per-tda stats
    grp_cols = ["tda", "k", "lp_norm", "c_threshold", "overlap"]
    rows = []
    for keys, grp in all_data.groupby(grp_cols):
        margins_f1  = grp["exp2_margin_f1"].dropna().tolist()  if "exp2_margin_f1"  in grp.columns else []
        margins_acc = grp["exp2_margin_acc"].dropna().tolist() if "exp2_margin_acc" in grp.columns else []
        n_obs = len(margins_f1)
 
        sc_f1,  wp_f1,  tp_f1  = stat_tests(margins_f1)  if margins_f1  else (0, 1.0, 1.0)
        sc_acc, wp_acc, tp_acc = stat_tests(margins_acc) if margins_acc else (0, 1.0, 1.0)
 
        cfg_means   = grp.groupby("cfg")["exp2_margin_f1"].mean().dropna().values if "exp2_margin_f1" in grp.columns else np.array([])
        cfg_stds    = grp.groupby("cfg")["exp2_margin_f1"].std().dropna().values  if "exp2_margin_f1" in grp.columns else np.array([])
        between_var = float(np.var(cfg_means, ddof=1)) if len(cfg_means) > 1 else 0.0
        within_var  = float(np.mean(cfg_stds ** 2))    if len(cfg_stds)  > 0 else 0.0
 
        rows.append({
            "tda":                    keys[0],
            "k":                      keys[1],
            "lp_norm":                keys[2],
            "c_threshold":            keys[3],
            "overlap":                keys[4],
            "n_obs":                  n_obs,
            "pooled_mean_margin_f1":  float(np.mean(margins_f1)) if margins_f1 else None,
            "pooled_std_margin_f1":   float(np.std(margins_f1))  if margins_f1 else None,
            "pooled_sign_count_f1":   sc_f1,
            "pooled_wilcoxon_p_f1":   wp_f1,
            "pooled_ttest_p_f1":      tp_f1,
            "pooled_mean_margin_acc": float(np.mean(margins_acc)) if margins_acc else None,
            "pooled_sign_count_acc":  sc_acc,
            "pooled_wilcoxon_p_acc":  wp_acc,
            "pooled_ttest_p_acc":     tp_acc,
            "between_cfg_var":        between_var,
            "within_cfg_var":         within_var,
            "exchangeable":           between_var < within_var,
        })
 
    pooled_df = pd.DataFrame(rows).sort_values("tda")
    pooled_df.to_csv(os.path.join(RES, "pooled_summary.csv"), index=False)
 
    n_cfgs   = len(SIMULATION_CONFIGS)
    n_seeds  = len(test_rest_seed)

    actual_n  = int(pooled_df["n_obs"].iloc[0]) if len(pooled_df) > 0 else 0
    min_p     = 2 ** (-actual_n) if actual_n > 0 else 1.0
    print(f"\npooled_summary.csv written — n={actual_n} ({n_cfgs} cfgs x {n_seeds} seeds), min achievable one-tailed p ≈ {min_p:.5f}")
 
    print(f"\n{'TDA config':<22} {'mean ΔF1':>9} {'sign':>6} {'ttest_p':>9} {'wilcoxon_p':>11} {'result':>22} {'exchg':>5}")
    print("-" * 90)
    for _, r in pooled_df.iterrows():
        p = r["pooled_ttest_p_f1"]
        verdict = ("SIGNIFICANT POSITIVE" if p < 0.05 else "SIGNIFICANT NEGATIVE" if p > 0.95 else "not significant")
        print(f"{r['tda']:<22} {r['pooled_mean_margin_f1']:>+9.5f} {int(r['pooled_sign_count_f1']):>3}/{int(r['n_obs'])} {p:>9.6f} {r['pooled_wilcoxon_p_f1']:>11.6f} {verdict:>22}  {'Y' if r['exchangeable'] else 'N':>5}")
 
    build_cm_summaries()
    build_importance_summaries()
    return pooled_df
 
 
def build_cm_summaries():
    classes = None
    per_dataset = defaultdict(lambda: defaultdict(lambda: {"raw_cm": None, "combined_cm": None, "n": 0}))
    pooled      = defaultdict(lambda: {"raw_cm": None, "combined_cm": None, "n": 0})
 
    for cfg_name in SIMULATION_CONFIGS:
        for seed in test_rest_seed:
            seed_dir = os.path.join(RES, cfg_name, f"seed_{seed}")
            if not exists(seed_dir):
                continue
            for tda_name in os.listdir(seed_dir):
                exp2_path = os.path.join(seed_dir, tda_name, "exp2.json")
                if not exists(exp2_path):
                    continue
                d = jload(exp2_path)
                if "combined_cm" not in d or "raw_cm" not in d:
                    continue
                if classes is None and "classes" in d:
                    classes = d["classes"]
 
                rcm = np.array(d["raw_cm"],      dtype=int)
                ccm = np.array(d["combined_cm"], dtype=int)
 
                e = per_dataset[cfg_name][tda_name]
                e["raw_cm"]      = rcm if e["raw_cm"]      is None else e["raw_cm"]      + rcm
                e["combined_cm"] = ccm if e["combined_cm"] is None else e["combined_cm"] + ccm
                e["n"] += 1
 
                p = pooled[tda_name]
                p["raw_cm"]      = rcm if p["raw_cm"]      is None else p["raw_cm"]      + rcm
                p["combined_cm"] = ccm if p["combined_cm"] is None else p["combined_cm"] + ccm
                p["n"] += 1
 
    if classes is None:
        print("cm_summaries: no confusion matrix data found in exp2.json files")
        return
 
    def fmt(e):
        return {
            "raw_cm":      e["raw_cm"].tolist()      if e["raw_cm"]      is not None else None,
            "combined_cm": e["combined_cm"].tolist() if e["combined_cm"] is not None else None,
            "n":           e["n"],
        }
    # write per-dataset and pooled cm json outputs
    per_ds_out = {cfg: {tda: fmt(e) for tda, e in td.items()} for cfg, td in per_dataset.items()}
    jsave({"classes": classes, "data": per_ds_out}, os.path.join(RES, "cm_per_dataset.json"))
    print("cm_per_dataset.json written")
 
    pooled_out = {tda: fmt(e) for tda, e in pooled.items()}
    jsave({"classes": classes, "data": pooled_out}, os.path.join(RES, "cm_pooled.json"))
    print("cm_pooled.json written")
 
 
def build_importance_summaries():
    per_dataset = defaultdict(lambda: defaultdict(lambda: {"raw": {}, "combined": {}, "n": 0}))
    pooled      = defaultdict(lambda: {"raw": {}, "combined": {}, "n": 0})
 
    def _accumulate(target, source):
        for feat, val in source.items():
            target[feat] = target.get(feat, 0.0) + val
 
    # raw and combined importances across all result files
    for cfg_name in SIMULATION_CONFIGS:
        for seed in test_rest_seed:
            seed_dir = os.path.join(RES, cfg_name, f"seed_{seed}")
            if not exists(seed_dir):
                continue
            for tda_name in os.listdir(seed_dir):
                exp2_path = os.path.join(seed_dir, tda_name, "exp2.json")
                if not exists(exp2_path):
                    continue
                d = jload(exp2_path)
                if "combined_importances" not in d or "raw_importances" not in d:
                    continue
 
                ri = d["raw_importances"]
                ci = d["combined_importances"]
 
                e = per_dataset[cfg_name][tda_name]
                _accumulate(e["raw"],      ri)
                _accumulate(e["combined"], ci)
                e["n"] += 1
 
                p = pooled[tda_name]
                _accumulate(p["raw"],      ri)
                _accumulate(p["combined"], ci)
                p["n"] += 1
 
    def _normalise(acc, n):
        return {f: v / n for f, v in sorted(acc.items(), key=lambda x: -x[1])} if n > 0 else {}
 
    # normalise and write per-dataset and pooled importance outputs
    per_ds_out = {}
    for cfg, tda_d in per_dataset.items():
        per_ds_out[cfg] = {}
        for tda, e in tda_d.items():
            per_ds_out[cfg][tda] = {
                "raw_importances":      _normalise(e["raw"],      e["n"]),
                "combined_importances": _normalise(e["combined"], e["n"]),
                "n": e["n"],
            }
 
    jsave(per_ds_out, os.path.join(RES, "importance_per_dataset.json"))
    print("importance_per_dataset.json written")
 
    pooled_out = {
        tda: {
            "raw_importances":      _normalise(e["raw"],      e["n"]),
            "combined_importances": _normalise(e["combined"], e["n"]),
            "n": e["n"],
        }
        for tda, e in pooled.items()
    }
    jsave(pooled_out, os.path.join(RES, "importance_pooled.json"))
    print("importance_pooled.json written")
 
 
def main():
    tda_dict = TDA
    configs  = list(SIMULATION_CONFIGS.keys())
    total    = len(configs) * len(test_rest_seed) * len(tda_dict)
 
       
    print("Experiment pipeline")
    print(f" target : {TARGET}")
    print(f" seeds  : {test_rest_seed}")
    print(f" configs: {len(configs)}  TDA configs: {len(tda_dict)}  seeds: {len(test_rest_seed)}")
    print(f" total runs: {total}")
 
    # check for and generate any missing datasets
    print("\n[0] Generating datasets")
    generate_all_datasets(base=DATA)
 
    # loop over entire pipeline
    done = 0
    t_start = time.time()
 
    for cn in configs:
        cfg = SIMULATION_CONFIGS[cn]
        print(f"Config: {cn}  ({n_agents(cfg)} agents, T={cfg['timesteps']}, idle={cfg['health_idle']})")
 
        for seed in test_rest_seed:
            for tda_name, tda_params in tda_dict.items(): 
                done += 1
                elapsed = time.time() - t_start
                print(f"\n  [{done}/{total}]  seed={seed}  tda={tda_name}  (elapsed {elapsed/3600:.2f}h)")
                try:
                    run_one(cn, seed, tda_name, tda_params)
                except Exception:
                    traceback.print_exc()
                    print(" Failed, skipping, continuing with next run")
 
            # gather all tda results for this seed into one csv
            print(f"\n  gathering seed results (seed={seed})...")
            try:
                write_seed_csv(cn, seed, tda_dict)
            except Exception as e:
                print(f"seed csv failed: {e}")
 
        # write config-level csv aggregating acros seeds
        try:  
            write_config_csv(cn)   
        except Exception as e:      
            print(f" config csv failed: {e}") 
 
    print(f"\n[done] Total time: {(time.time()-t_start)/3600:.2f}h")
 
    # write summary tables
    build_summaries()
 
if __name__ == "__main__":
    main()