#parameter sweep
 
import os
import json
import time
import itertools
from collections import defaultdict
 
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_1samp
 
from dp_tda import load_data, build_paths, tda_build_graph, load_and_set_geodesic
from test_2_mixed_model import run_predictive_comparison
from sim_configs import SIMULATION_CONFIGS, n_agents
 
 
SWEEP_DATA =    "./test_01_dataset"
RES =           "./results/sweep"
TARGET =        "behavior"
MMT  = 0.0      #holdover not used
SWEEP_SEEDS = [1, 2]  # run each sim config with these seeds 
SWEEP_FOLDS = 3

 
# mapper grid search values
K_VALUES       = [1, 2, 3, 5, 8, 12, 15, 17, 20]
LP_NORM_VALUES = [2.0, float("inf")]   # lp norm value
C_THRESHOLD    = [3, 5, 8, 10]
OV_VALUES      = [True, False]
 
# 9 x 2 x 4 x 2 = 144 combos
SWEEP_GRID = list(itertools.product(K_VALUES, LP_NORM_VALUES, C_THRESHOLD, OV_VALUES))
 
 
# helpers

#construct config name 
def sweep_name(k, lp_norm, c_threshold, ov):
    lp_str = "lp2" if lp_norm == 2.0 else "lpinf"
    ov_str = "ov" if ov else "nov"
    return f"k{k}_{lp_str}_c{c_threshold}_{ov_str}"
 

#data helpers 
def jfmt(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(type(o))
 
def jsave(d, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=4, default=jfmt)
 
def jload(path):
    with open(path) as f: return json.load(f)
 
def exists(path): return os.path.exists(path)
 
def safe_k(df, col, k):
    mn = df.drop_duplicates("agent_id")[col].value_counts().min()
    if mn < k:
        print(f"  min class={mn} < k={k}, reducing folds")
        return int(mn)
    return k
 
 

# runs one sweep config
#  
def run_sweep_one(cfg_name, seed, k, lp_norm, c_threshold, ov):
    name       = sweep_name(k, lp_norm, c_threshold, ov)
    out_dir    = os.path.join(RES, cfg_name, f"seed_{seed}", name)
    cache_path = os.path.join(out_dir, "sweep.json")
    if exists(cache_path):
        return jload(cache_path)
 
    csv_path = os.path.join(SWEEP_DATA, cfg_name, f"seed_{seed}", "trajectories.csv")
    if not exists(csv_path):
        print(f"    missing: {csv_path}"); return None
 
    geo_path = os.path.join(SWEEP_DATA, cfg_name, f"seed_{seed}", "geodesic.json")
    if not exists(geo_path):
        print(f"    missing geodesic: {geo_path}"); return None
    load_and_set_geodesic(geo_path)
 
    df = load_data(csv_path)
    kf = safe_k(df, TARGET, SWEEP_FOLDS)
 
    # raw scores cached per (cfg, seed) same across all config
    raw_cache_path = os.path.join(RES, cfg_name, f"seed_{seed}", "raw_scores_cache.json")
    cached_raw = jload(raw_cache_path) if exists(raw_cache_path) else None
 
    try:
        res = run_predictive_comparison(df=df, target_col=TARGET, tda_build_graph=tda_build_graph, build_paths=build_paths, k_folds=kf, tda_params=(k, lp_norm, c_threshold, ov), min_movement_threshold=MMT, precomputed_raw_scores=cached_raw)
 
        if cached_raw is None and "raw_cache" in res:
            jsave(res["raw_cache"], raw_cache_path)
 
        rec = {
            "name":                name,
            "k":                   k,
            "lp_norm":             lp_norm,
            "c_threshold":         c_threshold,
            "overlap":             ov,
            "seed":                seed,
            "combined_f1":         res["combined"]["f1_mean"],
            "raw_f1":              res["raw"]["f1_mean"],
            "taxonomy_f1":         res["taxonomy"]["f1_mean"],
            "margin_f1":           res["combined"]["f1_mean"] - res["raw"]["f1_mean"],
            "combined_acc":        res["combined"]["acc_mean"],
            "raw_acc":             res["raw"]["acc_mean"],
            "margin_acc":          res["combined"]["acc_mean"] - res["raw"]["acc_mean"],
            "classes":             res.get("classes"),
            "raw_cm":              res.get("raw_cm"),
            "combined_cm":         res.get("combined_cm"),
            "raw_importances":     res.get("raw_importances"),
            "combined_importances": res.get("combined_importances"),
        }
        jsave(rec, cache_path)
        return rec
 
    except Exception as e:
        print(f"    error: {e}"); return None
 
 
#runs the full sweep
 
def run_sweep():
    configs = list(SIMULATION_CONFIGS.keys())
    n_total = len(SWEEP_GRID) * len(configs) * len(SWEEP_SEEDS)
    done, t0 = 0, time.time()
 
    print(f"HP SWEEP  {len(SWEEP_GRID)} TDA configs x {len(configs)} sim configs x {len(SWEEP_SEEDS)} seeds")
    print(f"Total: {n_total} runs\n")
 
    # sweep_data[tda_name][cfg_name] is list of per-seed data. seeds are averaged per (tda, cfg) 
    sweep_data = defaultdict(lambda: defaultdict(list))
 
    for cfg_name in configs:
        cfg = SIMULATION_CONFIGS[cfg_name]
        print(f"\nConfig: {cfg_name}  ({n_agents(cfg)} agents, T={cfg['timesteps']}, idle={cfg['health_idle']})")
 
        for seed in SWEEP_SEEDS:
            print(f"  seed={seed}")
            for k, lp_norm, c_threshold, ov in SWEEP_GRID:
                done += 1
                name = sweep_name(k, lp_norm, c_threshold, ov)
                print(f"    [{done}/{n_total}] {name}", end="  ", flush=True)
                rec = run_sweep_one(cfg_name, seed, k, lp_norm, c_threshold, ov)
                if rec:
                    sweep_data[name][cfg_name].append(rec)
                    print(f"comb_f1={rec['combined_f1']:.3f}  margin_f1={rec['margin_f1']:+.3f}")
                else:
                    print("FAILED")
 
    # results table for each (tda_name, cfg_name), average combined_f1 and margin across seeds before computing the across-cfg statistics
    rows = []
    for name, cfg_dict in sweep_data.items():
        if not cfg_dict:
            continue
 
        cfg_combined_f1s = []
        cfg_margins_f1 = []
        cfg_combined_acc = []
        cfg_margins_acc = []
        for cfg_name, recs in cfg_dict.items():
            for r in recs:                               # one entry per seed
                cfg_combined_f1s.append(r["combined_f1"])
                cfg_margins_f1.append(r["margin_f1"])
                cfg_combined_acc.append(r["combined_acc"])
                cfg_margins_acc.append(r["margin_acc"])
 
        n_obs = len(cfg_margins_f1)  # = number of sim configs (datasets)
 
        # sign_count, datasets where combined beats raw
        sign_count_f1  = sum(1 for m in cfg_margins_f1      if m > 0)
        sign_count_acc = sum(1 for m in cfg_margins_acc if m > 0)
 
        # F1 stat tests
        wilcoxon_pvalue_f1 = 1.0
        if n_obs >= 3 and any(m != 0 for m in cfg_margins_f1):
            try:
                _, wilcoxon_pvalue_f1 = wilcoxon(cfg_margins_f1, alternative="greater", zero_method="zsplit")
                wilcoxon_pvalue_f1 = float(wilcoxon_pvalue_f1)
            except Exception:
                pass
 
        ttest_pvalue_f1 = 1.0
        if n_obs >= 3:
            try:
                ttest_pvalue_f1 = float(ttest_1samp(cfg_margins_f1, 0.0, alternative="greater").pvalue)
            except Exception:
                pass
 
        # accuracy statistical tests 
        wilcoxon_pvalue_acc = 1.0
        if n_obs >= 3 and any(m != 0 for m in cfg_margins_acc):
            try:
                _, wilcoxon_pvalue_acc = wilcoxon(cfg_margins_acc, alternative="greater", zero_method="zsplit")
                wilcoxon_pvalue_acc = float(wilcoxon_pvalue_acc)
            except Exception:
                pass
 
        ttest_pvalue_acc = 1.0
        if n_obs >= 3:
            try:
                ttest_pvalue_acc = float(ttest_1samp(cfg_margins_acc, 0.0, alternative="greater").pvalue)
            except Exception:
                pass
 
        first_rec = next(iter(cfg_dict.values()))[0]
        rows.append({
            "name":                  name,
            "k":                     first_rec["k"],
            "lp_norm":               first_rec["lp_norm"],
            "c_threshold":           first_rec["c_threshold"],
            "overlap":               first_rec["overlap"],
            #f1
            "mean_margin_f1":        float(np.mean(cfg_margins_f1)),
            "std_margin_f1":         float(np.std(cfg_margins_f1)),
            "mean_combined_f1":      float(np.mean(cfg_combined_f1s)),
            "std_combined_f1":       float(np.std(cfg_combined_f1s)),
            "sign_count_f1":         sign_count_f1,       # X/n_obs datasets improved
            "wilcoxon_pvalue_f1":    wilcoxon_pvalue_f1,  # underpowered at n=4
            "ttest_pvalue_f1":       ttest_pvalue_f1,     # underpowered at n=4
            # acc
            "mean_margin_acc":  float(np.mean(cfg_margins_acc)),
            "std_margin_acc":   float(np.std(cfg_margins_acc)),
            "mean_combined_acc":float(np.mean(cfg_combined_acc)),
            "std_combined_acc": float(np.std(cfg_combined_acc)),
            "sign_count_acc":        sign_count_acc,
            "wilcoxon_pvalue_acc":   wilcoxon_pvalue_acc,
            "ttest_pvalue_acc":      ttest_pvalue_acc,

            "n_datasets":            n_obs,
            "n_seeds":               len(SWEEP_SEEDS),
        })
 
    if not rows:
        print("\n No successful sweep runs.")
        return pd.DataFrame()
 
    df = pd.DataFrame(rows)
 
    # borda count rank configs per sim config avg combined_f1
    borda = defaultdict(int)
    for cfg_name in configs:
        scores = {}
        for name in sweep_data:
            if cfg_name in sweep_data[name] and sweep_data[name][cfg_name]:
                scores[name] = float(np.mean( [r["combined_f1"] for r in sweep_data[name][cfg_name]]))
        for rank, name in enumerate(sorted(scores, key=lambda x: scores[x], reverse=True), 1):
            borda[name] += rank
 
    df["borda_score"] = df["name"].map(borda).fillna(999).astype(int)
    df = df.sort_values("borda_score").reset_index(drop=True)
 
    results_path = os.path.join(RES, "sweep_results.csv")
    os.makedirs(RES, exist_ok=True)
    df.to_csv(results_path, index=False)
 
    print(f"\nSweep complete in {(time.time()-t0)/3600:.2f}h")
    print(f"Results -> {results_path}")
    n_ds    = int(df["n_datasets"].iloc[0]) if len(df) > 0 else 0
    n_cfgs  = len(configs)
    n_seeds = len(SWEEP_SEEDS)
    min_p   = 2 ** (-n_ds)
    print(f"\n(sign_count = X/{n_ds} individual (cfg, seed) runs where combined > raw; {n_cfgs} cfgs x {n_seeds} seeds; min achievable one-tailed p ≈ {min_p:.4f})\n")
 
    # result display
    print("F1 ")
    print(df[[
        "name", "k", "lp_norm", "c_threshold", "overlap", "mean_margin_f1", "std_margin_f1", "mean_combined_f1", "std_combined_f1", "sign_count_f1", "wilcoxon_pvalue_f1", "ttest_pvalue_f1", "borda_score",
    ]].to_string(index=False))
 
    print("\n Accuracy ")
    print(df[[ "name", "k", "lp_norm", "c_threshold", "overlap", "mean_margin_acc", "std_margin_acc", "mean_combined_acc", "std_combined_acc", "sign_count_acc", "wilcoxon_pvalue_acc", "ttest_pvalue_acc", "borda_score",
    ]].to_string(index=False))

    build_sweep_cm_summaries(sweep_data)
    build_sweep_importance_summaries(sweep_data)

    return df


#builds the summary file of the results
def build_sweep_cm_summaries(sweep_data):
    classes = None
    pooled  = defaultdict(lambda: {"raw_cm": None, "combined_cm": None, "n": 0})

    for tda_name, cfg_dict in sweep_data.items():
        for cfg_name, recs in cfg_dict.items():
            for rec in recs:
                if rec.get("combined_cm") is None or rec.get("raw_cm") is None:
                    continue
                if classes is None and rec.get("classes"):
                    classes = rec["classes"]

                rcm = np.array(rec["raw_cm"],      dtype=int)
                ccm = np.array(rec["combined_cm"], dtype=int)

                p = pooled[tda_name]
                p["raw_cm"]      = rcm if p["raw_cm"]      is None else p["raw_cm"]      + rcm
                p["combined_cm"] = ccm if p["combined_cm"] is None else p["combined_cm"] + ccm
                p["n"] += 1

    if classes is None:
        print("sweep cm_summaries: no CM data found")
        return

    out = {
        tda: {
            "raw_cm":      e["raw_cm"].tolist()      if e["raw_cm"]      is not None else None,
            "combined_cm": e["combined_cm"].tolist() if e["combined_cm"] is not None else None,
            "n":           e["n"],
        }
        for tda, e in pooled.items()
    }
    path = os.path.join(RES, "sweep_cm_pooled.json")
    os.makedirs(RES, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"classes": classes, "data": out}, f, indent=4, default=jfmt)
    print(f"sweep_cm_pooled.json written {path}")


#second summary resutl
def build_sweep_importance_summaries(sweep_data):
    pooled = defaultdict(lambda: {"raw": {}, "combined": {}, "n": 0})

    def _accumulate(target, source):
        for feat, val in source.items():
            target[feat] = target.get(feat, 0.0) + val

    for tda_name, cfg_dict in sweep_data.items():
        for cfg_name, recs in cfg_dict.items():
            for rec in recs:
                ri = rec.get("raw_importances")
                ci = rec.get("combined_importances")
                if not ri or not ci:
                    continue
                p = pooled[tda_name]
                _accumulate(p["raw"],      ri)
                _accumulate(p["combined"], ci)
                p["n"] += 1

    def _normalise(acc, n):
        return {f: v / n for f, v in sorted(acc.items(), key=lambda x: -x[1])} if n > 0 else {}

    out = {
        tda: {
            "raw_importances":      _normalise(e["raw"],      e["n"]),
            "combined_importances": _normalise(e["combined"], e["n"]),
            "n": e["n"],
        }
        for tda, e in pooled.items()
    }
    path = os.path.join(RES, "sweep_importance_pooled.json")
    os.makedirs(RES, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=4, default=jfmt)
    print(f"sweep_importance_pooled.json written -> {path}")
 
 
if __name__ == "__main__":
    run_sweep()