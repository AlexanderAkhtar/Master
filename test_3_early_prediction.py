# experiment 3 early prediction
# Tests how early in the simulation a target can be predicted



 
import os
import json
 
from test_2_mixed_model import run_predictive_comparison
 
# runs the experiment 3
def run_early_prediction(df, target_col, tda_build_graph, build_paths, time_slices=(0.25, 0.5, 0.75, 1.0), k_folds=3, tda_params=(5, 2.0, 5, True), min_movement_threshold=0.0, raw_cache_dir=None, static_target=True,):
    
    max_t = df["timestep"].max()
    
    
    if static_target:
        true_labels = (df.sort_values("timestep").groupby("agent_id").last().reset_index()[["agent_id", target_col]])
 
    results = {}
 
    for pct in time_slices:
        cutoff = int(max_t * pct)
        print(f"  Slice {pct*100:.0f}% (t<={cutoff}/{max_t})...")
 
        sliced_raw = df[df["timestep"] <= cutoff].copy()
        
        
        if static_target:
            sliced = sliced_raw.drop(columns=[target_col]).merge(true_labels, on="agent_id")
        else:
            sliced = sliced_raw.copy()
            n_classes = sliced.groupby("agent_id").last()[target_col].nunique()
            if n_classes <= 1:
                print(f"    WARNING: only {n_classes} distinct class(es) at slice {pct*100:.0f}% — skipping (not classifiable).")
                continue
 
        if sliced["agent_id"].nunique() == 0:
            continue
 
        cached_raw = None
        cache_file = None
        if static_target and raw_cache_dir is not None:
            suffix = str(pct).replace(".", "")
            cache_file = os.path.join(
                raw_cache_dir, f"exp3_raw_cache_slice_{suffix}.json"
            )
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    cached_raw = json.load(f)
 
        res = run_predictive_comparison(df=sliced, target_col=target_col, tda_build_graph=tda_build_graph, build_paths=build_paths, k_folds=k_folds, tda_params=tda_params, min_movement_threshold=min_movement_threshold, precomputed_raw_scores=cached_raw,)
 
        if (static_target and cached_raw is None and cache_file is not None and "raw_cache" in res):
            os.makedirs(raw_cache_dir, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(res["raw_cache"], f)
 
        results[f"slice_{pct}"] = {
            "pct_trajectory_used": pct,
            "target_label_mode":   "static" if static_target else "at_cutoff",
            "taxonomy_f1":         res["taxonomy"]["f1_mean"],
            "taxonomy_acc":        res["taxonomy"]["acc_mean"],
            "raw_f1":              res["raw"]["f1_mean"],
            "raw_acc":             res["raw"]["acc_mean"],
            "combined_f1":         res["combined"]["f1_mean"],
            "combined_acc":        res["combined"]["acc_mean"],
            "p_value":             res["p_combined_vs_raw"],
        }
 
    return results
