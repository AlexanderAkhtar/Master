# main thesis datasets generator
# dataset layout - sim_dataset/{config_name}/seed_{seed}/trajectories.csv

import os
from road_evac_sim import road_network_sim

test_01_seed = [1, 2]
test_rest_seed = list(range(101, 111))

SIMULATION_CONFIGS = {
    
    "base":         {"n_groups": (150, 30, 18, 9), "timesteps": 20, "health_idle": 0.35, "k_nn": 4},
    "sparse":       {"n_groups": (150, 30, 18, 9), "timesteps": 20, "health_idle": 0.35, "k_nn": 2},
    "connected":    {"n_groups": (150, 30, 18, 9), "timesteps": 20, "health_idle": 0.35, "k_nn": 5},
    "low_health":   {"n_groups": (150, 30, 18, 9), "timesteps": 20, "health_idle": 0.25, "k_nn": 4},
    "max_health":   {"n_groups": (150, 30, 18, 9), "timesteps": 20, "health_idle": 0.45, "k_nn": 4},
}


# counts the number of agents across all groups
def n_agents(cfg):
    g = cfg["n_groups"]
    return g[0] + g[1]*2 + g[2]*3 + g[3]*4


# runs a simulation run
def run_single_simulation(cfg_name, seed, base="sim_dataset"):
    out = os.path.join(base, cfg_name, f"seed_{seed}")
    csv = os.path.join(out, "trajectories.csv")

    if os.path.exists(csv):
        return csv  # if already generated then skip

    cfg = SIMULATION_CONFIGS[cfg_name]  
    print(f"  {cfg_name}/seed_{seed} {n_agents(cfg)} agents, T={cfg['timesteps']}, idle={cfg['health_idle']}) ", end=" ", flush=True) 

    road_network_sim(cfg=cfg, seed=seed, out_dir=out)
    

    
    return csv


# runs a collection of sim configs by a set of seeds
def generate_all_datasets(configs=None, seeds=None, base="sim_dataset"):
    if configs is None: configs = list(SIMULATION_CONFIGS.keys())
    if seeds   is None: seeds   = test_01_seed

    if seeds == test_01_seed: base = "test_01_dataset"  
    if seeds == test_rest_seed: base = "test_rest_dataset"

    total = len(configs) * len(seeds)
    print(f"Generating {len(configs)} configs x {len(seeds)} seeds = {total} datasets") 



    paths = {}
    for cn in configs:
        paths[cn] = {}
        for s in seeds:
            paths[cn][s] = run_single_simulation(cn, s, base)


    print("Datasets ready!")
    return paths


if __name__ == "__main__":  
    generate_all_datasets() 
    generate_all_datasets(seeds=test_rest_seed)
