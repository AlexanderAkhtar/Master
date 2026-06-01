To reproduce the thesis datasets and run the thesis test follow this procedure:
1. run sim_config.py - the two datasets used in the thesis will be produced.
2. run hp_sweep.py - the thesis sweep dataset will then be used for the parameter sweep.
3. set the selected mapper configurations in the TDA dict (name : (parameters)) of main_test.py.
4. run main_test.py - the thesis main experiment suite dataset will then be run with the configurations selected.



To run individual simulation runs:
1. change the parameters to desired.
2. run road_evac_sim.py.

To run alternative hyperparameter sweep:
1. change the parameters to desired.
2. set new RES (trajectory data file path) and SWEEP_SEEDS (the seeds used to run the road_evac_sim).

To run alternative main experiment suite:
1. change parameters to desired.
2. set new RES (trajectory data file path) and TDA dict (dict of the name and parameters for the mapper config).