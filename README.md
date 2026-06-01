# Master
Code base for master thesis 

\textbf{\texttt{road\_evac\_sim.py}} implements the multi-agent evacuation simulation described in Chapter~\ref{ch:mas}. It creates the agent populations, runs the simulation, and records the simulated datasets.

\textbf{\texttt{dp\_tda.py}} implements the Mapper pipeline described in Chapter~\ref{ch:tda}. It does interval construction, calculates within interval distance, within interval clustering, and taxonomy generation, and the test agent projection function.
 
\textbf{\texttt{sim\_configs.py}} contains the parameter dictionary for the five simulation configurations, the random seed described in Section~\ref{sec:sim_configs}. It also runs the \textbf{\texttt{road\_evac\_sim.py}} to generate the datasets used in this research project's experiments.

Of note for the files that run the model comparisons, the function for the model comparison is in the \textbf{\texttt{test\_2\_mixed\_model.py}}. As such all experiment files import the comparison functions from the \textbf{\texttt{test\_2\_mixed\_model.py}} file.

\textbf{\texttt{hp\_sweep.py}} implements the stage one hyperparameter sweep, and the grid of 144 Mapper configurations described in (Section~\ref{sec:sweep}). It also records the results of the sweep and also generates the results summary files.
 
\textbf{\texttt{test\_1\_baseline\_clustering.py}} implements Experiment 1 (Section~\ref{sec:exp1}), applying $k$-means clustering to flattened trajectories as a lower-bound baseline.
 
\textbf{\texttt{test\_2\_mixed\_model.py}} implements Experiment 2 (Section~\ref{sec:exp2}), including the \texttt{build\_raw\_features} and \texttt{build\_taxonomy\_features} functions and the stratified cross-validation model comparison function.
 
\textbf{\texttt{test\_3\_early\_prediction.py}} implements Experiment 3 (Section~\ref{sec:exp3}), the temporal truncation of the dataset before running the model comparison.
 
\textbf{\texttt{data\_vis.py}} contains the code used for visualizing the trajectory data and the taxonomy data.
 
\textbf{\texttt{main\_test.py}} is a common file that takes a manually input set of Mapper configurations and runs the set of stage two experiments on them. This file also records the results and aggregates them into the summary files. 


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
