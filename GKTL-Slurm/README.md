## GKTL implementation -  Slurm

#### Algorithm run

The algorithm is run with the script **run.sh**, which automates the entire algorithm procedure.

```
 nohup ./run.sh &
```
  
The experiments to be run are defined at the end of the file.

#### Scripts and setup

* *Initialising the experiment*
	+ Each experiment is setup with the *GKTL_init.py* script using the *submit_init.sh* job script.
	+ The script takes target directory, selection coefficient, number of trajectories and the initial states directory as input.
	+ The script creates the target directory and populates it with an initial condition for each trajectory, sampled randomly from the initial condition directory.
	+ Experiment parameters like the selection coefficient are set, and other necessary files for the algorithm are created.

* *Trajectory simulation*
	+ The trajectories are simulated with the *GKTL_traj.py* script using the *submit_traj_ser.sh* job script.
	+ The target directory, iteration number and the trajectory index are passed as input to the script, which simulates each trajectory from the correct initial condition.
	+ This script runs in parallel, with multiple trajectory simulations running at once.
	+ Each instance of the model runs a trajectory from an initial condition and stores relevant data. The final states of each simulation are stored as *traj_new\<>* files.
	+ The data generated from the trajectories are written in files with name format *data*_\<*iteration>*_*\<traj_num>*. The data files for each trajectory in the algorithm are tracked in the *link* file.
 