## GKTL implementation -  Slurm

#### Algorithm run

The algorithm is run with the script **run.sh**, which automates the entire algorithm procedure.

```
 nohup ./run.sh &
```

Algorithm parameters are set in the *GKTL_init.py* script. Experiment parameters are set at the end of the *run.sh* script.

#### Files and setup
* GKTL_init.py and submit_init.sh
* GKTL_traj.py and submit_traj_ser.sh
* GKTL_resampling.py and submit_resample.sh
* GKTL_wrap.py and submit_wrap.sh
* An initial state pool directory with files named in format *state<>*
* A *mask* file, which contains the mask array used for observable computation

***

### Scripts and setup

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
	+ The data generated from the trajectories are written in files with name format *data*_\<*iteration>*_*\<traj_num>*.
		+ Trajectory observable is stored at 20 minute frequency
		+ Lowest model level air temperature field is stored at 1 hour frequency
		+ 2D surface fields are stores at 6 hour frequency
		+ 3D fields like air temperature and wind are stored at 24 hour frequency 
	* The data files for each trajectory in the algorithm are tracked in the *link* file.

* *Trajectory resampling*
	+ The trajectory resampling is performed with the *GKTL_resampling.py* script using the *submit_resample.sh* job script.
	+ The target directory and iteration number are passed as arguments to this script.
	+ The script calculates weights for the trajectories obtained in simulation and modifies the R_log value.
	+ Based on the calculated weights, the number of clones for each trajectory is determined. The cloning procedure is carried out, and the trajectories with no clones are killed off.
	+ The data files of the killed trajectories are removed, and the other files are updated to reflect the cloning.

* *End of experiment and calculations*
	+ Each experiment is concluded by the *GKTL_wrap.py* script using the *submit_wrap.sh* job script.
	+ This script only takes the target directory as input.
	+ The script calculates the probability of each trajectory from the observable and the R_log value, and writes it to file.
	+ Temporary files created during algorithm run are removed and the directory is cleaned up.

***


 