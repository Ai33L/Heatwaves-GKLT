## GKTL implementation -  Slurm

#### Algorithm run

The algorithm is run with the script **run.sh**, which automates the entire algorithm procedure.

```
 nohup ./run.sh &
```
  
The experiment details are setup at the end of the file.

#### Scripts and setup

* Initialising the experiment
	+ Each experiment is setup with the **GKTL_init.py** script using the **submit_init.sh** job script.
	+ The script takes target directory, selection coefficient, number of trajectories and the initial states directory as input.
	+ 