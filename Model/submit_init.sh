#!/bin/bash
#SBATCH --ntasks-per-node=8
#SBATCH --time=96:00:00
#SBATCH --job-name=init_run
#SBATCH --error=err.err
#SBATCH --output=out.out
#SBATCH --partition=standard
#SBATCH -N 1
#SBATCH --exclude=cn020,cn029,cn166,gpu002,cn001,cn005,cn097,gpu001,cn143,cn164,cn035,cn086,cn087,cn088,cn081,cn034,gpu027,cn080,cn028,gpu024,cn168,cn002,cn085,cn116,cn042,cn161,cn110
#SBATCH --array=4-5

export LD_LIBRARY_PATH=/usr/lib64/libseccomp.so.2
module load intel/2018.5.274
module load singularity/3.2.1
#module load python/3.7
#module list
#env| grep LIB
#locate libseccomp
#singularity run CLIMT.sif
#held_suarez.py
singularity exec climt_rare.sif python init_and_clim.py "$SLURM_ARRAY_TASK_ID"
#mpirun -n 48 singularity exec CLIMT.sif python held_suarez.py
#mpirun -n 96 /home/arnab/ritu/software/gmx20194_plmd254_impi_dbl/bin/gmx_mpi_d mdrun -deffnm MD_ATP_onlywater_long3
