#!/bin/bash
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --job-name=GKTL_init
#SBATCH --partition=standard
#SBATCH --exclude=cn020,cn029,cn166,gpu002
#SBATCH -N 1

pass=0
while [ $pass -ne 1 ]
do

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib64:$LIBRARY_PATH
module load intel/2018.5.274
module load singularity/3.2.1

if singularity exec climt_rare.sif python GKTL_init.py $1 $2 $3 $4; then
pass=1
else
sleep 10
fi
done
~                                                                                                                                                                                                           
~                                                                                                                                                                                                           
~                                                                                                                                                                                                           
~                                    