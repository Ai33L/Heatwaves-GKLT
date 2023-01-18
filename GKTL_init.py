import random
import shutil
import numpy as np
import pickle
import gzip

# Initialises everything the algorithm needs
# Creates a directory and adds initial conditions randomly selected from spinup
# Creates a status file which keeps track

import os
import sys
dir=str(sys.argv[1])
k_val=int(sys.argv[2])
Num_traj=int(sys.argv[3])
dir_init=str(sys.argv[4])

# initialise run parameters
Num_iter = 16
tau = 8
Ta = 128

initial_sample = random.sample((range(1,1095)),Num_traj)

R_log=0

with open(dir+'/R_log', 'wb') as f:
    pickle.dump([R_log], f)

with open(dir+'/config', 'wb') as f:
    pickle.dump([k_val, Num_traj, tau, Ta], f)

for i in range(Num_traj):
    shutil.copy(dir_init+'/state'+str(initial_sample[i]), dir+'/traj'+str(i+1))

T=[]
for i in range(Num_traj):
    T.append([])

with open(dir+'/T', 'wb') as f:
    pickle.dump(T, f)

pert_flag=np.zeros(Num_traj)
with open(dir+'/pert_flag', 'wb') as f:
    pickle.dump(pert_flag, f)

link=[]
for i in range(Num_traj):
    link.append([])

with open(dir+'/link', 'wb') as f:
    pickle.dump(link, f)