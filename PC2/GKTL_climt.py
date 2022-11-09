import warnings
warnings.filterwarnings("ignore")
import climt
from sympl import (
    PlotFunctionMonitor,
    TimeDifferencingWrapper,get_constant
)
import numpy as np
from datetime import timedelta
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import time
import multiprocessing
from climt import bolton_q_sat
import mgzip
import shutil
import copy

import sys
k_val=int(sys.argv[1])
#Setting model parameters and global arrays
Num_traj=25
Ta=128
tau=8
num_iter=int(Ta/tau)
K=k_val/365
R_log=0
W_pool=[];Traj_pool=[];A_pool=[];T_pool=[]
W=np.ones(Num_traj);C=np.ones(Num_traj)
A=[];T=[]
pert_flag=0

#to perturb the log surface pressure spectral array
def perturb(X):
    X=np.array(X)
    N=np.random.uniform(-1,1,np.shape(X[4]))*np.sqrt(2)*10**-4
    X[4][:]=(X[4]+N)[:]

#load state from memory
def load_state(state, core, filename):
    
    with mgzip.open(filename, 'rb') as f:
        fields, spec = pickle.load(f)
    
    if pert_flag:
        perturb(spec)
        core.set_flag(False)
    core._gfs_cython.reinit_spectral_arrays(spec)
    
    state.update(fields)

#save state to memory
def save_state(state, core, filename):
    
    spec = core._gfs_cython.get_spectral_arrays()

    with mgzip.open(filename, 'wb') as f:
        pickle.dump([state,spec], f)

import os
os.chdir('/home/requiem/Thesis/GKTL/Data/GKTL')

#setting the observation mask
with mgzip.open('../initial/state1', 'rb') as f:
        State = pickle.load(f)[0]

obs_mask=np.zeros(State['latitude'].shape)

for i in range(128):
    for j in range(62):
        lat = State['latitude'][j,i]
        lon = State['longitude'][j,i]
        if (lat>20 and lat<40) and (lon>100 and lon<150): 
            obs_mask[j,i]=1

#calculate observation from state
def Obs(state):

    O=state['air_temperature'][0].values[:]*obs_mask
    return(O.sum()/obs_mask.sum())


#Trajectory run
def Traj(m,n):

    model_time_step = timedelta(minutes=20)

    # Create components
    convection = climt.EmanuelConvection()
    boundary=TimeDifferencingWrapper(climt.SimpleBoundaryLayer(scaling_land=0.5))
    radiation = climt.GrayLongwaveRadiation()
    slab_surface = climt.SlabSurface()

    dycore = climt.GFSDynamicalCore(
        [boundary,radiation, slab_surface,
        convection], number_of_damped_levels=5
    )
    grid = climt.get_grid(nx=128, ny=62)

    # Create model state
    my_state = climt.get_default_state([dycore], grid_state=grid)
    dycore(my_state, model_time_step)

    load_state(my_state, dycore, 'traj'+str(m))

    T_ext=[]
    A_ext=[]
    for i in range(72*tau):#72 - one day
        
        if i==1:
            dycore.set_flag(True)
        
        A_ext.append(Obs(my_state))

        if (i+1)%18==0:
            T_ext.append(my_state['air_temperature'].values[0])

        diag, my_state = dycore(my_state, model_time_step)
        my_state.update(diag)
        my_state['time'] += model_time_step

    save_state(my_state, dycore, 'traj_'+str(m)+'-'+str(n))    

    w = np.exp(np.array(A_ext).mean()*tau*K)

    return(A_ext, T_ext, w, str(m)+'-'+str(n))


#sampling randomly from initial state pool
initial_sample = random.sample((range(1,501)),Num_traj)

#setting up trajectory and global arrays
for i in range(Num_traj):
    shutil.copy('../initial/state'+str(initial_sample[i]),os.getcwd())
    os.rename('state'+str(initial_sample[i]), 'traj'+str(i+1))
    A.append([])
    T.append([])

#to have a unique name for data files
uniq=1

#loop over iterations
for j in range(num_iter):

    #loop over trajectories
    for i in range(Num_traj):
        #Calculate count from second iteration onwards (First iteration all trajectories have one clone)
        if j:
            C[i]=int(np.trunc(W[i]+np.random.uniform()))

        #run C[i] trajectories ith trajectory  
        pert_flag=0
        for k in range(int(C[i])):
            if k:
                pert_flag=1
            A_temp, T_temp,W_temp, Traj_temp=Traj(i+1,uniq)
            uniq+=1
            W_pool.append(W_temp); Traj_pool.append(Traj_temp)
            A_pool.append(A[i]+A_temp); T_pool.append(T[i]+T_temp)

    pert_flag=1
    #randomly kill trajectories if cloned number is larger than original
    if len(W_pool)>Num_traj:
        c=len(W_pool)-Num_traj
        for m in range(c):
            ind=random.randint(0, len(W_pool)-1)
            os.remove('traj_'+Traj_pool[ind]) 
            W_pool.pop(ind);Traj_pool.pop(ind)
            A_pool.pop(ind);T_pool.pop(ind)
    
    #clone randomly seleted trajecotries if number of clones is less than original
    elif len(W_pool)<Num_traj:
        c=Num_traj-len(W_pool)
        for m in range(c):
            while True:
                ind=random.randint(0, Num_traj-1)
                if int(C[ind]):
                    A_temp, T_temp, W_temp, Traj_temp=Traj(ind+1,uniq)
                    uniq+=1
                    W_pool.append(W_temp); Traj_pool.append(Traj_temp)
                    A_pool.append(A[ind]+A_temp); T_pool.append(T[ind]+T_temp)
                    break
    
    #save the data to global arrays
    W[:]=W_pool
    R=W.sum()/Num_traj
    R_log=R_log+np.log(R)
    W=W/R
    
    A=copy.deepcopy(A_pool);T=copy.deepcopy(T_pool)

    #clean up temporary data and clear the pools for next iteration
    for e in range(len(Traj_pool)):
        os.remove('traj'+str(e+1)) 
        os.rename('traj_'+Traj_pool[e],'traj'+str(e+1))

    W_pool=[];Traj_pool=[]
    A_pool=[];T_pool=[]

    pert_flag=1

#Final calculations and saving stats
lamb=R_log/Ta
p=np.zeros(Num_traj)

for i in range(Num_traj):

    p[i]=(1/Num_traj)*(np.exp(Ta*lamb-K*np.array(A[i]).mean()*Ta))

with mgzip.open('stats25_k'+str(k_val)+'a', 'wb') as f:
        pickle.dump([T,A,W,p], f)

#clean up data
for e in range(Num_traj):
        os.remove('traj'+str(e+1))