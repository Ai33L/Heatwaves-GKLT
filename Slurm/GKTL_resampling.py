import gzip
import pickle
import numpy as np
import random
import shutil

import os
# os.chdir('/home/lab_abel/Heatwaves-GKTL/PB/GKTL_PB/k_10_1')

import sys
dir=str(sys.argv[1])
iter=int(sys.argv[2])

with open(dir+'/config', 'rb') as f:
    config = pickle.load(f)

k=config[0]/365
Num_traj=config[1]
tau=config[2]

with open(dir+'/link', 'rb') as f:
    link = pickle.load(f)

for i in range(Num_traj):
    link[i]=link[i]+[str(iter)+'_'+str(i+1)]

W=np.zeros(Num_traj)

with open(dir+'/T', 'rb') as f:
    T = pickle.load(f)

for i in range(Num_traj):
    with gzip.open(dir+'/A'+str(i+1), 'rb') as f:
        A = pickle.load(f)
    T[i]=T[i]+A
    W[i]=np.exp(np.array(A).mean()*tau*k)

R_ext=W.sum()/Num_traj

with open(dir+'/R_log', 'rb') as f:
    R = pickle.load(f)

R=R+np.log(R_ext)

with open(dir+'/R_log', 'wb') as f:
    pickle.dump(R, f)

W=W/R_ext

N=np.zeros(Num_traj)
for i in range(Num_traj):
    N[i]=np.trunc(W[i]+np.random.uniform())

diff=int(Num_traj-N.sum())

if diff>0:
    for i in range(diff):
        while True:
            ind=random.randrange(0,Num_traj)
            if int(N[ind]):
                N[ind]+=1
                break

if diff<0:
    for i in range(abs(diff)):
        while True:
            ind=random.randrange(0,Num_traj)
            if int(N[ind]):
                N[ind]-=1
                break

for e in range(Num_traj):
    os.remove(dir+'/traj'+str(e+1))

print(N)
pert_flag=np.ones(Num_traj)
ind=1
T_new=[]
link_delete=[]
link_new=[]
for e in range(len(N)):
    if int(N[e])==0:
        link_delete.append(link[e])
    for i in range(int(N[e])):
        shutil.copy(dir+'/traj_new'+str(e+1), dir+'/traj'+str(ind))
        T_new.append(T[e])
        link_new.append(link[e])
        if i==0:
            pert_flag[ind-1]=0
        ind+=1

for e in link_delete:
    flag=0
    for i in range(len(e)):
        if flag==1:
            if os.path.exists(dir+'/data_temp_'+e[i]):
                os.remove(dir+'/data_temp_'+e[i])
                os.remove(dir+'/data2D_'+e[i])
                os.remove(dir+'/data3D_'+e[i])
        else:
            check=0
            for f in link_new:
                if f[i]==e[i]:
                    check=1
            if check==0:
                if os.path.exists(dir+'/data_temp_'+e[i]):
                    os.remove(dir+'/data_temp_'+e[i])
                    os.remove(dir+'/data2D_'+e[i])
                    os.remove(dir+'/data3D_'+e[i])
                flag=1

with open(dir+'/link', 'wb') as f:
    pickle.dump(link_new, f)

with open(dir+'/T', 'wb') as f:
    pickle.dump(T_new, f)

with open(dir+'/pert_flag', 'wb') as f:
    pickle.dump(pert_flag, f)

for e in range(Num_traj):
    os.remove(dir+'/traj_new'+str(e+1))

with open(dir+'/resample_log'+str(iter), 'wb') as f:
    pickle.dump([], f)

