from turtle import onscreenclick
import mgzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics import tsaplots

import os
os.chdir('/home/lab_abel/Thesis/GKTL/Data/long_run_temp')

with mgzip.open('temp_year1', 'rb') as f:
        Air_temp,Time,Longitude,Latitude,Area_type = pickle.load(f)

obs_mask=np.zeros(Latitude.shape)

for i in range(128):
    for j in range(62):
        lat = Latitude[j,i]
        lon = Longitude[j,i]
        if (lat>20 and lat<40) and (lon>100 and lon<150): 
            obs_mask[j,i]=1

def Obs(field):

    O=field*obs_mask
    return(O.sum()/obs_mask.sum())

# with mgzip.open('temp_year1', 'rb') as f:
#     Air_temp,Time,Longitude,Latitude,Area_type = pickle.load(f)

# Air_temp_avg=0
# count=0
# for e in range(1,101):

#     with mgzip.open('temp_year'+str(e), 'rb') as f:
#         Air_temp,Time,Longitude,Latitude,Area_type = pickle.load(f)
        
#         for i in Air_temp:
#             Air_temp_avg+=i
#             count+=1

# Air_temp_avg=Air_temp_avg/count

# A=[]
# for e in range(1,101):

#     with mgzip.open('temp_year'+str(e), 'rb') as f:
#         Air_temp,Time,Longitude,Latitude,Area_type = pickle.load(f)
        
#         for i in Air_temp:
#             A.append(Obs(i-Air_temp_avg))

# print(A)

# with mgzip.open('anomaly_A20_50', 'wb') as f:
#         pickle.dump(A, f)

# with mgzip.open('T_avg', 'wb') as f:
#         pickle.dump(Air_temp_avg, f)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)    

with mgzip.open('anomaly_A20_50', 'rb') as f:
        A = pickle.load(f)

A_mean90=running_mean(A,4*90)
# print(np.array(A).mean())
# print(A_mean90.max())
# plt.plot(np.array(range(len(A)))/1460,A)
# plt.plot(np.array(range(len(A)-359))/1460,A_mean90, label='anomaly')
# plt.xlabel('Years')
# plt.ylabel('Anomaly of 90 day avg temperature')
# # plt.grid()
# plt.legend()
# plt.show()

# aut=tsaplots.acf(A,nlags=4*20)
# plt.plot(aut)
# plt.show()

# with mgzip.open('T_avg', 'rb') as f:
#     T_avg = pickle.load(f)

# # print(Obs(T_avg))

# levels=np.linspace(210,300,10,endpoint=True)
# plt.contourf(Longitude, Latitude,T_avg,levels)
# plt.colorbar()
# # plt.contourf(Longitude, Latitude,obs_mask[mask], alpha=0.5)
# plt.contourf(Longitude[16:-37], Latitude[16:-37], T_avg[16:-37], levels,hatches=['/'])
# plt.contourf(Longitude[38:-15], Latitude[38:-15], T_avg[38:-15], levels,hatches=['/'])
# plt.contourf(Longitude[16:-37,35:54], Latitude[16:-37,35:54], T_avg[16:-37,35:54], levels,hatches=['/////'])
# # plt.contourf(Longitude, Latitude, obs_mask==1,alpha=0.5)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()

A_max=[]
T_block=100
M=int(len(A_mean90)/(4*T_block))
print(M)
for i in range(M):
    A_max.append((A_mean90[i*4*T_block:i*4*T_block+4*T_block]).max())

# plt.plot(A_max)
# plt.show()

A_max=np.sort(A_max)
A_hot=A_max[A_max>0.01]
# plt.plot(A_hot)
# plt.show()

# plt.plot(A_max)
# plt.show()
m=len(A_hot)

Ret=[]
prob=[]
for i in range(m):
    t=-T_block/(np.log((1-(m-i)/M)))
    Ret.append(t/365)
    prob.append(1-np.exp(-T_block/t))

plt.scatter(A_hot[:],prob,label='100 year long run',s=0.8)
plt.xlabel('Anomaly of 90 day avg temperature')
plt.ylabel('q(a)')
plt.grid()
plt.legend()
plt.show()

with mgzip.open('../GKTL/PDF_ctrl', 'wb') as f:
    pickle.dump([A_hot[:],prob],f)

for e in A_hot:
    print(e)
plt.plot((Ret),A_hot[:],label='100 yr long run')
plt.xlabel('Return time in yrs')
plt.ylabel('Anomaly of 90 day avg temperature')
plt.grid()
plt.xscale('log')
plt.legend()

with mgzip.open('../GKTL/return_plot_ctrl', 'wb') as f:
    pickle.dump([Ret,A_hot[:]],f)

# with mgzip.open('../GKTL/return_plot', 'rb') as f:
#     T,A = pickle.load(f)

# plt.plot(T,A)

plt.show()