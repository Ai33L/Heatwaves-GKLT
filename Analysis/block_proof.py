import pickle
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics import tsaplots
import gzip

import os
os.chdir('/home/lab_abel/Heatwaves/Data/compact')

def normal():

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)    

    with gzip.open('long_series', 'rb') as f:
        T=pickle.load(f)

    anomaly=T-np.mean(T)
    anomaly=running_mean(anomaly, 24*90)

    a_list=np.linspace(0.01 ,2.1, 100)
    # a_list=np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07])
    return_list=[]
    for a_th in a_list:
        cross=[]
        prev=0
        for i in range(len(anomaly)):
            if anomaly[i]>a_th and (i-prev)>24*19:
                cross.append(i/24)
                prev=i

        cross=np.array(cross)
        diff=(cross[1:]-cross[:-1])/365
        return_list.append(np.mean(diff))
        # return_list.append(np.sum(diff*diff)/(2*1000))

    # print(return_list)

    plt.plot((return_list),a_list, label='normal')


def analyse_direct(block):

    with gzip.open('long_series', 'rb') as f:
        T=pickle.load(f)

    anomaly=T-np.mean(T)
    print('mean = ',np.mean(T))

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)    

    T90=running_mean(anomaly,24*90)

    A_max=[]
    T_block=block
    M=int(len(T90)/(24*T_block))
    for i in range(M):
        A_max.append((T90[i*24*T_block:i*24*T_block+24*T_block]).max())

    A_max=np.sort(A_max)
    A_hot=A_max[A_max>0.01]

    m=len(A_hot)

    Ret=[]
    prob=[]
    for i in range(m):
        t=-T_block/(np.log((1-(m-i)/M)))
        # t=T_block/((m-i)/M)
        Ret.append(t/365)
        prob.append(1-np.exp(-T_block/t))

    # plt.scatter((Ret),A_hot[:],s=10, alpha=0.5, facecolors='none', edgecolors='black', 
    # label='38 block')
    plt.plot((Ret),A_hot[:], 
    label='38 block')

def analyse_divide128():

    with gzip.open('long_series', 'rb') as f:
        T=pickle.load(f)

    anomaly=T-np.mean(T)
    print(np.mean(T))

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)    
    
    M=int(len(anomaly)/(24*128))
    
    A_max=[]
    for i in range(M):
        bl=anomaly[i*24*128:i*24*128+24*128]
        A_max.append(running_mean(bl, 24*90).max())

    A_max=np.sort(A_max)
    A_hot=A_max[A_max>0.01]

    m=len(A_hot)

    Ret=[]
    prob=[]
    for i in range(m):
        # t=38/((m-i)/M)
        t=-38/(np.log((1-(m-i)/M)))
        Ret.append(t/365)
        # prob.append(1-np.exp(-T_block/t))

    # plt.scatter((Ret),A_hot[:],s=10, alpha=0.5, facecolors='none', edgecolors='black', 
    # label='128 seperated blocks')
    plt.plot((Ret),A_hot[:], 
    label='128 separated blocks')

normal()
analyse_direct(38)
analyse_divide128()

plt.grid()
plt.xscale('log')
plt.xlabel('Return time (years)')
plt.ylabel('Anomaly of 90 day avg temperature (K)')
plt.legend()
# plt.xlim([0,70000000])
plt.xticks([1/10,1,10,100,1000,10000,100000,1000000,10000000])
plt.ylim([0,3])
plt.show()