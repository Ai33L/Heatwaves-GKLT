import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np

import os
os.chdir('/home/lab_abel/Heatwaves/Data/compact')

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
        Ret.append(t/365)
        prob.append(1-np.exp(-T_block/t))

    plt.scatter((Ret),A_hot[:],s=10, alpha=0.5, facecolors='none', edgecolors='black', 
    label='1000 year direct run')
    # plt.plot((Ret),A_hot[:], 
    # label='1000 year direct run')

def analyse_GKTL(exp):

    with open('prob_'+str(exp), 'rb') as f:
        p=pickle.load(f)

    with open('T_'+str(exp), 'rb') as f:
        T=pickle.load(f)

    # with open('link_'+str(exp), 'rb') as f:
    #     link=pickle.load(f)

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)  

    A_max=[]
    for e in T:
        A_max.append(np.max(running_mean(np.array(e)-298.3797386241071, 3*24*90)))

    a, p = zip(*sorted(zip(A_max, p)))
    
    index=0
    for i in range(len(a)):
        if a[i]>0.7:
            break
        index+=1
    
    a=a[index:];p=p[index:]
    print(np.sum(p))

    r=[]
    for i in range(len(a)):
        T=-(128-90)/np.log(1-np.sum(p[i:]))
        # T=(128-90)/np.sum(p[i:])
        r.append((T)/365)

    # plt.plot(r,a, label='GKTL k=10', color='red', linewidth='0.8')
    
    mean=np.mean(r); std=np.std(r)
    front=0;rear=len(r)
    for i in range(len(r)):
        if r[i]<mean-1.0*std:
            front=i+1
        if r[i]<mean+1.0*std:
            rear= i+1
        
    a=a[front:rear];r=r[front:rear]
    plt.plot(r,a, label=exp)
    return r,a


analyse_direct(38)

# r=[];a=[]
# r_temp, a_temp=analyse_GKTL('10_new')
# r=r+r_temp;a=a+list(a_temp)

# r_temp, a_temp=analyse_GKTL('20_new')
# r=r+r_temp;a=a+list(a_temp)

# r_temp, a_temp=analyse_GKTL('40_new')
# r=r+r_temp;a=a+list(a_temp)

# r_temp, a_temp=analyse_GKTL('50_new')
# r=r+r_temp;a=a+list(a_temp)

# z = np.polyfit(np.log10(r), a, 5)
# P=np.polyval(z,np.log10(r))
# plt.plot(r,P, label='GKTL run',color='red',linewidth=1.5)

plt.grid()
plt.xscale('log')
plt.xlabel('Return time (years)')
plt.ylabel('Anomaly of 90 day avg temperature (K)')
plt.legend()
# plt.xlim([0,70000000])
plt.xticks([1/10,1,10,100,1000,10000,100000,1000000,10000000])
plt.ylim([0,3])
plt.show()