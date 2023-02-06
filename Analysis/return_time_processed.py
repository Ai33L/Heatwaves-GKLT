import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


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
        A_max.append(np.max(running_mean((np.array(e)-298.3797386241071)[:], 3*24*90)))

    a, p = zip(*sorted(zip(A_max, p)))
    
    # print(np.sum(p))

    # index=0
    # for i in range(len(a)):
    #     if a[i]>0.01:
    #         break
    #     index+=1
    
    # a=a[index:];p=p[index:]

    r=[]
    for i in range(len(a)):
        T=-(128-90)/np.log(1-np.sum(p[i:]))
        # T=(128-90)/np.sum(p[i:])
        r.append((T)/365)

    # plt.plot(r,a, label='GKTL k=10', color='red', linewidth='0.8')
    
    mean=np.mean(a); std=np.std(a)
    front=0;rear=len(a)
    for i in range(len(a)):
        if a[i]<mean-2.0*std:
            front=i+1
        if a[i]<mean+2.0*std:
            rear= i+1

    a=a[front:rear];r=r[front:rear]
    
    # r_interpol_range=np.linspace(0,1000000,10000001, endpoint=True)
    r_interpol_range=np.power(10, np.linspace(-0.5,6,100000, endpoint=True))
    f = interp1d(r, a)
    r_interpol=r_interpol_range[np.logical_and(r_interpol_range>np.min(r), r_interpol_range<np.max(r))]
    a_interpol=f(r_interpol)

    # plt.plot(r_interpol,a_interpol, label=exp)
    return r_interpol,a_interpol


analyse_direct(38)

r=np.power(10, np.linspace(-0.5,6,100000, endpoint=True))
print(r.min(), r.max())
c=np.zeros(len(r))
val=np.zeros(len(r))
val2=np.zeros(len(r))

for exp in  ['10b', '20', '20b', '30', '40', '40b', '50' ]:
    r_temp, a_temp=analyse_GKTL(exp)
    mask=np.logical_and(r>=r_temp.min(), r<=r_temp.max())
    val[mask]=val[mask]+a_temp
    val2[mask]=val2[mask]+a_temp*a_temp
    c[mask]=c[mask]+1

a_avg=val/c
std=np.sqrt(val2/c-(val*val)/(c*c))
print(std.min(),std.max())

plt.plot(r,a_avg, label='GKTL run',color='red',linewidth=1.5)
# plt.fill_between(r,a_avg-std,a_avg+std,color="grey",alpha=0.3)
plt.grid()
plt.xscale('log')
plt.xlabel('Return time (years)')
plt.ylabel('Anomaly of 90 day avg temperature (K)')
plt.legend()
# plt.xlim([0,70000000])
plt.xticks([1/10,1,10,100,1000,10000,100000,1000000,1000000])
plt.ylim([0,3])
plt.show()


analyse_direct(38)

# front=0;rear=len(c)
# stop=0
# for i in range(len(c)):
#     if c[i]>=1 and stop==0:
#         front=i+1
#         stop=1
#     if c[i]>=3:
#         rear= i+1

# a_avg=a_avg[front:rear];r=r[front:rear];std=std[front:rear]

z = np.polyfit(np.log10(r), a_avg, 5)
P=np.polyval(z,np.log10(r))
plt.plot(r,P, label='GKTL run',color='red',linewidth=1.5)
plt.fill_between(r,P-std,P+std,color='grey',alpha=0.2)

plt.grid()
plt.xscale('log')
plt.xlabel('Return time (years)')
plt.ylabel('Anomaly of 90 day avg temperature (K)')
plt.legend()
# plt.xlim([0,70000000])
plt.xticks([1/10,1,10,100,1000,10000,100000,1000000,1000000])
plt.ylim([0,3])
plt.show()