import mgzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import os
os.chdir('/home/lab_abel/Thesis/GKTL/Data/GKTL')

a=[]
p=[]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)        

stat=['stats50_k10a','stats50_k10b',
    'stats50_k20a','stats50_k20b',
    'stats50_k40a','stats50_k40b',
    'stats50_k50a','stats50_k50b'
    ]

label_list=['GKTL k=10','GKTL k=20', 'GKTL k=40','GKTL k=50']
r_interpol_range=np.linspace(0,100000,10000001, endpoint=True)

r_pool=[]
a_pool=[]

for e in range(len(stat)):
    with mgzip.open(stat[e], 'rb') as f:
        T,A_temp,W,p_temp = pickle.load(f)
    for i in range(len(p_temp)):
        p.append(p_temp[i])
        a.append(np.max(running_mean(A_temp[i][::18],360))-285.76354801081936)
            
    a, p = zip(*sorted(zip(a, p)))
    mean=np.mean(a); std=np.std(a)
    front=0;rear=len(a)
    for i in range(len(a)):
        if a[i]<mean-std:
            front=i+1
        if a[i]<mean+std:
            rear= i+1
    
    p_array=[]
    # p_cum=np.cumsum(p[::-1])[::-1]
    r=[]
    for i in range(len(a)):
        T=-(128-90)/np.log(1-np.sum(p[i:]))
        # T=(128-90)/np.sum(p_all[i:])
        r.append(T/365)
        p_array.append(1-np.exp(-40/T))

    a=a[front:rear];r=r[front:rear];p=p[front:rear];p_array=p_array[front:rear]
    f = interp1d(r, a)
    r_interpol=r_interpol_range[np.logical_and(r_interpol_range>r[0], r_interpol_range<r[-1])]
    a_interpol=f(r_interpol)
    # plt.plot((r_interpol),(np.array(a_interpol)), label=label_list[e], alpha=1)
    # plt.plot(np.log10(r),(np.array(a)), label=label_list[e], alpha=0.8)

    r_pool.append(r_interpol); a_pool.append(a_interpol)

    plt.scatter(a,p_array,color='darkviolet',s=10)

    r=[]
    a=[]
    p=[]

with mgzip.open('../GKTL/PDF_ctrl', 'rb') as f:
    a_ctrl,p_ctrl=pickle.load(f)

plt.plot(a_ctrl,p_ctrl, label='100 year long run',color='black', linewidth='1.5')
plt.xlabel('Anomaly of 90 day avg temperature')
plt.ylabel('q(a)')
plt.grid()
plt.legend()
# plt.xlim([1.2,2.6])
# plt.ylim([-0.005,0.03])
plt.yscale('log')
plt.show()

a_final=[]
r_final=[]

c=np.zeros(r_interpol_range.shape)
a_temp=np.zeros(r_interpol_range.shape)
for i in range(len(r_pool)):
    index=np.where(r_interpol_range==r_pool[i][0])[0][0]
    a_temp[index:index+len(r_pool[i])]=a_temp[index:index+len(r_pool[i])]+np.array(a_pool[i])
    c[index:index+len(r_pool[i])]+=1

for i in range(len(c)):
    if c[i]!=0:
        a_final.append(a_temp[i]/c[i])
        r_final.append(r_interpol_range[i])

# plt.plot(r_final,a_final, label='GKTL averaged',color='darkviolet',linewidth=1.5)

# z = np.polyfit(np.log10(r_final), a_final, 5)
# P=np.polyval(z,np.log10(r_final))
# plt.plot(np.log10(r_final),P, label='GKTL run',color='red',linewidth=1.5)

with mgzip.open('return_plot_ctrl', 'rb') as f:
    T,A = pickle.load(f)
plt.plot((T),A, label='100 yr run', color='black',linewidth=2)

plt.xlabel('Return time in yrs')
plt.ylabel('Anomaly of 90 day avg temperature')
plt.grid()
plt.legend()
plt.xscale('log')
plt.show()