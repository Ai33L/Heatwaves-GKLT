from cProfile import label
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
    'stats50_k50a', 'stats50_k50b'
    ]

label_list=['GKTL k=10','GKTL k=20a','GKTL k=20b', 'GKTL k=40a', 'GKTL k=40b','GKTL k=50']
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
    index=0
    for i in range(len(a)):
        if a[i]>0:
            break
        index+=1
    
    a=a[index:];p=p[index:]
    r=[]
    for i in range(len(a)):
        T=-(128-90)/np.log(1-np.sum(p[i:]))
        # T=(128-90)/np.sum(p_all[i:])
        r.append(T/365)

    f = interp1d(r, a)
    r_interpol=r_interpol_range[np.logical_and(r_interpol_range>r[0], r_interpol_range<r[-1])]
    a_interpol=f(r_interpol)
    # plt.plot(np.log10(r_interpol),(np.array(a_interpol)), label=label_list[e], alpha=0.4)
    # plt.plot(np.log10(r),(np.array(a)), label=label_list[m], alpha=0.8)

    r_pool.append(r_interpol); a_pool.append(a_interpol)

    # plt.scatter(a,np.cumsum(p[::-1])[::-1] )

    r=[]
    a=[]
    p=[]


with mgzip.open('../GKTL/PDF_ctrl', 'rb') as f:
    a_ctrl,p_ctrl=pickle.load(f)

# plt.plot(a_ctrl,p_ctrl, label='PDF from 100 year run')
# plt.xlabel('Anomaly of 90 day avg temperature')
# plt.ylabel('PDF')
# plt.grid()
# plt.legend()
# # plt.xlim([1.2,2.6])
# # plt.ylim([-0.005,0.03])
# plt.show()

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

plt.plot(r_final,a_final, label='Averaged GKTL',color='darkviolet',linewidth=1.5)

# z = np.polyfit(np.log10(r_final), a_final, 2)
# P=np.polyval(z,np.log10(r_final))
# plt.plot(np.log10(r_final),P, label='GKTL run',color='darkviolet',linewidth=1.5)

with mgzip.open('return_plot_ctrl', 'rb') as f:
    T,A = pickle.load(f)

plt.plot(T,A, label='100 yr run', color='black',linewidth=1.5)
plt.xlabel('log10(Return time in yrs)')
plt.ylabel('Anomaly of 90 day avg temperature')
plt.grid()
plt.legend()
plt.xscale('log')
plt.show()