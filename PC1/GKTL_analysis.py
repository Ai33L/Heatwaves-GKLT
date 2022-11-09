import mgzip
import pickle
import matplotlib.pyplot as plt
import numpy as np

import os
os.chdir('/home/lab_abel/Thesis/GKTL/Data/GKTL')

a=[]
p_all=[]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)        

stat=[
    'stats50_k10a','stats50_k10b']

for e in stat:
    with mgzip.open(e, 'rb') as f:
        T,A,W,p = pickle.load(f)
    for i in p:
        p_all.append(i/2)
    for i in A:
         a.append(np.max(running_mean(i[:],360*18)))
    # plt.scatter(a,p)
    # a=[]
    # print(a)
# plt.show()

# print(p)
# for i in range(10):
#     plt.plot(A[i])
# plt.show()

a, p_all = zip(*sorted(zip(a, p_all)))
# print(a)
# print(p_all)

r=[]
for i in range(len(a)):
    # print(np.sum(p_all[i:]))
    T=-(128-90)/np.log(1-np.sum(p_all[i:]))
    # T=(128-90)/np.sum(p_all[i:])
    r.append(T/365)

a=np.array(a)-285.76354801081936
# r=[]
# a_plot=[0.01]
# while a_plot[-1]<a[-1]:
#     p_tot=0
#     for i in range(len(a)):
#         if a[i]>a_plot[-1]:
#             p_tot+=p_all[i]
#     T=-(128-90)/np.log(1-p_tot)
#     # T=(128-90)/np.sum(p_all[i:])
#     r.append(T/365)

#     a_plot.append(a_plot[-1]*1.05)

# print(r)

plt.plot(np.log10(r[:]),(np.array(a[:])), label='GKTL algorithm')

# with mgzip.open('return_plot', 'wb') as f:
#         pickle.dump([np.log10(r[:]),(np.array(a)-285.76354801081936)], f)

with mgzip.open('return_plot_ctrl', 'rb') as f:
    T,A = pickle.load(f)

plt.plot(T,A, label='100 yr run')
plt.xlabel('log10(Return time) in yrs')
plt.ylabel('Anomaly of 90 day avg temperature')
plt.grid()
plt.legend()
plt.show()

# # for e in range(len(A)):
# #     plt.plot(A[e])
# # plt.show()
# print(np.sum(np.array(p)))