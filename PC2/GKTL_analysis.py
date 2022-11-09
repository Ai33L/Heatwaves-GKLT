import mgzip
import pickle
import matplotlib.pyplot as plt
import numpy as np

import os
os.chdir('/home/requiem/Thesis/GKTL/Data/GKTL')

a=[]
p_all=[]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)        

stat=['stats25_k50a','stats25_k40a','stats25_k20a','stats25_k10a']

for e in stat:
    with mgzip.open(e, 'rb') as f:
        T,A,W,p = pickle.load(f)
        for i in p:
            p_all.append(i/len(stat))
        for i in A:
            a.append(np.max(running_mean(i,6480)))
    # print(a)

# print(p)
# for i in range(10):
#     plt.plot(A[i])
# plt.show()

a, p_all = zip(*sorted(zip(a, p_all)))
print(a)
print(p_all)

r=[]
for i in range(len(a)):
    # print(np.sum(p_all[i:]))
    T=-(128-90)/(np.log(1-np.sum(p_all[i:])))
    # T=(128-90)/np.sum(p_all[i:])
    r.append(T/365)

# print(r)

plt.plot(np.log10(r[:]), np.array(a)-285.7635480)

with mgzip.open('return_plot_ctrl', 'rb') as f:
    T,A = pickle.load(f)

plt.plot(T,A)
plt.show()
# # print(p)
# for e in range(len(A)):
#     plt.plot(A[e])
# plt.show()
# # print(np.sum(np.array(p)))