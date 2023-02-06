import pickle
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics import tsaplots
import gzip

import os
os.chdir('/home/lab_abel/Heatwaves/Data/compact')

def auto(size):

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)    

    with gzip.open('long_series', 'rb') as f:
        T=pickle.load(f)

    anomaly=T-np.mean(T)
    anomaly90=running_mean(anomaly, 24*90)

    aut=tsaplots.acf(anomaly[::24],nlags=size)
    plt.plot(np.array(range(len(aut))), aut)
    plt.xlabel('Time(Days)')
    plt.ylabel('Autocorrelation')
    plt.grid()
    plt.show()

auto(100)