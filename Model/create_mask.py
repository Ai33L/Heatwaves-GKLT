import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('/home/lab_abel/Heatwaves-GKTL/PB/GKTL_PB')

with gzip.open('common', 'rb') as f:
    common=pickle.load(f)

lon=common[0]
lat=common[1]

obs_mask=np.zeros(lat.shape)

for i in range(128):
    for j in range(62):
        if (lat[j,i]>20 and lat[j,i]<40) and (lon[j,i]>100 and lon[j,i]<130): 
            obs_mask[j,i]=1

plt.contourf(lon, lat,obs_mask)
plt.show()

with gzip.open('mask', 'wb') as f:
    pickle.dump(obs_mask ,f)

def Obs(state):

    O=state['air_temperature'][0].values[:]*obs_mask
    return(O.sum()/obs_mask.sum())