import cProfile
import warnings
warnings.filterwarnings("ignore")
# import os
# os.environ['OPENMP_NUM_THREADS']='2'

import climt
from sympl import (
    PlotFunctionMonitor,
    TimeDifferencingWrapper, get_constant
)
import numpy as np
from datetime import timedelta
import pickle
import time
import matplotlib.pyplot as plt
from climt import bolton_q_sat
from statsmodels.graphics import tsaplots
import mgzip

def load_state(state, core, filename):
    
    with mgzip.open(filename, 'rb') as f:
        fields, spec = pickle.load(f)
    
    core._gfs_cython.reinit_spectral_arrays(spec)
    core.set_flag(False)
    
    state.update(fields)

import os
os.chdir('/home/lab_abel/Thesis/GKTL/Data/spinup')

with mgzip.open('spinup_2year', 'rb') as f:
    my_state = pickle.load(f)[0]

surface_shape = my_state['latitude'].shape
obs_mask=np.zeros(surface_shape)

#setting the land
for i in range(128):
    for j in range(62):
        lat = my_state['latitude'][j,i]
        lon = my_state['longitude'][j,i]
        if (lat>20 and lat<40) and (lon>100 and lon<150): 
            obs_mask[j,i]=1

def Obs(state):

    O=state['air_temperature'][0].values[:]*obs_mask
    return(O.sum()/obs_mask.sum())

climt.set_constants_from_dict({
    'stellar_irradiance': {'value': 200, 'units': 'W m^-2'}})

model_time_step = timedelta(minutes=20)


# Create components
convection = climt.EmanuelConvection()
boundary=TimeDifferencingWrapper(climt.SimpleBoundaryLayer(scaling_land=0.5))
radiation = climt.GrayLongwaveRadiation()
slab_surface = climt.SlabSurface()

dycore = climt.GFSDynamicalCore(
    [boundary,radiation, slab_surface,
    convection], number_of_damped_levels=5
)
dycore.set_flag(True)

grid = climt.get_grid(nx=128, ny=62)

# Create model state
my_state = climt.get_default_state([dycore], grid_state=grid)
dycore(my_state, model_time_step)

load_state(my_state, dycore, 'spinup_2year')

A=[]  
for i in range(720*10):#720 - 10 days 
    
    if i==1:
        dycore.set_flag(True)
    A.append(Obs(my_state))
    diag, my_state = dycore(my_state, model_time_step)
    my_state.update(diag)
    my_state['time'] += model_time_step

    if i%720==0:
        net_heat_flux = (
            my_state['downwelling_shortwave_flux_in_air'][-1] +
            my_state['downwelling_longwave_flux_in_air'][-1] -
            my_state['upwelling_shortwave_flux_in_air'][-1] -
            my_state['upwelling_longwave_flux_in_air'][-1])
    
        flux=(net_heat_flux*np.cos(my_state['latitude'].values[:])).mean()
        print(flux.item())

aut=tsaplots.acf(A,nlags=720*10)

file = open('../Aut_data', 'wb')
pickle.dump([A, aut], file)
file.close()

# file = open('../Aut_data', 'rb')
# A, aut=pickle.load(file)
# file.close()

plt.plot(np.array(range(len(A)))/72, A)
plt.xlabel('Time(Days)')
plt.ylabel('Observation(K)')
plt.grid()
plt.show()

plt.plot(np.array(range(len(aut)))/72, aut)
plt.xlabel('Time(Days)')
plt.ylabel('Autocorrelation')
plt.grid()
plt.show()