import warnings
warnings.filterwarnings("ignore")
# import os
# os.environ['OPENMP_NUM_THREADS']='2'

import climt
from sympl import (
    PlotFunctionMonitor, NetCDFMonitor,
    TimeDifferencingWrapper, get_constant
)
import numpy as np
from datetime import timedelta
import pickle
import time
import matplotlib.pyplot as plt
import mgzip

def load_state(state, core, filename):
    
    with mgzip.open(filename, 'rb') as f:
        fields, spec = pickle.load(f)
    
    core._gfs_cython.reinit_spectral_arrays(spec)
    # core.set_flag(False)
    
    state.update(fields)

def plot_function(fig, state):
    
    
    fig.set_size_inches(6, 10)
    
    net_heat_flux = (
            state['downwelling_shortwave_flux_in_air'][-1] +
            state['downwelling_longwave_flux_in_air'][-1] -
            state['upwelling_shortwave_flux_in_air'][-1] -
            state['upwelling_longwave_flux_in_air'][-1])
    
    flux=(net_heat_flux*np.cos(state['latitude'].values[:])).mean()
    print(-flux.item())

    ax = fig.add_subplot(4, 2, 1)
    state['surface_temperature'].plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title(state['time'])

    ax = fig.add_subplot(4, 2, 3)
    state['eastward_wind'].mean(dim='lon').plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title('Zonal Wind')

    ax = fig.add_subplot(4, 2, 2)
    state['boundary_layer_height'].plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title(flux.item())

    ax = fig.add_subplot(4, 2, 4)
    state['air_temperature'].mean(dim='lon').plot.contourf(
        ax=ax, levels=16)
    ax.set_title('Temperature')
    
    ax = fig.add_subplot(4, 2, 5)
    state['surface_upward_sensible_heat_flux'].plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title('Sensible flux')
    
    ax = fig.add_subplot(4, 2, 6)
    state['surface_upward_latent_heat_flux'].plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title('Latent flux')
    
    ax = fig.add_subplot(4, 2, 7)
    state['specific_humidity'][0].plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title('Specific humidity(lowest lvl)')
    
    ax = fig.add_subplot(4, 2, 8)
    state['air_temperature'][0].plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title('Air temperature(lowest lvl)')

    fig.tight_layout()

import os
os.chdir('/home/lab_abel/Thesis_rem')

monitor = PlotFunctionMonitor(plot_function)

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

grid = climt.get_grid(nx=128, ny=62)
# grid = climt.get_grid(nx=12, ny=16)

# Create model state
my_state = climt.get_default_state([dycore], grid_state=grid)
dycore(my_state, model_time_step)

# load_state(my_state, dycore, '../Thesis/GKTL/Data/spinup/spinup_2year')
 
# st = time.time()
for i in range(100):#26280 one year
    
    if i==1:
        dycore.set_flag(True)

    diag, my_state = dycore(my_state, model_time_step)
    my_state.update(diag)
    my_state['time'] += model_time_step

# et = time.time()

# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')
spec = dycore._gfs_cython.get_spectral_arrays()
with mgzip.open('restore', 'wb') as f:
    pickle.dump([my_state,spec], f)

# load_state(my_state, dycore, 'restore')

for i in range(1):#26280 one year
    
    if i==1:
        dycore.set_flag(True)

    diag, my_state = dycore(my_state, model_time_step)
    my_state.update(diag)
    my_state['time'] += model_time_step

spec = dycore._gfs_cython.get_spectral_arrays()
with mgzip.open('state1', 'wb') as f:
    pickle.dump([my_state,spec], f)
