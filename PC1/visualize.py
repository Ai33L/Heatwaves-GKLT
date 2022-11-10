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
import mgzip

def load_state(state, core, filename):
    
    with mgzip.open(filename, 'rb') as f:
        fields, spec = pickle.load(f)
    
    core._gfs_cython.reinit_spectral_arrays(spec)
    core.set_flag(False)
    
    state.update(fields)

def plot_function(fig, state):
    
    
    fig.set_size_inches(6, 10)
    
    net_heat_flux = (
            state['downwelling_shortwave_flux_in_air'][-1] +
            state['downwelling_longwave_flux_in_air'][-1] -
            state['upwelling_shortwave_flux_in_air'][-1] -
            state['upwelling_longwave_flux_in_air'][-1])
    
    flux=(net_heat_flux*np.cos(state['latitude'].values[:])).mean()

    ax = fig.add_subplot(4, 2, 1)
    state['surface_temperature'].plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title(state['time'])

    ax = fig.add_subplot(4, 2, 3)
    state['eastward_wind'].mean(dim='lon').plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title('Zonal Wind')

    ax = fig.add_subplot(4, 2, 2)
    (state['area_type'].astype(str)=='land').plot.contourf(
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
os.chdir('/home/lab_abel/Heatwaves-GKTL/PC1')

monitor = PlotFunctionMonitor(plot_function, interactive=False)

with mgzip.open('Data/clim_test', 'rb') as f:
    my_state= pickle.load(f)

monitor.store(my_state)