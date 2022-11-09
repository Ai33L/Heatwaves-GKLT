from logging import lastResort
from re import L
from tkinter import NE
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

def save_state(state, core, filename):
    
    spec = core._gfs_cython.get_spectral_arrays()

    with mgzip.open(filename, 'wb') as f:
        pickle.dump([state,spec], f)

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
os.chdir('/home/lab_abel/Thesis/GKTL/Data/spinup')

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

# Create model state
my_state = climt.get_default_state([dycore], grid_state=grid)
dycore(my_state, model_time_step)

load_state(my_state, dycore, 'spinup_2year')
 

# setup fields to save
# Spec_hum=[]
Air_temp=[]
# Air_press_int=[]
# Surf_air_press=[]
# East_wind=[]
# North_wind=[]
Time=[]
Longitude=my_state['longitude'].values[:]
Latitude=my_state['latitude'].values[:]
Area_type=my_state['area_type']
# Down_long=[]
# Down_short=[]
# Up_long=[]
# Up_short=[]
# Latent_flux=[]
# Sensible_flux=[]

# monitor=NetCDFMonitor('../long_run', store_names=['specific_humidity', 'air_temperature',
# 'air_pressure_on_interface_levels', 'surface_air_pressure', 'eastward_wind', 'northward_wind',
# 'time', 'longitude', 'latitude', 'area_type', 'downwelling_longwave_flux_in_air',
# 'downwelling_shortwave_flux_in_air','upwelling_longwave_flux_in_air',
# 'upwelling_shortwave_flux_in_air', 'surface_upward_latent_heat_flux',
# 'surface_upward_sensible_heat_flux'])

for i in range(26280*100):#26280 one year
    
    if i==1:
        dycore.set_flag(True)

    diag, my_state = dycore(my_state, model_time_step)
    my_state.update(diag)
    my_state['time'] += model_time_step

    if (i+1)%720==0:
        n=int((i+1)/720)
        save_state(my_state, dycore, '../initial/state'+str(n))
        
    if i%2160==0:
        monitor.store(my_state)

    #saving fields
    if (i+1)%18==0:
        # Spec_hum.append(my_state['specific_humidity'].values[:])
        Air_temp.append(my_state['air_temperature'].values[0])
        # Air_press_int.append(my_state['air_pressure_on_interface_levels'].values[:])
        # Surf_air_press.append(my_state['surface_air_pressure'].values[:])
        # East_wind.append(my_state['eastward_wind'].values[:])
        # North_wind.append(my_state['northward_wind'].values[:])
        Time.append(my_state['time'])
        # Down_long.append(my_state['downwelling_longwave_flux_in_air'].values[:])
        # Down_short.append(my_state['downwelling_shortwave_flux_in_air'].values[:])
        # Up_long.append(my_state['upwelling_longwave_flux_in_air'].values[:])
        # Up_short.append(my_state['upwelling_shortwave_flux_in_air'].values[:])
        # Latent_flux.append(my_state['surface_upward_latent_heat_flux'].values[:])
        # Sensible_flux.append(my_state['surface_upward_sensible_heat_flux'].values[:])

    if (i+1)%26280==0:
        n=int((i+1)/26280)
        with mgzip.open('../long_run_temp/temp_year'+str(n), 'wb') as f:
            # pickle.dump([Spec_hum,Air_temp,Air_press_int,Surf_air_press,East_wind,North_wind,Time,
            # Longitude,Latitude, Area_type,Down_long,Down_short,Up_long,Up_short,Latent_flux,
            # Sensible_flux], f)
            pickle.dump([Air_temp,Time,Longitude,Latitude,Area_type], f)
        
        # Spec_hum=[]
        Air_temp=[]
        # Air_press_int=[]
        # Surf_air_press=[]
        # East_wind=[]
        # North_wind=[]
        Time=[]
        # Down_long=[]
        # Down_short=[]
        # Up_long=[]
        # Up_short=[]
        # Latent_flux=[]
        # Sensible_flux=[]
    
    # monitor.store(my_state)
    # monitor.write()
