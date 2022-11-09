from ast import Interactive
from code import interact
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
import mgzip

def save_state(state, core, filename):
    
    spec = core._gfs_cython.get_spectral_arrays()

    with mgzip.open(filename, 'wb') as f:
        pickle.dump([state,spec], f)

Flux=[]

def plot_function(fig, state):
    
    
    fig.set_size_inches(6, 10)
    
    net_heat_flux = (
            state['downwelling_shortwave_flux_in_air'][-1] +
            state['downwelling_longwave_flux_in_air'][-1] -
            state['upwelling_shortwave_flux_in_air'][-1] -
            state['upwelling_longwave_flux_in_air'][-1])
    
    flux=(net_heat_flux*np.cos(state['latitude'].values[:])).mean()
    print(flux.item())
    Flux.append(-flux.item())

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
dycore.set_flag(True)

grid = climt.get_grid(nx=128, ny=62)

# Create model state
my_state = climt.get_default_state([dycore], grid_state=grid)

# Set initial/boundary conditions
latitudes = my_state['latitude'].values
longitudes = my_state['longitude'].values

surface_shape = latitudes.shape

# Set initial/boundary conditions
sw_flux_equator = 400
sw_flux_pole = 0

sw_flux_profile = sw_flux_equator - (
    (sw_flux_equator - sw_flux_pole)*(np.sin(np.radians(latitudes))**2))

my_state['downwelling_shortwave_flux_in_air'].values[:] = sw_flux_profile[np.newaxis, :]
my_state['surface_temperature'].values[:] = 290.
my_state['ocean_mixed_layer_thickness'].values[:] = 2
my_state['soil_layer_thickness'].values[:] = 1

my_state['eastward_wind'].values[:] = np.random.randn(
    *my_state['eastward_wind'].shape)

#setting the land
for i in range(128):
    for j in range(62):
        l = my_state['latitude'][j,i]
        if (l>20 and l<40) or (l>-40 and l<-20): 
            my_state['area_type'][j,i]=b'land'

    
start_time = time.time()
for i in range(26280*2):#26280 one year
        
    diag, my_state = dycore(my_state, model_time_step)
    my_state.update(diag)
    my_state['time'] += model_time_step

    if i%720==0:
        monitor.store(my_state)

print("--- %s seconds ---" % (time.time() - start_time))

save_state(my_state, dycore, 'spinup_2year')

with mgzip.open('flux_2year', 'wb') as f:
        pickle.dump(Flux, f)

plt.close()
plt.ioff()
plt.plot(np.array(range(len(Flux)))/3, Flux)
plt.xlabel('Time(Months)')
plt.ylabel('Flux')
plt.grid()
plt.show()