import warnings
warnings.filterwarnings("ignore")
import climt
from sympl import (
    PlotFunctionMonitor,
    TimeDifferencingWrapper,get_constant
)
import numpy as np
from datetime import timedelta
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from climt import bolton_q_sat


def perturb(X):
    X=np.array(X)
    N=np.random.uniform(-1,1,np.shape(X[4]))*np.sqrt(2)*10**-4
    X[4][:]=(X[4]+N)[:]


def Traj(k,pert=True):
   
    climt.set_constants_from_dict({
    'stellar_irradiance': {'value': 200, 'units': 'W m^-2'}})

    model_time_step = timedelta(seconds=900)

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

    file = open('spinup_2year', 'rb')
    state_saved = pickle.load(file)
    file.close()

    my_state.update(state_saved)

    for i in range(960*100):#960 - 10 days 
        
        if (i+1)%960==0:
            n=int((i+1)/960)
            file = open('../initial/state'+str(n), 'wb')
            pickle.dump(my_state, file)
            file.close()

            Sp = dycore._gfs_cython.get_spectral_arrays()
            file = open('../initial/spectral'+str(n), 'wb')
            pickle.dump(Sp, file)
            file.close()        
        
        diag, my_state = dycore(my_state, model_time_step)
        my_state.update(diag)
        my_state['time'] += model_time_step
    
#Taj(0,False)
for i in range(1,2):
    Traj(i)