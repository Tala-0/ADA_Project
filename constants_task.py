import numpy as np

##################################################################################################################################################
# simulation parameters: dynamics and DA
n = 250                                                     # number of gridpoints
dte = 10                                                    # time step between assimilations of observations in DA, corresponds to 5 model minutes
dt_burnin = 10                                              # time interval to discard for computing statistics
nens = [20,50 ,80]                                                # original number of ensemble members (number used for DA)
loc_radius = [2,5,8]                                        # localisation radius for DA
infl_factor = [1., 1.02, 1.05]
ncyc = 50                                                   # number of DA cycles

# observation error
variances = np.array([0.1, 1.0, 0.001])
sig = np.sqrt(variances/0.4)                               # For DA. Noise added to truth to create observations

# observation operator (or forward operator)
obs_var_name = "h"                                         # which variable to observe? Choose from "u", "h" or "r"
nth_point = 5                                              # observe the state every nth_point grid point

##################################################################################################################################################
# Initial conditions distribution

# Initial condition mean: h = 90 - harray (orography is installed in the code), u = 2m/s, r = 0
mu_ic = np.zeros(3*n)
mu_ic[0:n] = 2                          # 2m/s
mu_ic[n:2*n]   = 90.0                   # fluid depth above orography
mu_ic[2*n:3*n] = 0.0                    # rain starts at zero

# Initial condition mean variances
C_diag = np.ones(3*n)
C_diag[0:n]    = variances[0]    # u: ±0.3 m/s
C_diag[n:2*n]  = variances[1]    # h: ±1 m around 90 m mean
C_diag[2*n:3*n] = variances[2]   # r: tiny rain perturbations

##################################################################################################################################################
# other parameters
dx = 500.0                                                  # horizontal resolution of 500
dts = 1.0                                                   # timestep used in discretisation
g = 9.81                                                    # gravitational constant
h_cloud = 15                                                # height threshold for cloud formation
phic = 147.15                                               # geopotential constant value above first threshold to allow for unstable convection.
h_0 = 10.0                                                  # resting height of fluid
gamma = np.sqrt(g*h_0)                                               # weight for negative buoyancy of rain (c in paper) 
h_rain = 20                                                # height threshold for rain
beta = 0.1                                                  # lag between cloud and rain formation.
alpha = 0.01                                                # half-life of influence of rain of roughly 1 hour 
kr = 10.0                                                   # diffusion constant for rain.
hw = 5                                                      # half width of mountain ridge
amp = 1.2                                                   # amplitude of mountain ridge
mu = n/2                                                    # centre point of mountain ridge

n_array = np.array([[4,0.000008,0,n]])
#n_array = np.array([[4,0.005,0,n-1],[6,0.00,60,90]])       # info of random noise if nindex = 1. Each row indicates a location of noise in the order 
                                                            # [sigma (half width) of the noise field,amplitude, start, end of location where the
                                                            # model can randomnly selcet the noise position. Currently perturbations in whole domain (first row).
                                                            # Standard amplitude = 0.00895. change to 0.000008 if mindex = 1.
