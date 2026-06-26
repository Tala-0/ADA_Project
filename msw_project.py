import sys
sys.path.append("path to the folder where you stored constants.py from the MSW github repository")

import numpy as np
import dapper as dp
import dapper.da_methods as da
from constants import *
import matplotlib.pyplot as plt

# NOTE: Dynamics without orography here...
# =============================================================================
# 1. Dapper step function
# =============================================================================
def dxdt(x):
    """
    x: shape (3*n,) or (N_ens, 3*n)
    Returns tendency, same shape as input.
    """
    ku = 2000    # diffusion constant for u
    kh = 6000    # diffusion coefficient for h
    
    # Normalize to 2D: (N_ens, 3*n)
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]   # (1, 3*n)
    N_ens = x.shape[0]

    # Unpack into padded arrays with ghost cells
    u = np.zeros((N_ens, n + 2))
    h = np.zeros((N_ens, n + 2))
    r = np.zeros((N_ens, n + 2))

    u[:, 1:n+1] = x[:, 0:n]
    h[:, 1:n+1] = x[:, n:2*n]
    r[:, 1:n+1] = x[:, 2*n:3*n]

    # Periodic BCs
    u[:, 0], u[:, n+1] = u[:, n], u[:, 1]
    h[:, 0], h[:, n+1] = h[:, n], h[:, 1]
    r[:, 0], r[:, n+1] = r[:, n], r[:, 1]

    # Geopotential
    phi = np.zeros((N_ens, n + 2))
    phi[:, 1:n+1] = np.where(h[:, 1:n+1] > h_cloud, phic, g * h[:, 1:n+1])
    phi[:, 0]   = phi[:, n]
    phi[:, n+1] = phi[:, 1]
    phi += gamma * r

    # Momentum tendency
    du = (
        - (1/(2*dx)) * (u[:, 2:n+2]**2 - u[:, 0:n]**2)
        - (2/dx)     * (phi[:, 1:n+1] - phi[:, 0:n])
        + (ku/dx**2) * (u[:, 2:n+2] - 2*u[:, 1:n+1] + u[:, 0:n])
    )

    # Continuity tendency
    dh = (
        - (1/dx) * (u[:, 2:n+2]*(h[:, 1:n+1]+h[:, 2:n+2]) - u[:, 1:n+1]*(h[:, 0:n]+h[:, 1:n+1]))
        + (kh/dx**2) * (h[:, 2:n+2] - 2*h[:, 1:n+1] + h[:, 0:n])
    )

    # Rain source
    mask = np.logical_and(h[:, 1:n+1] > h_rain, u[:, 2:n+2] - u[:, 1:n+1] < 0)
    beta_loc = np.where(mask, beta, 0.0)

    dr = (
        - (1/(2*dx)) * (u[:, 2:n+2] + u[:, 1:n+1]) * (r[:, 2:n+2] - r[:, 0:n])
        - alpha * r[:, 1:n+1]
        - beta_loc * (1/dx) * (u[:, 2:n+2] - u[:, 1:n+1])
        + (kr/dx**2) * (r[:, 2:n+2] - 2*r[:, 1:n+1] + r[:, 0:n])
    )

    out = np.concatenate([du, dh, dr], axis=1)  # (N_ens, 3*n)

    return out[0] if single else out

def swm_step(x, t, dt):
    """
    DAPPER-compatible step function. Rain clipped to zero if negative.
    x: shape (3*n,) for single realization, or (N_ens, 3*n) for ensemble
    """
    x_new = dp.mods.integration.rk4(lambda x, t: dxdt(x), x, t, dt)
    # clip rain (indices 2n:3n) to non-negative
    if x_new.ndim == 1:
        x_new[2*n:] = np.maximum(x_new[2*n:], 0.0)
    else:
        x_new[:, 2*n:] = np.maximum(x_new[:, 2*n:], 0.0)
    return x_new

# =============================================================================
# 2. Localization for LETKF
# =============================================================================

def make_localizer_1D_periodic(state_coord, obs_coord, n):
    """
    state_coord: 1D array, shape (3*n,), grid index for each state dim
    obs_coord:   1D array, shape (Ny,),  grid index for each obs
    n: domain size for periodic wrapping
    """
    def gaspari_cohn(r):
        r = np.abs(r)
        gc = np.zeros_like(r, dtype=float)
        m1 = r < 0.5
        gc[m1] = 1 - 6*r[m1]**2 + 6*r[m1]**3
        m2 = (r >= 0.5) & (r < 1.0)
        gc[m2] = 2*(1 - r[m2])**3
        return gc

    def localizer(loc_rad, direction, taper="GC"):
        if direction == "x2y":
            state_batches = [np.array([i]) for i in range(len(state_coord))]

            def obs_taperer(ii):
                s_pos = state_coord[ii[0]]           # fixed: 1D indexing
                raw_dist = np.abs(obs_coord - s_pos)
                dist = np.minimum(raw_dist, n - raw_dist)
                within = dist <= loc_rad
                oBatch = np.where(within)[0]
                tapering = gaspari_cohn(dist[within] / loc_rad)
                return oBatch, tapering

            return state_batches, obs_taperer

        elif direction == "y2x":
            def state_taperer(j):
                o_pos = obs_coord[j]                 # fixed: 1D indexing
                raw_dist = np.abs(state_coord - o_pos)
                dist = np.minimum(raw_dist, n - raw_dist)
                within = dist <= loc_rad
                ii = np.where(within)[0]
                tapering = gaspari_cohn(dist[within] / loc_rad)
                return ii, tapering

            return state_taperer

    return localizer

# =============================================================================
# 2. HMM
# =============================================================================

# Setup basic grid parameters matching SWM constants
Nx = n*3  # State size (grid points * 3, that is the number of fields u, v, h)

# Dynamic Model object in DAPPER
Dyn = {
    'M': Nx,
    'model': swm_step,
    'noise': 0,  # Set to a positive float/matrix if you want stochastic model noise
}

# Initial conditions distribution
C_diag = np.ones(Nx)
C_diag[0:n]    = 0.1    # u: ±0.3 m/s
C_diag[n:2*n]  = 1.0    # h: ±1 m around 90 m mean
C_diag[2*n:3*n] = 0.001 # r: tiny rain perturbations
X0 = dp.tools.randvars.GaussRV(mu=mu, C=np.diag(C_diag))

# Setup Observation Operators: observing h every 4th point
obs_indices = np.arange(n, 2*n, 4)
Ny = len(obs_indices)

def hmod(x):
    # Operator mapping state to observation space
    return x[..., obs_indices]

# Setup
grid = np.arange(n)
state_coord = np.concatenate([grid, grid, grid])  # shape (3*n,), 1D
obs_grid    = np.arange(0, n, 4)
obs_coord   = obs_grid.astype(float)              # shape (Ny,),  1D

localizer = make_localizer_1D_periodic(state_coord, obs_coord, n)

Obs = dp.mods.Operator(
    M= Ny,
    model = hmod,
    noise =  dp.tools.randvars.GaussRV(C= np.diag(np.ones(Ny))), # Observational error covariance, same as C_diag for h
    localizer = localizer
)

# Time sequences (dt must align with SWM's stable time stepping constraints)
tseq = dp.mods.Chronology(dt=1.0, dto=5.0, T=100.0, BurnIn=10.0)

# Put everything together into an HMM object
HMM = dp.mods.HiddenMarkovModel(Dyn, Obs, tseq, X0)

# =============================================================================
# 3. RUN THE DA EXPERIMENT
# =============================================================================

# suggestions for LETKF
# N=50 / N=70
# loc_rad = 10 and infl = 1

if __name__ == "__main__":
    print("Generating synthetic truth and observations...")
    # Generate true trajectory and noisy simulated observations
    xx, yy = HMM.simulate()
    
    print("Setting up DA Method (EnKF)...")
    # Configure an Ensemble Kalman Filter using DAPPER
    xp = da.EnKF("Sqrt", N=150, infl=1.05, rot=True) 

    xp.assimilate(HMM, xx, yy)
