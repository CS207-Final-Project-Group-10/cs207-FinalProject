"""
Michael S. Emanuel
Wed Dec  5 09:12:18 2018
"""

import os
from importlib import util
import numpy as np
from numpy import cbrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date

# Import from solar_system
from solar_system import km2m, au2m, day2sec, julian_day
from solar_system import jpl_kernel, jpl_body_id, load_constants, calc_mse, plot_energy

# Types
from typing import Tuple, Dict, Optional

# *************************************************************************************************
# Handle import of module fluxions differently if module
# module is being loaded as __main__ or a module in a package.
if util.find_spec("fluxions") is not None:
    import fluxions as fl
else:    
    cwd = os.getcwd()
    os.chdir('..')
    import fluxions as fl
    os.chdir(cwd)

# *************************************************************************************************
# Set plot style
mpl.rcParams.update({'font.size': 20})


# *************************************************************************************************
def configuration_ES(t0: date, t1: Optional[date] = None, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the positions and velocities of the earth and sun from date t0 to t1.
    Returned as a tuple q, v
    q: Nx3 array of positions (x, y, z) in the 
    """
    # Default dt = 1.0
    if dt is None:
        dt = 1.0

    # Default t1 to one day after t0
    if t1 is not None:
        # Convert t to a julian day
        jd0: int = julian_day(t0)
        jd1: int = julian_day(t1)
    else:
        jd0: int = julian_day(t0)
        jd1: int = jd0 + dt
    # Pass the times as an array of julian days
    jd: np.ndarray = np.arange(jd0, jd1, dt)

    # Position and velocity of the sun as arrays of length 3
    sun_id: int = jpl_body_id['sun']
    pos_sun, vel_sun = jpl_kernel[0, sun_id].compute_and_differentiate(jd)

    # Position and velocity of the earth as arrays of length 3
    earth_id: int = jpl_body_id['earth']
    pos_earth, vel_earth = jpl_kernel[0, earth_id].compute_and_differentiate(jd)

    # Convert positions from km to meters (multiply by km2m)
    q = np.vstack([pos_sun, pos_earth]).T * km2m
    # Convert velocities from km / day to meters / sec (multiply by km2m, divide by day2sec)
    v = np.vstack([vel_sun, vel_earth]).T * (km2m / day2sec)

    # Return tuple of Tx6 arrays for position q and velocity v
    return q, v
    

# *************************************************************************************************
def plot_ES(q: np.ndarray, sim_name: str, fname: Optional[str] = None):
    """
    Plot the earth-sun orbits.
    q is a Tx6 array.  T indexes time points.  6 columns are sun (x, y, z) and earth (x, y, z)
    """
    # Convert all distances from meters to astronomical units (AU)
    # Unpack sun (x, y) in au; z not used in plots
    sun_x = q[:,0] / au2m
    sun_y = q[:,1] / au2m
    # Unpack earth (x, y); z not used in plots
    earth_x = q[:,3] / au2m
    earth_y = q[:,4] / au2m
    
    # Set up chart title and scale
    fig, ax = plt.subplots(figsize=[12,12])
    ax.set_title(f'Orbit of Earth in 2018; Weekly from {sim_name}')
    ax.set_xlabel('x in J2000.0 Frame; Astronomical Units (au)')
    ax.set_ylabel('y in J2000.0 Frame; Astronomical Units (au)')
    a = 1.2
    ticks = np.arange(-a, a+0.2, 0.2)
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Set marker sizes proportional to size of bodies    
    radius_earth = radius['earth']
    markersize_earth = 8.0
    markersize_sun = cbrt(radius['sun'] / radius_earth) * markersize_earth
    
    # Orbit of the Sun (it moves a little in barycentric coordinates)
    ax.plot(sun_x, sun_y, label='Sun', color='orange', linewidth=0, markersize = markersize_sun, marker='o')
    # Orbit of the Earth
    ax.plot(earth_x, earth_y, label='Earth', color='b', linewidth=0, markersize = markersize_earth, marker='o')
    
    # Legend and grid
    ax.legend()
    ax.grid()

    # Save plot if a filename was provided
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')

    # Display plot
    plt.show()


# *************************************************************************************************
def accel_ES(q: np.ndarray):
    """
    Compute the gravitational accelerations in the earth-sun system.
    q in row vector of 6 elements: sun (x, y, z), earth (x, y, z)
    """

    # Number of celestial bodies
    num_bodies: int = 2
    # Number of dimensions in arrays; 3 spatial dimensions times the number of bodies
    dims = 3 * num_bodies

    # Body 0 is the sun
    mass_0 = mass_sun
    # Body 1 is the earth
    mass_1 = mass_earth

    # Extract position of the sun and earth as 3-vectors
    pos_0 = q[slice_0]
    pos_1 = q[slice_1]

    # Displacement vector from sun to earth
    dv_01: np.ndarray = pos_1 - pos_0

    # Distance from sun to earth
    r_01: float = np.linalg.norm(dv_01)
    
    # Unit vector pointing from sun to earth
    udv_01 = dv_01 / r_01

    # The force between these has magnitude G*m1*m2 / r^2
    f_01: float = (G * mass_0 * mass_1) / (r_01 ** 2)
    
    # Initialize acceleration as 6x1 array
    a: np.ndarray = np.zeros(dims)
    
    # The force vectors are attractive
    a[slice_0] += f_01 * udv_01 / mass_0
    a[slice_1] -= f_01 * udv_01 / mass_1

    # Return the acceleration vector
    return a


def simulate_leapfrog_ES(t0: date, t1: date, dt: float):
    """
    Simulate the earth-sun system from t0 to t1 using Leapfrog Integration.
    dt is a time step in days.
    """
    
    # Length of the simulation in days
    N: int = round((t1 - t0).days / dt)
    # Number of celestial bodies
    num_bodies: int = 2
    # Number of dimensions in arrays; 3 spatial dimensions times the number of bodies
    dims = 3 * num_bodies
    
    # Convert dt from days to seconds (want all calculations in MKS)
    dt *= day2sec
    # Square of the time step
    dt2: float = dt * dt
    # Number of time steps in the sumulation
    
    # Initialize arrays to store computed positions and velocities
    q: np.ndarray = np.zeros((N, dims))
    v: np.ndarray = np.zeros((N, dims))
    
    # Initialize the first row with the initial conditions from the JPL ephemerides
    q0, v0 = configuration_ES(t0)
    q[0, :] = q0
    v[0, :] = v0
    
    # Initialize an array to store the acceleration at each time step
    a: np.ndarray = np.zeros((N, dims))
    # First row of accelerations
    a[0, :] = accel_ES(q[0])
    
    # Perform leapfrog integration simulation
    # https://en.wikipedia.org/wiki/Leapfrog_integration
    for i in range(N-1):
        # Positions at the next time step
        q[i+1,:] = q[i,:] + v[i,:] * dt + 0.5 * a[i,:] * dt2
        # Accelerations of each body in the system at the next time step
        a[i+1,:] = accel_ES(q[i+1])        
        # Velocities of each body at the next time step
        v[i+1,:] = v[i,:] + 0.5 * (a[i,:] + a[i+1,:]) * dt
    return q, v


# *************************************************************************************************
def energy_ES(q, v):
    """Compute the kinetic and potential energy of the earth sun system"""
    # Body 0 is the sun
    mass_0 = mass_sun
    # Body 1 is the earth
    mass_1 = mass_earth

    # Positions of sun and earth
    q0: np.ndarray = q[:, slice_0]
    q1: np.ndarray = q[:, slice_1]

    # Velocities of sun and earth
    v0: np.ndarray = v[:, slice_0]
    v1: np.ndarray = v[:, slice_1]

    # Kinetic energy is 1/2 mv^2
    T0: np.ndarray = 0.5 * mass_0 * np.sum(v0 * v0, axis=1)
    T1: np.ndarray = 0.5 * mass_1 * np.sum(v1 * v1, axis=1)
    T: np.ndarray = T0 + T1
    
    # Potential energy is -G m1 m2  / r
    dv_01 = q1 - q0
    r_01 = np.linalg.norm(dv_01, axis=1)
    U_01: np.ndarray = -G * mass_0 * mass_1 * 1.0 / r_01
    U: np.ndarray = U_01
    
    # Total energy H = T + U
    H = T + U
    
    return H, T, U


# *************************************************************************************************
# main
# Load physical constants
G, body_name, mass, radius = load_constants()

# Masses of sun and earth
mass_sun = mass['sun']
mass_earth = mass['earth']

# Slices for celestial bodies
slice_0 = slice(0, 3)
slice_1 = slice(3, 6)

# Set simulation time step to one day
dt: float = 2.0**(-4)

# Extract position and velocity of earth-sun system in 2018
t0 = date(2018,1,1)
t1 = date(2019,1,1)
q_jpl, v_jpl = configuration_ES(t0, t1, dt)

# Simulate solar earth-sun system 
q_sim, v_sim = simulate_leapfrog_ES(t0, t1, dt)

# Compute energy time series for earth-sun system with JPL and leapfrog simulations
H_jpl, T_jpl, U_jpl = energy_ES(q_jpl, v_jpl)
H_sim, T_sim, U_sim = energy_ES(q_sim, v_sim)

# Plot the earth-sun orbits in 2018 at weekly intervals using the simulation
step = round(7.0 / dt)
plot_ES(q_jpl[::step], 'JPL', 'figs/earth_sun_jpl.png')
plot_ES(q_sim[::step], 'Leapfrog', 'figs/earth_sun_leapfrog.png')

# Compute the MSE in AUs between the two simulations
mse = calc_mse(q_jpl, q_sim)
print(f'MSE between lapfrog simulation with dt={dt:0.2f} days and JPL: {mse:0.3e} astronomical units.')

# Compute energy change as % of original KE
energy_chng_jpl = (H_jpl[-1] - H_jpl[0]) / T_jpl[0]
energy_chng_sim = (H_sim[-1] - H_sim[0]) / T_sim[0]
print(f'Energy change as fraction of original KE during simulation with dt={dt:0.2f}:')
print(f'JPL:      {energy_chng_jpl*100:0.2e}.')
print(f'Leapfrog: {energy_chng_sim*100:0.2e}.')

# Plot time series of kinetic and potential energy
time = np.arange(0.0, (t1-t0).days, dt)
plot_energy(time, H_jpl, T_jpl, U_jpl)
