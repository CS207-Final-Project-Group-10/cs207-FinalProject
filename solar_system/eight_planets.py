"""
Michael S. Emanuel
Thu Dec  6 21:26:00 2018
"""

import os
from subprocess import call
from importlib import util
import numpy as np
from numpy import cbrt
import matplotlib as mpl
# Allows matplotlib to run without being install as a framework on OSX
mpl.use('TkAGG')
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm

# Import from solar_system
from solar_system import km2m, au2m, day2sec, julian_day, load_constants
from solar_system import jpl_kernel, jpl_body_id, simulate_leapfrog, calc_mse, plot_energy, U_ij

# Types
from typing import Tuple, List, Dict, Optional

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
def configuration(t0: date, t1: Optional[date] = None,  
                  steps_per_day: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the positions and velocities of the sun and eight planets
    Returned as a tuple q, v
    q: Nx3 array of positions (x, y, z) in the J2000.0 coordinate frame.
    """
    # Default steps_per_day = 1
    if steps_per_day is None:
        steps_per_day = 1
        
    # Time step dt is 1.0 over steps per day
    dt: float = 1.0 / float(steps_per_day)

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

    # Number of time steps
    N: int = len(jd)
    
    # bodies is a list of the celestial bodies considered; should be in an enclosing scope
    # Number of bodies
    B: int = len(bodies)
    # Number of dimensions
    dims: int = B * 3
    
    # Initialize empty arrays for position q and velocity v
    q: np.ndarray = np.zeros((N, dims))
    v: np.ndarray = np.zeros((N, dims))

    # Position and velocity of the sun as arrays of length 3
    body_ids: List[int] = [jpl_body_id[body] for body in bodies]

    # Fill in the position and velocity for each body in order
    for i, body_id in enumerate(body_ids):
        # The slice of columns for this body (same in q and v)
        slice_i = slice(3*i, 3*(i+1))
        # Extract the position and velocity from jpl
        qi, vi = jpl_kernel[0, body_id].compute_and_differentiate(jd)
        # Convert positions from km to meters (multiply by km2m)
        q[:, slice_i] = qi.T * km2m
        # Convert velocities from km / day to meters / sec (multiply by km2m, divide by day2sec)
        v[:, slice_i] = vi.T * (km2m / day2sec)

    # Return tuple of Tx6 arrays for position q and velocity v
    return q, v
    

# *************************************************************************************************
def plot(q: np.ndarray, bodies: List[str], plot_colors: Dict[str, str],
         sim_name: str, fname: Optional[str] = None):
    """
    Plot the planetary orbits.  Plot size limited to box of 10 AU around the sun.
    q is a Nx3B array.  t indexes time points.  3B columns are (x, y, z) for the bodies in order.
    """
    # Get N and number of dims
    N, dims = q.shape

    # Slices for 
    x_slice = slice(0, dims, 3)
    y_slice = slice(1, dims, 3)
    # Convert all distances from meters to astronomical units (AU)
    plot_x = q[:, x_slice] / au2m
    plot_y = q[:, y_slice] / au2m
    
    # Set up chart title and scale
    fig, ax = plt.subplots(figsize=[12,12])
    ax.set_title(f'Inner Planetary Orbits in 2018; Weekly from {sim_name}')
    ax.set_xlabel('x in J2000.0 Frame; Astronomical Units (au)')
    ax.set_ylabel('y in J2000.0 Frame; Astronomical Units (au)')
    # Scale and tick size
    a = 5.0
    da = 1.0
    ticks = np.arange(-a, a+da, da)
    # Set limits and ticks
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Set marker sizes proportional to size of bodies
    radius_earth = radius_tbl['earth']
    markersize_earth = 4.0
    markersize_tbl = {body : cbrt(radius_tbl[body] / radius_earth) * markersize_earth for body in bodies}
    
    # Plot the orbit of each body
    for k, body in enumerate(bodies):
        ax.plot(plot_x[:, k], plot_y[:, k], label=body, color=plot_colors[body], 
                linewidth=0, markersize = markersize_tbl[body], marker='o')
    
    # Legend and grid
    fig.legend(loc=7, bbox_to_anchor=(0.85, 0.5))
    # ax.legend()
    ax.grid()

    # Save plot if a filename was provided
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')

    # Display plot
    plt.show()


# *************************************************************************************************
def make_frame(fig, ax, plot_x: np.ndarray, plot_y: np.ndarray, frame_num: int, 
               bodies: List[str], plot_colors: Dict[str, str], markersize_tbl: Dict[str, float],
               fname: str):
    """
    Make a series of frames of the planetary orbits that can be assembled into a movie.
    q is a Nx3B array.  t indexes time points.  3B columns are (x, y, z) for the bodies in order.
    """
    # Clear the axis
    ax.clear()

    ax.set_title(f'Inner Planetary Orbits in 2018')
    ax.set_xlabel('x in J2000.0 Frame; Astronomical Units (au)')
    ax.set_ylabel('y in J2000.0 Frame; Astronomical Units (au)')

    # Scale and tick size
    a = 2.0
    da = 1.0
    ticks = np.arange(-a, a+da, da)

    # Set limits and ticks
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Plot the orbit of each body
    for k, body in enumerate(bodies[0:5]):
        ax.plot(plot_x[k], plot_y[k], label=body, color=plot_colors[body], 
                linewidth=0, markersize = markersize_tbl[body], marker='o')    
    ax.grid()

    # Save this frame
    fig.savefig(f'{fname}_{frame_num:05d}.png')


def make_movie(q: np.ndarray, step: int, bodies: List[str], plot_colors: Dict[str, str], fname: str):
    """
    Make a series of frames of the planetary orbits that can be assembled into a movie.
    q is a Nx3B array.  t indexes time points.  3B columns are (x, y, z) for the bodies in order.
    """
    # Get N and number of dims
    N, dims = q.shape

    # Slices for x and y
    x_slice = slice(0, dims, 3)
    y_slice = slice(1, dims, 3)
    # Convert all distances from meters to astronomical units (AU)
    plot_x = q[:, x_slice] / au2m
    plot_y = q[:, y_slice] / au2m
    
    # Set marker sizes proportional to size of bodies
    radius_earth = radius_tbl['earth']
    markersize_earth = 8.0
    markersize_tbl = {body : (radius_tbl[body] / radius_earth)**0.25 * markersize_earth for body in bodies}
    
    # Set up chart title and scale
    fig, ax = plt.subplots(figsize=[12,12])

    # Make each frame
    print(f'Generating movie with {N} frames...')
    for fn in tqdm(range(N//step)):
        # The row number in the array is the frame number times the step size
        rn: int = fn * step
        make_frame(fig, ax, plot_x[rn], plot_y[rn], fn, bodies, plot_colors, markersize_tbl, fname)

    # Assemble the frames into a movie (video only)
    cmd1: List[str] = ['ffmpeg', '-r', '24', '-f', 'image2', '-s', '864x864', '-i', 'movie/planets_%05d.png',
                     '-vcodec',  'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', 'movie/planets_video.mp4']
    call(cmd1)

    # Add the soundtrack to the movie
    cmd2: List[str] = ['ffmpeg', '-i', 'movie/planets_video.mp4', '-i', 'movie/holst_jupiter.mp3', 
                      '-c:v', 'copy', '-shortest', 'movie/planets.mp4']
    call(cmd2) 

    # Delete the frames
    cmd3: List[str] = ['rm', 'movie/*.png']
    call(cmd3)

# *************************************************************************************************
def accel(q: np.ndarray):
    """
    Compute the gravitational accelerations in the system
    q in row vector of 6 elements: sun (x, y, z), earth (x, y, z)
    """

    # Infer number of dimensions from q
    dims: int = len(q)

    # Initialize acceleration as dimsx1 array
    a: np.ndarray = np.zeros(dims)
    
    # Iterate over each distinct pair of bodies
    for i in range(B):
        for j in range(i+1, B):
    
            # Masses of body i and j
            m0 = mass[i]
            m1 = mass[j]
        
            # Extract position of body i and j as 3-vectors
            pos_0 = q[slices[i]]
            pos_1 = q[slices[j]]
        
            # Displacement vector from body i to body j
            dv_01: np.ndarray = pos_1 - pos_0
        
            # Distance from body i to j
            r_01: float = np.linalg.norm(dv_01)
            
            # Unit vector pointing from body i to body j
            udv_01 = dv_01 / r_01
        
            # The force between these has magnitude G*m0*m1 / r^2
            f_01: float = (G * m0 * m1) / (r_01 ** 2)
            
            # The force vectors are attractive
            a[slices[i]] += f_01 * udv_01 / m0
            a[slices[j]] -= f_01 * udv_01 / m1
        
    # Return the acceleration vector
    return a


# *************************************************************************************************
def energy(q, v):
    """Compute the kinetic and potential energy of the planetary system"""
    
    # Number of points
    N: int = len(q)
    
    # Initialize arrays to zero of the correct size
    T: np.ndarray = np.zeros(N)
    U: np.ndarray = np.zeros(N)    
    
    # Add up kinetic energy of each body
    for i in range(B):
        # Kinetic energy is 1/2 mv^2
        m = mass[i]
        vi = v[:, slices[i]]
        T += 0.5 * m * np.sum(vi * vi, axis=1)
    
    # Add up potential energy of each pair of bodies
    for i in range(B):
        for j in range(i+1, B):
            # Masses of these two bodies
            mi = mass[i]
            mj = mass[j]
        
            # Positions of body i and j
            qi: np.ndarray = q[:, slices[i]]
            qj: np.ndarray = q[:, slices[j]]
        
            # Potential energy is -G m1 m2  / r
            dv_ij = qj - qi
            r_ij = np.linalg.norm(dv_ij, axis=1)
            U -= G * mi * mj * 1.0 / r_ij
    
    # Total energy H = T + U
    H = T + U
    
    return H, T, U


# *************************************************************************************************
# q variables for the eight planets system
x0, y0, z0 = fl.Vars('x0', 'y0', 'z0')
x1, y1, z1 = fl.Vars('x1', 'y1', 'z1')
x2, y2, z2 = fl.Vars('x2', 'y2', 'z2')
x3, y3, z3 = fl.Vars('x3', 'y3', 'z3')
x4, y4, z4 = fl.Vars('x4', 'y4', 'z4')
x5, y5, z5 = fl.Vars('x5', 'y5', 'z5')
x6, y6, z6 = fl.Vars('x6', 'y6', 'z6')
x7, y7, z7 = fl.Vars('x7', 'y7', 'z7')
x8, y8, z8 = fl.Vars('x8', 'y8', 'z8')

# Arrange qs into a list
q_vars = [x0, y0, z0,
          x1, y1, z1,
          x2, y2, z2,
          x3, y3, z3,
          x4, y4, z4,
          x5, y5, z5,
          x6, y6, z6,
          x7, y7, z7,
          x8, y8, z8]


def make_force(q_vars, mass):
    """Fluxion with the potential energy of the eight planets sytem"""
    # Number of bodies
    B: int = len(mass)
    
    # Build the potential energy fluxion by iterating over distinct pairs of bodies
    U = fl.Const(0.0)
    for i in range(B):
        for j in range(i+1, B):
            U +=  U_ij(q_vars, mass, i, j)
    
    # Varname arrays for both the coordinate system and U
    vn_q = np.array([q.var_name for q in q_vars])
    vn_fl = np.array(sorted(U.var_names))
    # Permutation array for putting variables in q in the order expected by U (alphabetical)
    q2fl = np.array([np.argmax((vn_q == v)) for v in vn_fl])
    # Permutation array for putting results of U.diff() in order of q_vars
    fl2q = np.array([np.argmax((vn_fl == v)) for v in vn_q])
    # Return a force function from this potential
    force_func = lambda q: -U.diff(q[q2fl]).squeeze()[fl2q]
    return force_func


def accel_fl(q: np.ndarray):
    """Accelaration in the earth-sun system using Fluxion potential energy"""
    # Infer number of dimensions from q
    dims: int = len(q)
    # Number of celestial bodies
    B: int = dims // 3

    # The force given the positions q of the bodies
    f = force(q)

    # The accelerations from this force
    a = np.zeros(dims)
    for i in range(B):
        a[slices[i]] = f[slices[i]] / mass[i]
    
    return a


# *************************************************************************************************
# main

# Load physical constants
G, body_name, mass_tbl, radius_tbl = load_constants()

# The celestial bodies in this simulation
bodies = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

# Number of bodies in this simulation
B: int = len(bodies)
# Number of dimensions
dims: int = B*3

# Masses of sun and earth
mass = np.array([mass_tbl[body] for body in bodies])

# Slices for the B celestial bodies
slices = [slice(b*3, (b+1)*3) for b in range(B)]

# Colors for plotting each body
plot_colors = {
        'sun': 'orange',
        'mercury': 'gray',
        'venus': 'yellow',
        'earth': 'blue',
        'mars': 'red',
        'jupiter':'orange',
        'saturn':'gold',
        'uranus':'blue',
        'neptune':'blue'
        }

# Build a force function
force = make_force(q_vars, mass)

def main():
    # Set simulation time step to one day
    steps_per_day: int = 16
    
    # Extract position and velocity of earth-sun system in 2018
    t0 = date(2018,1,1)
    t1 = date(2019,1,1)
    q_jpl, v_jpl = configuration(t0, t1, steps_per_day)
    
    # Simulate solar earth-sun system 
    q_sim, v_sim = simulate_leapfrog(configuration, accel, t0, t1, steps_per_day)
    
    # Compute energy time series for earth-sun system with JPL and leapfrog simulations
    H_jpl, T_jpl, U_jpl = energy(q_jpl, v_jpl)
    H_sim, T_sim, U_sim = energy(q_sim, v_sim)
    
    # Plot the earth-sun orbits in 2018 at weekly intervals using the simulation
    plot_step: int = 7 * steps_per_day
    plot(q_jpl[::plot_step], bodies, plot_colors, 'JPL', 'figs/eight_planets_jpl.png')
    plot(q_sim[::plot_step], bodies, plot_colors, 'Leapfrog', 'figs/eight_planets_leapfrog.png')
    
    # Compute the MSE in AUs between the two simulations
    mse = calc_mse(q_jpl, q_sim)
    print(f'MSE between leapfrog simulation with {steps_per_day} steps per day and JPL:')
    print(f'{mse:0.3e} astronomical units.')
    
    # Compute energy change as % of original KE
    energy_chng_jpl = (H_jpl[-1] - H_jpl[0]) / T_jpl[0]
    energy_chng_sim = (H_sim[-1] - H_sim[0]) / T_sim[0]
    print(f'\nEnergy change as fraction of original KE during simulation with {steps_per_day} steps per day:')
    print(f'JPL:      {energy_chng_jpl:0.2e}.')
    print(f'Leapfrog: {energy_chng_sim:0.2e}.')
    
    # Plot time series of kinetic and potential energy
    # N: int = len(q_jpl)
    # plot_days = np.linspace(0.0, (t1-t0).days, N)
    # plot_energy(plot_days, H_jpl, T_jpl, U_jpl)
    
    # Make movie frames if necessary    
    if not os.path.isfile('movie/planets_video.mp4'):
        make_movie(q_jpl, steps_per_day//2, bodies, plot_colors, 'movie/planets')
    else:
        print(f'Found movie/planets_video.mp4, not regenerating it.')
        
    # Move post-processing 

if __name__ == '__main__':
    main()

