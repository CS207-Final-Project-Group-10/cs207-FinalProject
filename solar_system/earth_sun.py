"""
Michael S. Emanuel
Wed Dec  5 09:12:18 2018
"""

import os
import numpy as np
from numpy import cbrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date
from jplephem.spk import SPK

# Handle import of module fluxions differently if module
# module is being loaded as __main__ or a module in a package.
if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir('..')
    import fluxions as fl
    os.chdir(cwd)
else:
    import fluxions as fl

from typing import Tuple, Dict

# *************************************************************************************************
# Set plot style
mpl.rcParams.update({'font.size': 20})

# *************************************************************************************************
# Constants used in multiple calculations

# Conversion factor from kilometers to meters
km2m: float = 1000.0
# Conversion factor from astronomical units (AU) to meters
# https://en.wikipedia.org/wiki/Astronomical_unit
au2m: float = 149597870700

# Constant with the base date for julian day conversions
julian_base_date: date = date(1899,12,31)
# The Julian date of 1899-12-31 is 2415020; this is the "Dublin JD"
# https://en.wikipedia.org/wiki/Julian_day
julian_base_number: int = 2415020

# *************************************************************************************************
# Objects pertaining to JPL ephemerides used repeatedly
# Load the data file with planetary positions from 2000/01/01 to 2040/0101
jpl_kernel = SPK.open('planets.bsp')
# Dictionary with the integer IDs of the celestial bodies in frame 0 of planets.spk
# Verify this by issuing print(kernel)
jpl_body_id: Dict[str, int] = {
        'mercury': 1,
        'venus': 2,
        'earth': 3,
        'mars': 4,
        'jupiter': 5,
        'saturn': 6,
        'uranus': 7,
        'neptune': 8,
        'pluto': 9,
        'sun': 10
        }

# *************************************************************************************************
def load_constants():
    """Load physical constants to simulate the earth-sun system"""
    
    # The universal gravitational constant
    # https://en.wikipedia.org/wiki/Gravitational_constant
    G: float = 6.67408E-11
    
    # The names of the celestial bodies
    body_name = \
        ['sun',
         'moon',
         'mercury',
         'venus',
         'earth',
         'mars',
         'jupiter',
         'saturn',
         'uranus',
         'neptune']
    
    # The mass of the celestial bodies
    # https://en.wikipedia.org/wiki/Earth_mass
    mass_earth: float = 5.9722E24
    mass: Dict[str, float]  = \
        {'sun': mass_earth * 332946.0487,
         'moon' : mass_earth * 0.012300037,
         'mercury': mass_earth * 0.0553,
         'venus': mass_earth * 0.815, 
         'earth': mass_earth * 1.0000,
         'mars': mass_earth * 0.107,
         'jupiter': mass_earth * 317.8,
         'saturn': mass_earth * 95.2,
         'uranus': mass_earth * 14.5,
         'neptune': mass_earth * 17.1
         }
    
    # The radii of the celestial bodiea
    # https://nineplanets.org/data1.html
    radius: Dict[str, float] = \
    {'sun': 695000e3,
     'moon': 1738e3,
     'mercury': 2440e3,
     'venus': 6052e3,
     'earth': 6378e3,
     'mars': 3397e3,
     'jupiter': 71492e3,
     'saturn': 60268e3,
     'uranus': 35559e3,
     'neptune': 24766e3     
     }

    return G, body_name, mass, radius


# *************************************************************************************************
def julian_day(t: date) -> int:
    """Convert a Python datetime to a Julian day"""
    # Compute the number of days from January 1, 2000 to date t
    dt = t - julian_base_date
    # Add the julian base number to the number of days from the julian base date to date t
    return julian_base_number + dt.days


def configuration_ES(t0: date, t1: date) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the positions and velocities of the earth and sun from date t0 to t1.
    Returned as a tuple q, v
    q: Nx3 array of positions (x, y, z) in the 
    """
    # Convert t to a julian day
    jd0: int = julian_day(t0)
    jd1: int = julian_day(t1)
    jd = np.arange(jd0, jd1)

    # Position and velocity of the sun as arrays of length 3
    sun_id: int = jpl_body_id['sun']
    pos_sun, vel_sun = jpl_kernel[0, sun_id].compute_and_differentiate(jd)

    # Position and velocity of the earth as arrays of length 3
    earth_id: int = jpl_body_id['earth']
    pos_earth, vel_earth = jpl_kernel[0, earth_id].compute_and_differentiate(jd)

    # Convert these from km to meters (multiply by 1000)
    q = np.vstack([pos_sun, pos_earth]).T * km2m
    v = np.vstack([vel_sun, vel_earth]).T * km2m

    # Return tuple of Tx6 arrays for position q and velocity v
    return q, v
    

def plot_ES(q):
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
    
    # Plot the earth's orbits in blue
    fig, ax = plt.subplots(figsize=[12,12])
    ax.set_title('Orbit of Earth in 2018, Weekly')
    ax.set_xlabel('x in J2000.0 Frame; Astronomical Units (au)')
    ax.set_ylabel('y in J2000.0 Frame; Astronomical Units (au)')
    # Set marker sizes proportional to size of bodies    
    radius_earth = radius['earth']
    markersize_earth = 8.0
    markersize_sun = cbrt(radius['sun'] / radius_earth) * markersize_earth
    
    # Orbit of the Sun (it moves a little in barycentric coordinates)
    ax.plot(sun_x, sun_y, label='Sun', color='orange', linewidth=0, markersize = markersize_sun, marker='o')
    # Orbit of the Earth
    ax.plot(earth_x, earth_y, label='Earth', color='b', linewidth=0, markersize = markersize_earth, marker='o')
    
    ax.legend()
    ax.grid()
    plt.show()


# *************************************************************************************************
# Load physical constants
G, body_name, mass, radius = load_constants()

# Extract position and velocity of earth-sun system in 2018
t0 = date(2018,1,1)
t1 = date(2019,1,1)
q, v = configuration_ES(t0, t1)

# Plot the earth-sun orbits in 2018 at weekly intervals
plot_ES(q[::7])
