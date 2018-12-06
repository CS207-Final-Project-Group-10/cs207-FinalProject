"""
Michael S. Emanuel
Thu Dec  6 15:05:39 2018
"""

import os
from importlib import util
import numpy as np
from numpy import cbrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date
from jplephem.spk import SPK
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
# Number of seconds in one day
day2sec: float = 24.0 * 3600.0

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


def calc_mse(q1, q2):
    """Compare the results of two simulations"""
    # Difference in positions between two simulations
    dq = q2 - q1
    # Mean squared error in AUs
    return np.linalg.norm(dq) / au2m


# *************************************************************************************************
def plot_energy(time, H, T, U):
    """Plot kinetic and potential energy of system over time"""
    # Normalize energy to initial KE
    T0 = T[0]
    H = H / T0
    T = T / T0
    U = U / T0
    
    # Plot
    fig, ax = plt.subplots(figsize=[16,8])
    ax.set_title('System Energy vs. Time')
    ax.set_xlabel('Time in Days')
    ax.set_ylabel('Energy (% Initial KE)')
    ax.plot(time, T, label='T', color='r')
    ax.plot(time, U, label='U', color='b')
    ax.plot(time, H, label='H', color='k')
    ax.legend()
    ax.grid()
    plt.show()
