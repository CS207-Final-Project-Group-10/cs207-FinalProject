"""
Michael S. Emanuel
Wed Dec  5 09:12:18 2018
"""

import os
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


from typing import Dict

# Load the data file with planetary positions from 2000/01/01 to 2040/0101
kernel = SPK.open('planets.bsp')


# *************************************************************************************************
def load_constants():
    """Load physical constants to simulate the earth-sun system"""
    
    # The universal gravitational constant
    # https://en.wikipedia.org/wiki/Gravitational_constant
    G: float = 6.67408E-11
    
    # The names of the celestial bodies
    bodies = ['sun',
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
    # https://nineplanets.org/data1.html
    masses: Dict[str, float] = \
        {'sun': 1.99E30,
         'moon' : 7.35E22,
         'mercury': 3.30E23,
         'venus': 4.87E24, 
         'earth': 5.97E24,
         'mars': 6.42E23,
         'jupiter': 1.90E27,
         'saturn': 5.68E26,
         'uranus': 8.68E25,
         'neptune': 1.02E26
         }
            
    
    # The mass of the celestial bodies
    # https://en.wikipedia.org/wiki/Earth_mass
#    mass_earth: float = 5.9722E24
#    masses_wiki: Dict[str, float]  = \
#        {'sun': mass_earth * 332946.0487,
#         'moon' : mass_earth * 0.012300037,
#         'mercury': mass_earth * 0.0553,
#         'venus': mass_earth * 0.815, 
#         'earth': mass_earth * 1.0000,
#         'mars': mass_earth * 0.107,
#         'jupiter': mass_earth * 317.8,
#         'saturn': mass_earth * 95.2,
#         'uranus': mass_earth * 14.5,
#         'neptune': mass_earth * 17.1
#         }
    
    # The radii of the celestial bodiea
    # https://nineplanets.org/data1.html
    radii: Dict[str, float] = \
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

    return G, bodies, masses, radii


# *************************************************************************************************
def positions_at(T: float):
    """Get the positions of the planets at time T."""
    pass

    # https://www.rschr.de/PRPDF/aprx_pos_planets.pdf
    # https://ssd.jpl.nasa.gov/txt/p_elem_t1.txt
    # https://gist.github.com/robbykraft/7578514

# *************************************************************************************************
G, bodies, masses, radii = load_constants()

