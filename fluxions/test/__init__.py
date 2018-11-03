import sys
import os

# Get path of this file and add use it to add the path of the fluxions
# directory to the python path if necessary
current_file: str = os.path.abspath(__file__)
current_directory: str = os.path.dirname(current_file)
parent_directory_1: str = os.path.dirname(current_directory)
parent_directory_2: str = os.path.dirname(parent_directory_1)


def debug_print():
    # Debug printing information used to figure all of this stuff out:
    print(f'\nfluxions/test/test_basics.py:')
    print(f'__name__ variable: {__name__}')
    print(f'__file__ variable: {__file__}')
    print(f'sys.argv[0]:       {sys.argv[0]}')
    print(f'abspath(__file__): {os.path.abspath(__file__)}')
    # Directories
    print(f'\nCurrent Directory: {current_directory}')
    print(f'Parent Directory 1: {parent_directory_1}')
    print(f'Parent Directory 2: {parent_directory_2}')


def set_path():
    if parent_directory_1 not in sys.path:
        sys.path.append(parent_directory_1)
        
    if parent_directory_2 not in sys.path:
        sys.path.append(parent_directory_2)
    
    
