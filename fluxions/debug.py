import numpy as np
import os
# module is being loaded as __main__ or a module in a package.
if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir('../')
    import fluxions as fl
    os.chdir(cwd)
else:
    import fluxions as fl


