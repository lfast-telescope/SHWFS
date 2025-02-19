import numpy as np
import matplotlib.pyplot as plt
from hcipy import *

def make_lfast_aperture(dims,grid_diameter):
    if len(dims) > 1:
        if not (len(dims) == 2 and dims[0]==dims[1]):
            print('Warning: attempted to construct an irregular grid of shape ' + str(dims))
            dims = min(dims)
            print('Using grid ' + str([dims]*2))
    spider_width = 0.125*25.4e-3
    pupil_diameter = 0.762
    ID = 6*25.4e-3

    s1 = [-30.1625e-3, 69.85e-3]
    s2 = [30.1625e-3, 69.85e-3]
    s_width = 19.05e-3

    pupil_grid = make_pupil_grid(dims, grid_diameter)
    LFAST_pupil_generator = make_rotated_aperture(make_obstructed_circular_aperture(pupil_diameter,ID/pupil_diameter,4,spider_width),np.pi/4)
    square_obscuration_generator = make_spider(s1,s2,s_width)

    LFAST_pupil = LFAST_pupil_generator(pupil_grid)*square_obscuration_generator(pupil_grid)
    return LFAST_pupil

