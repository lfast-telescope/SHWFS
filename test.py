#%%
import os
from skimage import io
import cv2 as cv
from SH_utils import *
import numpy as np
import time
import pickle
import sys
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib import patches
from aperture_utils import *
from high_level_SH_utils import *
from Zernike import *

pwd = os.getcwd()
sys.path.extend([pwd.split('SHWFS')[0] + 'primary_mirror'])
from LFAST_wavefront_utils import *
from LFAST_TEC_output import *


#%%
if __name__ == "__main__":

    if False:
        base_path = '/Documents/lfast/'
    else:
        base_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/'
    sh_path = base_path + 'on-sky/20250501/' #path pointing to folder of SH images for current night
    reference_path = os.path.join(sh_path,'220452/') #path to set of images used for pupil definition
    folder_path = os.path.join(sh_path, '221340/')

    eigenvectors_path = base_path + 'mirrors/M9/' #path to the folder containing the TEC eigenvector data from interferometer
    tec_path = eigenvectors_path
    eigenvectors = np.load(eigenvectors_path + 'eigenvectors.npy',allow_pickle=True) #load TEC eigenvector data
    eigenvalue_bounds = [] #Used for bounded optimizer (deprecated)
    eigenvalues = [0]*24 #Starting values for TEC currents

    for i in np.arange(len(eigenvalues)):
        eigenvalue_bounds.append([-0.6, 0.6])
    eigen_gain = 0.3 #Gain relating surface error to next iteration adjustement
    eigenvalues = [0]*24 #Starting values for TEC currents
 
    in_to_m = 25.4e-3
    OD = 31.9 * in_to_m  # Outer mirror diameter (m)
    ID = 6 * in_to_m  # Central obscuration diameter (m)
    clear_aperture_outer = 0.47 * OD
    clear_aperture_inner = ID / 2

    for folder in os.listdir(sh_path)[12:36]:
        print('Now processing ' + folder)
        folder_path = os.path.join(sh_path,folder) + '/'
        if os.path.isdir(folder_path):
            try:
                mean_surface = full_SHWFS_reconstruction(sh_path, folder_path, redefine_pupil=True)
            except:
                print('Error occurred when processing ' + folder)

    grid_diameter = (clear_aperture_outer*2)
    corresponding_pupil = make_lfast_aperture(mean_surface.shape,grid_diameter)
#%%
    current_eigenvalues = eigenvalues.copy()

    eigenvalues, reduced_surface = suggest_next_iteration_of_TEC_correction(current_eigenvalues, folder_path, tec_path, mean_surface, eigenvectors, clear_aperture_outer, clear_aperture_inner, Z, eigenvalue_bounds, eigen_gain)