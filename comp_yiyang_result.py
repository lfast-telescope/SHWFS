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
from plotting_utils import *

in_to_m = 25.4e-3
OD = 31.9 * in_to_m  # Outer mirror diameter (m)
ID = 6 * in_to_m  # Central obscuration diameter (m)
clear_aperture_outer = 0.47 * OD
clear_aperture_inner = ID / 2

base_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/'
sh_path = base_path + 'on-sky/20250501/'  # path pointing to folder of SH images for current night
eigenvectors_path = base_path + 'mirrors/M9/'  # path to the folder containing the TEC eigenvector data from interferometer
eigenvectors = np.load(eigenvectors_path + 'eigenvectors.npy', allow_pickle=True)  # load TEC eigenvector data

yiyang_result = np.load(sh_path + 'yiyang_phase_estimate.npy')
sh_result = np.load(sh_path + 'wf_214247.npy')
yiyang_result[yiyang_result == 0] = np.nan

interpolated_sh = griddata_interpolater(sh_result, eigenvectors[0], clear_aperture_outer, clear_aperture_inner)
interpolated_yy = griddata_interpolater(yiyang_result, eigenvectors[0], clear_aperture_outer, clear_aperture_inner)

Z = General_zernike_matrix(44,int(15*25.4 * 1e3),int(3*25.4*1e3),interpolated_sh.shape[0])
#%%
surf_holder = []
for surf in [interpolated_sh, interpolated_yy]:
    M,C = get_M_and_C(surf,Z)
    updated_surface = remove_modes(M,C,Z,[0,1,2,4])
    surf_holder.append(updated_surface)

surf_holder[1] = np.multiply(surf_holder[1],-1e-3)
plot_mirrors_side_by_side(surf_holder[0],surf_holder[1],'Reconstructed wavefront comparison', subtitles=['SHWFS has ','PD has '])
