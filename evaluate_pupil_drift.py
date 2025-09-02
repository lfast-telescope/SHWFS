import os
from skimage import io
import cv2 as cv
from SH_utils import *
import time
import pickle
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib import patches
from aperture_utils import *
from high_level_SH_utils import *
from zernike import *
from LFAST_wavefront_utils import *
from LFAST_TEC_output import *

in_to_m = 25.4e-3
OD = 31.9 * in_to_m  # Outer mirror diameter (m)
ID = 6 * in_to_m  # Central obscuration diameter (m)
clear_aperture_outer = 0.47 * OD
clear_aperture_inner = ID / 2

reference_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/on-sky/20250319/SHWFS/'

extended_object_start_time = 545
extended_object_end_time = 606

list_of_timestamps = [float(subfolder) for subfolder in os.listdir(reference_path)]
list_of_booleans = [timestamp >= extended_object_start_time and timestamp <= extended_object_end_time for timestamp in list_of_timestamps]

extended_object_subfolders = np.array(os.listdir(reference_path))[list_of_booleans]

unresolved_object_start_time = 608
unresolved_object_end_time = 622

list_of_timestamps = [float(subfolder) for subfolder in os.listdir(reference_path)]
list_of_booleans = [timestamp >= unresolved_object_start_time and timestamp <= unresolved_object_end_time for timestamp in list_of_timestamps]

unresolved_object_subfolders = np.array(os.listdir(reference_path))[list_of_booleans]
#%%
for subfolder in extended_object_subfolders:
    extend_path = os.path.join(reference_path, subfolder) + '/'
    if os.path.exists(extend_path + 'xyr.npy'):
        os.remove(extend_path + 'xyr.npy')
    xyr, extend_image = xyr_pupil_definition(extend_path, extend_path)
    referenceX,referenceY,magnification,nominalSpot,rotation = lenslet_definition(extend_path, reference_path, xyr, output_plots=False)

