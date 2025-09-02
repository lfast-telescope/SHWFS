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

base_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/'

sh_path = base_path + 'on-sky/20250515/SHWFS/'  # path pointing to folder of SH images for current night
reference_path = os.path.join(sh_path, '211851/')  # path to set of images used for pupil definition
folder_path = os.path.join(sh_path, '211851/')

#%%
mean_surface = full_SHWFS_reconstruction(sh_path, reference_path, redefine_pupil=False)

