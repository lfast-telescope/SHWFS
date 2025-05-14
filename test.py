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

#%%
if __name__ == "__main__":

    reference_path = r'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/on-sky/20250501/' #path pointing to images used for pupil definition
    reference_path = os.path.join(reference_path,'220452/') #path to
    folder_path = os.path.join(reference_path, '221340/')

    eigenvectors_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M9/eigenvectors.npy'
    tec_path = eigenvectors_path
    eigenvectors = np.load(eigenvectors_path,allow_pickle=True)
    eigenvalue_bounds = []
    eigen_gain = 0.4
    eigenvalues = [0]*24
    for i in np.arange(len(eigenvalues)):
        eigenvalue_bounds.append([-0.6, 0.6])

    in_to_m = 25.4e-3
    OD = 31.9 * in_to_m  # Outer mirror diameter (m)
    ID = 6 * in_to_m  # Central obscuration diameter (m)
    clear_aperture_outer = 0.47 * OD
    clear_aperture_inner = ID / 2

    xyr, extend_image = xyr_pupil_definition(reference_path, reference_path)
    jup_crop = crop_image(extend_image,xyr)
    output_plots = False
    referenceX,referenceY,magnification,nominalSpot,rotation = lenslet_definition(reference_path, reference_path, xyr, output_plots=True)

    X, Y = np.meshgrid(np.arange(extend_image.shape[0]), np.arange(extend_image.shape[1]))
    proposed_pupil = np.sqrt(np.square(X - xyr[0]) + np.square(Y - xyr[1])) < xyr[2]
    cropped_pupil = crop_image(proposed_pupil, xyr)
    xyr_cropped = [cropped_pupil.shape[0] / 2, cropped_pupil.shape[1] / 2, xyr[-1]]

    surfaces, Z = process_folder_of_images(folder_path, rotation, xyr, referenceX, referenceY, magnification,
                                           nominalSpot, xyr_cropped, output_plots=False)
    mean_surface = np.mean(surfaces, axis=0)

    grid_diameter = (clear_aperture_outer*2)
    corresponding_pupil = make_lfast_aperture(mean_surface.shape,grid_diameter)

    current_eigenvalues = eigenvalues.copy()
    eigenvalues, surface = suggest_next_iteration_of_TEC_correction(current_eigenvalues, folder_path, tec_path, mean_surface, eigenvectors, clear_aperture_outer, clear_aperture_inner, Z, eigenvalue_bounds, eigen_gain)