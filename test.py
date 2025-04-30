import os
from skimage import io
import cv2 as cv
from SH_utils import *
import numpy as np
import time
import pickle
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib import patches
from aperture_utils import *
from high_level_SH_utils import *
from Zernike import *
from LFAST_wavefront_utils import *
from LFAST_TEC_output import *

#%%
if __name__ == "__main__":

    reference_path = '/run/user/1000/gvfs/google-drive:host=gmail.com,user=lfastelescope/0ABWAmPdRyHoKUk9PVA/1tMLwOJ8KbNA7trH1MLRVAMo835F-kGyq/1I28oSs6aYDfuuy_nnJWG0Wk-pwYBrVco/1s42YTPTV4NQ4IuA3MDr9d1m9C9c6DWY5/1PJlgxbQX9DTzFsJuBKm-8rC8am762weZ/1PAW2Bi15zNQnlk04FOovyAgeaw7EXDMi/1XkTBXKykRwB29X5_gyEc9KpuYeQfFXu-'
    jupiter_subfolder = '/0303/'
    extend_path = reference_path + jupiter_subfolder
    
    polaris_path = reference_path + '/insanity_check/'
    
    
    
    eigenvectors_path = '/home/steward/lfast/PMC_GUI/tec_csv/'
    tec_path = eigenvectors_path
    eigenvectors = np.load(eigenvectors_path + 'eigenvectors.npy',allow_pickle=True)
    eigenvalue_bounds = []
    eigen_gain = 0.3
    eigenvalues = [0]*24
    for i in np.arange(len(eigenvalues)):
        eigenvalue_bounds.append([-0.6, 0.6])

    in_to_m = 25.4e-3
    OD = 31.9 * in_to_m  # Outer mirror diameter (m)
    ID = 6 * in_to_m  # Central obscuration diameter (m)
    clear_aperture_outer = 0.47 * OD
    clear_aperture_inner = ID / 2

    xyr, extend_image = xyr_pupil_definition(extend_path, reference_path)
    jup_crop = crop_image(extend_image,xyr)
    grid_diameter = (clear_aperture_outer*2) * (jup_crop.shape[0]/(2*xyr[-1]))
    corresponding_pupil = make_lfast_aperture(jup_crop.shape,grid_diameter)
    output_plots = False
    
    folder_path = polaris_path
    
    #This is the geometery definition that needs to be updated throughout night
    referenceX,referenceY,magnification,nominalSpot,rotation = lenslet_definition(folder_path, reference_path, xyr, output_plots)

    X, Y = np.meshgrid(np.arange(extend_image.shape[0]), np.arange(extend_image.shape[1]))
    proposed_pupil = np.sqrt(np.square(X - xyr[0]) + np.square(Y - xyr[1])) < xyr[2]
    cropped_pupil = crop_image(proposed_pupil, xyr)
    xyr_cropped = [cropped_pupil.shape[0] / 2, cropped_pupil.shape[1] / 2, xyr[-1]]

    surfaces, Z = process_folder_of_images(folder_path, rotation, xyr, referenceX, referenceY, magnification,
                                           nominalSpot, xyr_cropped, output_plots=False)
    mean_surface = np.mean(surfaces, axis=0)*-1
    
    Z = General_zernike_matrix(44,int(15*25.4 * 1e3),int(3*25.4*1e3), 500)
    current_eigenvalues = eigenvalues.copy()
    eigenvalues, surface = suggest_next_iteration_of_TEC_correction(current_eigenvalues, folder_path, tec_path, mean_surface, eigenvectors, clear_aperture_outer, clear_aperture_inner, Z, eigenvalue_bounds, eigen_gain)

# %%
for surface in surfaces:
    M,C = get_M_and_C(surface,Z)
    updated_surface = remove_modes(M,C,Z,[0,1,2,4])*1e3
    vals = updated_surface[~np.isnan(updated_surface)]
    sorted_vals = np.sort(vals)
    sorted_index = int(0.001 * len(sorted_vals))  # For peak-valley, throw out the extreme tails
    pv = sorted_vals[-sorted_index] - sorted_vals[sorted_index]
    rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals))

    plt.imshow(updated_surface)
    plt.xticks([])
    plt.yticks([])
    plt.title('Surface has ' + str(np.round(rms*1000)) + 'nm wavefront error')
    plt.show()
