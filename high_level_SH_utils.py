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
from zernike import *
from LFAST_wavefront_utils import *

def compute_wavefront(image,referenceX, referenceY, magnification, nominalSpot, xyr=None, output_plots = False):

    arrows, ideal_coords, actual_coords, pupil_center, pupil_radius = get_quiver(image, nominalSpot[0], nominalSpot[1], magnification, xyr, output_plots)

    #Redo these numbers with N9 - different focal length.

    # Calibrate slope data according to the quiver.
    s = 5.63116  # distance between 2 stars in pixels
    tanTheta = 5.5  # angular separation of 2 stars in arcseconds. Small angle approximation.
    tanTheta = tanTheta * 4.848e-6  # angular separation in radians
    d = s / tanTheta  # calibrated distance in pixels between the lens array and sensor
    slope = 0.5 * arrows / d  # calibrated slope
    #The 0.5 factor comes from accounting for the mirror reflection

    # Get the regular slope maps.
    ptsNumX = (np.max(ideal_coords[:, 0]) - np.min(ideal_coords[:, 0])) / magnification
    ptsNumY = (np.max(ideal_coords[:, 1]) - np.min(ideal_coords[:, 1])) / magnification
    slopeMagnification = 1  # equal to 1 once the slope is calibrated in previous sections
    mirrorDiameter = 30 * 25.4  # diameter of the primary mirror in millimeters
    lensletDiameter = 0.3 #lenslet pitch in millimeters
    mirrorFocallength = 2534.3 #units: mm
    reimagingLensFocallength = 25 #units : mm

    actual_spacing = lensletDiameter * mirrorFocallength / reimagingLensFocallength  # actual spacing on the primary mirror indicated by micro lenses. Units = mm
    N = round(mirrorDiameter / actual_spacing) # number of micro lenses on the diameter

    #magnification is the number of pixels from one lenslet focus to the next
    #so lateral_magnification is the distance on the mirror corresponding to one pixel on the SH sensor
    lateralMagnification = actual_spacing / magnification  # actual spacing on the primary mirror covered by a pixel
    integration_step = 3 #Units:mm. Size of reconstructed slope pixel

    regularSlopeX, regularSlopeY, xCoordinates, yCoordinates = quiver2regular_slope(slope, ideal_coords, slopeMagnification, lateralMagnification, integration_step, output_plots, pupil_center, pupil_radius, interpolation='RBF')

    mirrorPos = lateralMagnification * ideal_coords

    regularSlopeX, regularSlopeY = trim_maps_to_square(regularSlopeX, regularSlopeY)

    #shape_diff = SlopeIntegration(regularSlopeX, regularSlopeY)
    shape_diff = SouthwellIntegration(regularSlopeX,regularSlopeY)

    return shape_diff

def xyr_pupil_definition(extend_path, reference_path):
    jup_image = average_folder_of_images(extend_path)

    if not os.path.isfile(reference_path + 'xyr.pkl'):
        xyr = define_pupil_from_extended_object(jup_image, thresh=127)
        with open(reference_path + 'xyr.pkl', 'wb') as f:
            pickle.dump(xyr,f)
    else:
        with open(reference_path + 'xyr.pkl', 'rb') as f:
            xyr = pickle.load(f)

    return xyr, jup_image

def lenslet_definition(folder_path, reference_path, xyr, output_plots):
    if os.path.isfile(reference_path + 'references.pkl'):
        with open(reference_path + 'references.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        referenceX = loaded_dict["refX"]
        referenceY = loaded_dict["refY"]
        magnification = loaded_dict["magnification"]
        nominalSpot = loaded_dict["nominalSpot"]
        rotation = loaded_dict["rotation"]
    else:
        referenceX, referenceY, magnification, nominalSpot, rotation = define_reference(folder_path, xyr, output_plots)
        references = {"refX":referenceX,
                      "refY":referenceY,
                      "magnification":magnification,
                      "nominalSpot":nominalSpot,
                      "rotation":rotation}
        with open(reference_path + 'references.pkl', 'wb') as f:
            pickle.dump(references,f)
    return referenceX,referenceY,magnification,nominalSpot,rotation

def process_folder_of_images(folder_path,rotation, xyr, referenceX, referenceY, magnification, nominalSpot, xyr_cropped, output_plots):
    surfaces = []
    remove_coef = [0, 1, 2, 4]

    for num, filename in enumerate(os.listdir(folder_path)):
        starting_time = time.time()
        image,_ = prepare_image(folder_path + '/' + filename, rotation, xyr)
        shape_diff = compute_wavefront(image,referenceX, referenceY, magnification, nominalSpot, xyr_cropped, output_plots=False)

        if num==0 or shape_diff.shape[0] != Z[1].shape[0]:
            Z = General_zernike_matrix(44,int(15*25.4 * 1e3),int(3*25.4*1e3),shape_diff.shape[0])
        M,C = get_M_and_C(shape_diff, Z)
        updated_surface = remove_modes(M,C,Z,remove_coef)*1e3
        surfaces.append(updated_surface)

        if output_plots:
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

        print('SH image #' + str(num) + ' processed in ' + str(round(time.time()-starting_time, 1)) + ' seconds')

    return surfaces, Z

#%%
def suggest_next_iteration_of_TEC_correction(current_eigenvalues, folder_path, tec_path, mean_surface, eigenvectors, clear_aperture_outer, clear_aperture_inner, Z, eigenvalue_bounds, eigen_gain):
    timestamp = folder_path.split('/')[-2]

    interpolated_surface = griddata_interpolater(mean_surface, eigenvectors[0], clear_aperture_outer,
                                                 clear_aperture_inner)
    reduced_surface, eigenvalue_delta = optimize_TECs(interpolated_surface, eigenvectors, current_eigenvalues,
                                                      eigenvalue_bounds, clear_aperture_outer, clear_aperture_inner, Z,
                                                      metric='rms_neutral')
    eigenvalues = current_eigenvalues + eigenvalue_delta * eigen_gain
    write_eigenvalues_to_csv(tec_path + 'corrections_based_on_' + timestamp + '.csv', eigenvalues)

    vals = mean_surface[~np.isnan(mean_surface)]
    sorted_vals = np.sort(vals)
    sorted_index = int(0.001 * len(sorted_vals))  # For peak-valley, throw out the extreme tails
    pv = sorted_vals[-sorted_index] - sorted_vals[sorted_index]
    rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals))

    plt.imshow(mean_surface)
    plt.xticks([])
    plt.yticks([])
    plt.title('Surface at ' + timestamp + ' has ' + str(np.round(rms * 1000)) + 'nm wavefront error')
    plt.show()

    return eigenvalues, mean_surface