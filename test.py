import os
from skimage import io
import cv2 as cv
from SH_utils import *
import time
import pickle
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib import patches
#%%

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


#%%
if __name__ == "__main__":
#%%
    jupiter_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/on-sky/20250206/0519/'
    folder_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/on-sky/20250206/0309/'
    jup_image = average_folder_of_images(jupiter_path)

    if not os.path.isfile(jupiter_path + 'xyr.pkl'):
        xyr = define_pupil_from_extended_object(jup_image, thresh=127)
        with open(jupiter_path + 'xyr.pkl', 'wb') as f:
            pickle.dump(xyr,f)
    else:
        with open(jupiter_path + 'xyr.pkl', 'rb') as f:
            xyr = pickle.load(f)

#%%
    print('Start test')
    starting_time = time.time()
    output_plots = False

    if os.path.isfile(folder_path + 'references.pkl'):
        with open(folder_path + 'references.pkl', 'rb') as f:
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
        with open(folder_path + 'references.pkl', 'wb') as f:
            pickle.dump(references,f)
    print('Define reference: ' + str(round(time.time() - starting_time)))
#%%
    starting_time = time.time()

    image,_ = prepare_image(folder_path + '/' + os.listdir(folder_path)[0], rotation, xyr)
    X, Y = np.meshgrid(np.arange(jup_image.shape[0]), np.arange(jup_image.shape[1]))
    proposed_pupil = np.sqrt(np.square(X - xyr[0]) + np.square(Y - xyr[1])) < xyr[2]
    cropped_pupil = crop_image(proposed_pupil, xyr)
    xyr_cropped = [cropped_pupil.shape[0] / 2, cropped_pupil.shape[1] / 2, xyr[-1]]

    print('Prepare image: ' + str(round(time.time() - starting_time)))
    starting_time = time.time()
    shape_diff = compute_wavefront(image,referenceX, referenceY, magnification, nominalSpot, xyr_cropped, output_plots = False)
    print('Compute wavefront: ' + str(round(time.time() - starting_time)))

#%%


if False:
    im_blur = gaussian_filter(image, 11)
    img = im_blur.astype('uint8')
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=2, minDist=200,
                              param1=27, param2=30, minRadius=500, maxRadius=1000)
    fig,ax = plt.subplots()
    plt.imshow(img)
    artist = patches.Circle(circles[0][0][:2],circles[0][0][-1],fill=False,color='red')
    ax.add_artist(artist)
    plt.show()


