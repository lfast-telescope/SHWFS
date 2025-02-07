import os

import cv2
from SH_utils import *

def prepare_image(file_name, output_plots = False):
    # Read the image
    image_color = cv2.imread(file_name)

    # Check if the image was loaded successfully
    if image_color is None:
        raise FileNotFoundError(f"Image file not found at {file_name}")

    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # Crop the image using the CropImage function
    im_crop = crop_image(image_gray)

    # Optional: Display the cropped image
    if output_plots:
        cv2.imshow("Cropped Image", im_crop.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image = improve_image_rotation(im_crop)

    return image

def define_reference(folder_path, output_plots = False):
    refX_holder = []
    refY_holder = []
    magnification_holder = []
    nominalSpot_holder = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image = prepare_image(file_path)
        referenceX, referenceY, magnification, nominalSpot = GetGrid(image, output_plots)
        refX_holder.append(referenceX)
        refY_holder.append(referenceY)
        magnification_holder.append(magnification)
        nominalSpot_holder.append(nominalSpot)
    return np.mean(refX_holder), np.mean(refY_holder), np.mean(magnification_holder), np.mean(nominalSpot_holder,0)

def compute_wavefront(image,referenceX, referenceY, magnification, nominalSpot, output_plots = False):

    arrows, ideal_coords, actual_coords, pupil_center, pupil_radius = get_quiver(image, nominalSpot[0], nominalSpot[1], magnification, output_plots)

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

if __name__ == "__main__":
    output_plots = False
    folder_path =r'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/on-sky/20240118/test_sample/'
    referenceX, referenceY, magnification, nominalSpot = define_reference(folder_path, output_plots)
    #compute_wavefront(image, referenceX, referenceY, magnification, nominalSpot,



