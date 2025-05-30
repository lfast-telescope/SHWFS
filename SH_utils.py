import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter, rotate, maximum_filter
from scipy.signal import find_peaks, medfilt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import griddata, RBFInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from skimage.feature import peak_local_max
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits

def find_radius(image):
    """
    Compute pupil radius from SH lenslet image.
    Used as preprocessing for wavefront reconstruction.

    Parameters:
    image (numpy.ndarray): Input image as a 2D numpy array.

    Returns:
    float: Estimated radius of the pupil.
    """
    # Convert image to double (float64)
    image = image.astype(np.float64)

    # Normalize the image to the range [0, 255] and convert to uint8
    max_value = np.max(image)
    min_value = np.min(image)
    image_uint8 = ((image - min_value) / (max_value - min_value) * 255).astype(np.uint8)

    # Apply global thresholding to binarize the image
    _, binary_im = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the coordinates of non-zero pixels
    rows, cols = np.where(binary_im > 0)

    # Calculate the difference between max and min row/column
    row_top = np.sort(rows)[2]
    row_bot = np.sort(rows)[-2]
    col_top = np.sort(cols)[2]
    col_bot = np.sort(cols)[-2]
    row_diff = row_bot - row_top
    col_diff = col_bot - col_top

    # Compute the radius as the average of row_diff and col_diff divided by 4
    radius = (row_diff + col_diff) / 4

    return radius


def find_center(image):
    """
    Compute the centroid of an image.

    Parameters:
        image: Input image (numpy array).

    Returns:
        x_cent: x-coordinate of the centroid.
        y_cent: y-coordinate of the centroid.
    """
    # Convert image to double (float64)
    image = image.astype(np.float64)

    # Get image dimensions
    im_size = image.shape
    x = np.arange(1, im_size[0] + 1)  # 1-based indexing like MATLAB
    y = np.arange(1, im_size[1] + 1)

    # Create meshgrid
    X, Y = np.meshgrid(y, x)

    # Compute weighted sums
    im_x = image * X
    im_y = image * Y

    # Compute centroid
    x_cent = np.sum(im_x) / np.sum(image)
    y_cent = np.sum(im_y) / np.sum(image)

    return x_cent, y_cent

def FindClosestPoint(pointSet, point1, point2=None, point3=None):
    """
    This function is used to find the point closest to the reference points.

    INPUTS:
        pointSet: N x 2 point set in format [x y] used to find the point closest to reference points.
        point1, point2, point3: reference points.

    OUTPUT:
        closestPoint: closest point.
        ptIndex: closest point index in 'pointSet'.
    """

    # Validate inputs
    if pointSet.size == 0:
        return np.array([np.nan, np.nan]), np.nan

    if point2 is None and point3 is None:
        # Case 1: Only point1 is provided
        distances = cdist(pointSet, [point1])
        ptIndex = np.argmin(distances)
        closestPoint = pointSet[ptIndex, :]

    elif point3 is None:
        # Case 2: point1 and point2 are provided
        # Find 6 closest points to point1 and point2
        ptsNum = pointSet.shape[0]
        centroidsTemp = pointSet.copy()
        k1, k2 = [], []

        for i in range(6):
            if ptsNum == 0:
                break
            idx1 = np.argmin(cdist(centroidsTemp, [point1]))
            idx2 = np.argmin(cdist(centroidsTemp, [point2]))
            k1.append(idx1)
            k2.append(idx2)
            centroidsTemp[idx1, :] = -100
            centroidsTemp[idx2, :] = -100
            if idx1 == idx2:
                ptsNum -= 1
            else:
                ptsNum -= 2

        k = np.union1d(k1, k2)
        ptsNumTBD = len(k)
        distanceTotal = np.zeros(ptsNumTBD)

        for i in range(ptsNumTBD):
            distance1 = np.linalg.norm(pointSet[k[i], :] - point1)
            distance2 = np.linalg.norm(pointSet[k[i], :] - point2)
            distanceTotal[i] = distance1 + distance2

        distanceMin = np.min(distanceTotal)
        distanceMinIndex = np.where(distanceTotal == distanceMin)[0]
        ptIndex = k[distanceMinIndex]
        closestPoint = pointSet[ptIndex, :]

    else:
        # Case 3: point1, point2, and point3 are provided
        # Find 6 closest points to point1, point2, and point3
        ptsNum = pointSet.shape[0]
        centroidsTemp = pointSet.copy()
        k1, k2, k3 = [], [], []

        for i in range(6):
            if ptsNum == 0:
                break
            idx1 = np.argmin(cdist(centroidsTemp, [point1]))
            idx2 = np.argmin(cdist(centroidsTemp, [point2]))
            idx3 = np.argmin(cdist(centroidsTemp, [point3]))
            k1.append(idx1)
            k2.append(idx2)
            k3.append(idx3)
            centroidsTemp[idx1, :] = -100
            centroidsTemp[idx2, :] = -100
            centroidsTemp[idx3, :] = -100
            if idx1 == idx2 and idx2 == idx3:
                ptsNum -= 1
            elif idx1 == idx2 or idx2 == idx3:
                ptsNum -= 2
            else:
                ptsNum -= 3

        k0 = np.union1d(k1, k2)
        k = np.union1d(k0, k3)
        ptsNumTBD = len(k)
        distanceTotal = np.zeros(ptsNumTBD)

        for i in range(ptsNumTBD):
            distance1 = np.linalg.norm(pointSet[k[i], :] - point1)
            distance2 = np.linalg.norm(pointSet[k[i], :] - point2)
            distance3 = np.linalg.norm(pointSet[k[i], :] - point3)
            distanceTotal[i] = distance1 + distance2 + distance3

        distanceMin = np.min(distanceTotal)
        distanceMinIndex = np.where(distanceTotal == distanceMin)[0]
        ptIndex = k[distanceMinIndex]
        closestPoint = pointSet[ptIndex, :]

    return closestPoint, ptIndex


def crop_image(image, xyr=None):
    """
    Crop the image based on the center and radius.

    Parameters:
        image: Input image (numpy array).

    Returns:
        image_crop: Cropped image.
    """
    # Convert image to double (float64)
    if xyr is not None:
        x_cent = xyr[0]
        y_cent = xyr[1]
        radius = xyr[2]*1.1

    else:
        image = image.astype(np.float64)

        # Find the center of the image
        x_cent, y_cent = find_center(image)

            # Find the radius
        radius = find_radius(image) * 1.1  # Add a little fudge so that all the spots fit

        # Optional: Visualize the circle (for debugging)
        if False:
            plt.imshow(image, cmap='gray')
            circle = Circle((x_cent, y_cent), radius, color='r', fill=False)
            plt.gca().add_patch(circle)
            plt.show()

    # Get image dimensions
    y_size, x_size = image.shape

    # Calculate boundaries
    y_top = int(y_cent - radius)
    y_bot = int(y_cent + radius)
    x_left = int(x_cent - radius)
    x_right = int(x_cent + radius)

    image_crop = image[y_top:y_bot, x_left:x_right]

    if image_crop.shape[0] > image_crop.shape[1]:
        image_crop = image_crop[1:,:]
    elif image_crop.shape[0] < image_crop.shape[1]:
        image_crop = image_crop[:,1:]

    return image_crop


def find_best_rotation(imCrop, angle_start, angle_range):
    """
    This function finds the best rotation angle for an image by evaluating
    the average peak intensity of smoothed row means after rotating the image.

    Parameters:
    imCrop (numpy.ndarray): Input image (2D array).
    angle_start (float): Starting angle for rotation.
    angle_range (float): Range of angles to explore around the starting angle.

    Returns:
    float: Best rotation angle.
    """
    num_trials = 3
    pk_holder = np.zeros(num_trials)
    angles = np.linspace(angle_start - angle_range, angle_start + angle_range, num_trials)

    for i, angle in enumerate(angles):
        # Rotate the image
        pk_holder[i] = -1 * rotation_objective_function(angle, imCrop)

    # Find the angle corresponding to the maximum peak value
    best_angle = angles[np.argmax(pk_holder)]

    return best_angle

def rotation_objective_function(angle, imCrop):
    # Rotate the image
    rotImage = rotate(imCrop, angle, reshape=False)

    # Compute the mean along the rows
    x = np.mean(rotImage, axis=0)

    # Smooth the row means using a median filter
    x_smooth = medfilt(x, kernel_size=5)

    # Find peaks in the smoothed data
    peaks, _ = find_peaks(x_smooth, distance=20)

    # Sort peaks in descending order and take the top 20
    peak_list = np.sort(x_smooth[peaks])[::-1]
    peak_avg = np.mean(peak_list[:20])

    return -peak_avg

def improve_image_rotation(im_crop, use_optimizer = True):
    angle_start = 0
    angle_range = 1

    if use_optimizer:
        res = minimize_scalar(rotation_objective_function, args = (im_crop,),options={'maxiter':10})
        best_angle = res.x
    else:
        for trial in np.arange(5):
            best_angle = find_best_rotation(im_crop, angle_start, angle_range)
            if not(best_angle == (angle_start-angle_range) or best_angle == (angle_start+angle_range)):
                angle_range = angle_range / 2
            angle_start = best_angle

    image = rotate(im_crop, best_angle, reshape=False)

    return image, best_angle


def GetGrid(image, xyr=None, output_plots=False):
    # Get pixel coordinates of bright spots.

    # Binarize the image to get a mask.
    image = image.astype(np.float64)
    maxValue = np.max(image)
    minValue = np.min(image)
    imageUint8 = ((image - minValue) / (maxValue - minValue) * 255).astype(np.uint8)
    _, binaryIm = cv2.threshold(imageUint8, np.max(imageUint8)*0.01, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if output_plots:
        plt.imshow(binaryIm, cmap='gray')
        plt.title("Binary Image")
        plt.show()

    # Use the mask to get the target regions on the smoothed image.
    sigma = 3  # (!)
    smoothIm = gaussian_filter(image, sigma)
    maskedIm = binaryIm * smoothIm

    if output_plots:
        plt.imshow(maskedIm, cmap='gray')
        plt.title("Masked Image")
        plt.show()

    # Find local maximums of target regions to show positions of bright spots.
    brightSpots = maximum_filter(maskedIm, size=3) == maskedIm
    brightSpots = brightSpots & (maskedIm > 0)

    if output_plots:
        plt.imshow(brightSpots, cmap='gray')
        plt.title("Bright Spots")
        plt.show()

    rows, cols = np.where(brightSpots == 1)

    # Get magnification and central reference of bright spots.
    # Calculate the magnification.
    spotsNum = len(rows)
    spotSet = np.column_stack((cols, rows))
    distance = np.zeros(spotsNum)
    for i in range(spotsNum):
        judgedSpot = spotSet[i, :]
        judgedSpotSet = np.delete(spotSet, i, axis=0)
        closestSpot, _ = FindClosestPoint(judgedSpotSet, judgedSpot)
        distance[i] = np.linalg.norm(closestSpot - judgedSpot)
    sortedDist = np.sort(distance)
    croppedPercent = 0.1  # (!)
    croppedNum = int(croppedPercent * spotsNum)
    croppedDist = sortedDist[croppedNum:spotsNum - croppedNum]
    magnification = np.max(croppedDist)

    # Find the central reference.
    if xyr is None:
        rowAve = np.mean(rows)
        colAve = np.mean(cols)
    else:
        rowAve = xyr[0]
        colAve = xyr[1]
    centralSpot = np.array([colAve, rowAve])
    closestSpot, _ = FindClosestPoint(spotSet, centralSpot)
    referenceX = closestSpot[0]
    referenceY = closestSpot[1]

    nominal_col = referenceX + np.arange(-5,5) * magnification
    nominal_col_arg = np.argmin(np.abs(nominal_col - colAve))
    nominal_row = referenceY + np.arange(-5,5) * magnification
    nominal_row_arg = np.argmin(np.abs(nominal_row - rowAve))
    nominalSpot = np.array([nominal_col[nominal_col_arg],nominal_row[nominal_row_arg]])

    if output_plots:
        markersize = 1
        plt.imshow(maskedIm, cmap='gray')
        plt.scatter(centralSpot[0], centralSpot[1], s = markersize, color='g', label='Central Spot')
        plt.scatter(closestSpot[0], closestSpot[1], s = markersize, color='r', label='Closest Spot')
        plt.scatter(nominalSpot[0], nominalSpot[1], s = markersize, color='c', label='Nominal Spot')
        plt.legend()
        plt.title("Final Reference Points")
        plt.show()

    return referenceX, referenceY, magnification, nominalSpot


def target_contrast_control(image):
    """
    This function is used to pre-process the target images. Primary work is
    to improve image contrast.

    INPUTS:
        image: Input image (numpy array).
    OUTPUTS:
        corrected_im: Contrast-enhanced image (numpy array).
    """
    # Stretch the grayscale
    image = image.astype(np.float64)
    max_value = np.max(image)
    min_value = np.min(image)
    trans_im = np.uint8(255 * (image - min_value) / (max_value - min_value))

    # Uncomment to visualize the stretched image
    # plt.imshow(trans_im, cmap='gray')
    # plt.title("Stretched Image")
    # plt.pause(0.2)
    # plt.close()

    # Improve imaging effects using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corrected_im = clahe.apply(trans_im)

    # Uncomment to visualize the contrast-enhanced image
    # plt.imshow(corrected_im, cmap='gray')
    # plt.title("Contrast-Enhanced Image")
    # plt.pause(0.2)
    # plt.close()

    return corrected_im

def get_quiver(image, reference_x, reference_y, magnification, xyr=None, output_plots=False):
    """
    Get the quiver for actual points according to the regular grid.

    INPUTS:
        image: captured image of the actual mirror (numpy array);
        reference_x/reference_y: x/y coordinate of the reference point;
        magnification: magnification of the regular grid.
    OUTPUTS:
        arrows: N x 2 numpy array showing displacements on the image;
        ideal_coords/actual_coords: N x 2 numpy array showing coordinates of available ideal/actual points.
    """
    # Generate the regular grid
    size_y, size_x = image.shape
    x_left = int(np.floor(reference_x / magnification))
    x_right = int(np.floor((size_x - reference_x) / magnification))
    y_left = int(np.floor(reference_y / magnification))
    y_right = int(np.floor((size_y - reference_y) / magnification))
    x_num = x_left + x_right + 1  # number of knots in x direction
    y_num = y_left + y_right + 1  # number of knots in y direction

    x = magnification * np.arange(x_num) + 1
    y = magnification * np.arange(y_num) + 1

    displacement_x = reference_x - x[x_left]
    displacement_y = reference_y - y[y_left]
    x += displacement_x
    y += displacement_y  # align with the reference point

    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()

    # Find center and radius
    if xyr is None:
        x_cent, y_cent = find_center(image)
        radius = find_radius(image) * 1.05  # Add a little fudge so that all the spots fit
    else:
        x_cent = xyr[0]
        y_cent = xyr[1]
        radius = xyr[2]

    central_obscuration_radius = radius * 6 / 30

    if output_plots:
        fig,ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.scatter(X, Y, c='r', marker='+')
        artist = Circle(xyr[:2],xyr[2],fill=False,color='g')
        ax.add_artist(artist)
        plt.show()

    # Stretch contrast of the image
    image = target_contrast_control(image)  # Replace with your implementation
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Find arrows
    cross_num = len(X)
    threshold_percent_cutoff = 0.993 # (Sometimes arbitrary numbers can eat your lunch)
    grayscale_threshold = np.sort(image.ravel())[int(len(image.ravel())*threshold_percent_cutoff)]
    half_size = int(round(0.5 * magnification))  #Semidiameter of lenslet
    available_pts_num = 0
    ideal_coords = []
    actual_coords = []

    if output_plots:
        plt.imshow(image, cmap='gray')
        circle = plt.Circle((x_cent, y_cent), radius, color='r', fill=False)
        plt.gca().add_patch(circle)

    # Evaluate if the cross should be evaluated within the pupil
    for i in range(cross_num):
        # Get the to-be-judged region
        lower_x = int(round(X[i] - half_size))
        lower_x = max(lower_x, 0)
        upper_x = int(round(X[i] + half_size))
        upper_x = min(upper_x, size_x - 1)

        lower_y = int(round(Y[i] - half_size))
        lower_y = max(lower_y, 0)
        upper_y = int(round(Y[i] + half_size))
        upper_y = min(upper_y, size_y - 1)

        target_region = image[lower_y:upper_y + 1, lower_x:upper_x + 1]

        # Judge if there is an available point
        target_size_x = target_region.shape[1]
        target_size_y = target_region.shape[0]
        x_target = np.arange(target_size_x)
        y_target = np.arange(target_size_y)
        x_target_1d, y_target_1d = np.meshgrid(x_target, y_target)

        distance_from_center = np.sqrt((x_target_1d + lower_x - x_cent) ** 2 + (y_target_1d + lower_y - y_cent) ** 2)
        pixel_inside_circle = distance_from_center < radius
        pixel_outside_ID = distance_from_center > central_obscuration_radius
        valid_pixel = pixel_inside_circle * pixel_outside_ID

        # If more than 75% of the region is outside the defined pupil, reject this region
        if np.sum(valid_pixel) < valid_pixel.size * 0.25:
            if output_plots:
                rect = plt.Rectangle((lower_x, lower_y), upper_x - lower_x, upper_y - lower_y, edgecolor='r',
                                     fill=False)
                plt.gca().add_patch(rect)
            continue
        else:
            x_target_1d = x_target_1d.flatten()
            y_target_1d = y_target_1d.flatten()
            target_region_1d = target_region.flatten()

            pt_index = np.argmax(target_region_1d)
            max_grayscale = target_region_1d[pt_index]

            if max_grayscale > grayscale_threshold:  # Judge if there is a captured point
                x_local = x_target_1d[pt_index]
                y_local = y_target_1d[pt_index]

                if output_plots:
                    rect = plt.Rectangle((lower_x, lower_y), upper_x - lower_x, upper_y - lower_y, edgecolor='c',
                                         fill=False)
                    plt.gca().add_patch(rect)
            else:
                if output_plots:
                    rect = plt.Rectangle((lower_x, lower_y), upper_x - lower_x, upper_y - lower_y, edgecolor='y',
                                         fill=False)
                    plt.gca().add_patch(rect)
                continue

        # Save the available point
        x_global = x_local + lower_x
        y_global = y_local + lower_y
        available_pts_num += 1
        ideal_coords.append([X[i], Y[i]])
        actual_coords.append([x_global, y_global])

    ideal_coords = np.array(ideal_coords)
    actual_coords = np.array(actual_coords)
    arrows = actual_coords - ideal_coords
    #Should we subtract the mean (x,y) pixels here?
    #Need to enforce the same number of zones in either dimension?)

    if output_plots:
        plt.show()
        plt.imshow(image, cmap='gray')
        plt.quiver(ideal_coords[:, 0], ideal_coords[:, 1], arrows[:, 0], arrows[:, 1], color='r')
        plt.show()

    if False: #old method: define pupil based on location of "ideal coords" inside pupil - this is inconsistent!
        pupil_center = np.mean(ideal_coords,0)
        pupil_radius = np.median([np.max(ideal_coords,0) - np.mean(ideal_coords,0),np.mean(ideal_coords,0) - np.min(ideal_coords,0)])
    else: #new method: define pupil based on xyr optimizer - this is consistent but can drift / accumulate errors
        pupil_center = np.array(xyr[:2])
        pupil_radius = xyr[-1]

    return arrows, ideal_coords, actual_coords, pupil_center, pupil_radius

def quiver2regular_slope(arrows, idealCoords, slopeMagnification, lateralMagnification, integrationStep, output_plots, pupil_center = None, pupil_radius = None, interpolation = 'griddata', enforce_pupil_size = True):
    """
    This function is used to get the regular slope maps based on irregular slope maps.

    INPUTS:
        arrows: N x 2 array showing displacements on the image;
        idealCoords: N x 2 array showing x and y coordinates of available ideal points;
        slopeMagnification: magnification used to turn arrows to slope data;
        lateralMagnification: magnification of the mirror, which indicates the
            actual spacing on the mirror covered by a single camera pixel (mm);
        integrationStep: integration step, unit in mm.

    OUTPUTS:
        regularSlopeX: regular x slope map;
        regularSlopeY: regular y slope map;
        xCoordinates: x coordinates of fitted points;
        yCoordinates: y coordinates of fitted points.
    """

    # Decompose data
    slope = slopeMagnification * arrows
    mirrorPos = lateralMagnification * idealCoords

    # Get regular grids showing x and y coordinates of to-be-fitted positions
    xStart = round(np.min(mirrorPos[:, 0]))
    xEnd = round(np.max(mirrorPos[:, 0]))
    yStart = round(np.min(mirrorPos[:, 1]))
    yEnd = round(np.max(mirrorPos[:, 1]))

    scaled_pupil_center = pupil_center * lateralMagnification
    scaled_pupil_radius = pupil_radius * lateralMagnification

    if not enforce_pupil_size: #Old method: define the pupil based on the range of focused lenslet spots
        rangeX = np.arange(xStart, xEnd + integrationStep, integrationStep)
        rangeY = np.arange(yStart, yEnd + integrationStep, integrationStep)
    else: #New method: enforce that the fitted pupil is symmetrical in x and y
        rangeX = np.arange(scaled_pupil_center[0]-scaled_pupil_radius,scaled_pupil_center[0]+ scaled_pupil_radius + integrationStep, integrationStep)
        rangeY = np.arange(scaled_pupil_center[1]-scaled_pupil_radius,scaled_pupil_center[1]+ scaled_pupil_radius + integrationStep, integrationStep)

    xCoordinates, yCoordinates = np.meshgrid(rangeX, rangeY)  # x and y coordinates of fitted points

    # Fit slope maps
    if interpolation == 'griddata':
        regularSlopeX = griddata(
            (mirrorPos[:, 0], mirrorPos[:, 1]), slope[:, 0], (xCoordinates, yCoordinates), method='cubic')
        regularSlopeY = griddata(
            (mirrorPos[:, 0], mirrorPos[:, 1]), slope[:, 1], (xCoordinates, yCoordinates), method='cubic')
        regularSlopeX = integrationStep * regularSlopeX  # unify integration spacing
        regularSlopeY = integrationStep * regularSlopeY
    elif interpolation == 'RBF':
        inter_x = RBFInterpolator(mirrorPos, slope[:, 0])
        inter_y = RBFInterpolator(mirrorPos, slope[:, 1])
        coords = np.column_stack([xCoordinates.ravel(), yCoordinates.ravel()])
        vals_x = inter_x(coords)
        vals_y = inter_y(coords)
        regularSlopeX = np.reshape(vals_x, xCoordinates.shape)
        regularSlopeY = np.reshape(vals_y, yCoordinates.shape)

        # Mask out values outside pupil
        scaled_ID_radius = scaled_pupil_radius * (6 / 30)
        OD_mask = np.sqrt(np.square(xCoordinates-scaled_pupil_center[0])+np.square(yCoordinates-scaled_pupil_center[1])) < scaled_pupil_radius
        ID_mask = np.sqrt(np.square(xCoordinates-scaled_pupil_center[0])+np.square(yCoordinates-scaled_pupil_center[1])) > scaled_ID_radius
        pupil_mask = OD_mask * ID_mask
        regularSlopeX[~pupil_mask] = np.nan
        regularSlopeY[~pupil_mask] = np.nan

    else:
        raise ValueError('Have not implemented interpolation technique ' + interpolation)
        regularSlopeX = np.nan
        regularSlopeY = np.nan

    # Display and save regular slope maps (optional)
    if output_plots:
        plt.figure()
        plt.pcolormesh(xCoordinates, yCoordinates, regularSlopeX, shading='auto')
        artist = Circle(scaled_pupil_center, scaled_pupil_radius,fill=False,color='r')
        plt.gca().add_artist(artist)
        plt.colorbar(label='Slope X')
        plt.title('Regular slope in x direction')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.axis('equal')
        plt.show()

    if output_plots:
        plt.figure()
        plt.pcolormesh(xCoordinates, yCoordinates, regularSlopeY, shading='auto')
        artist = Circle(scaled_pupil_center, scaled_pupil_radius,fill=False,color='r')
        plt.gca().add_artist(artist)
        plt.colorbar(label='Slope Y')
        plt.title('Regular slope in y direction')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.axis('equal')
        plt.show()

    return regularSlopeX, regularSlopeY, xCoordinates, yCoordinates


def SlopeIntegration(slopeX, slopeY, type='Southwell'):
    """
    Integrate surface gradient using matrix solution given in Southwell paper.
    The sampled slope data are assumed to align with the surface data. The
    calculation leverages Python's ability to handle large sparse matrices to
    directly solve the linear system of equations. The calculations closely
    follow Southwell's paper. See reference for details.

    INPUT:
        slopeX: 2D array of numbers representing the horizontal gradient of the
            surface. NaN values denote missing data.
        slopeY: 2D array of numbers representing the vertical gradient of the
            surface. NaN values denote missing data. Must have same dimensions
            as slopeX.
        type: string indicator for the type of integration kernel to use. The
            following options are available:  [default = 'Southwell']
            type = 'GradientInverse'
                This type uses a kernel which matches the MATLAB 'gradient'
                function. The gradient is computed as the symmetric
                difference of the +1 pt and the -1 pt. For slope data
                without NaN values, it will exactly undo the gradient
                function. However because this function skips points, it
                can produce checkerboard artifacts in the presence of sharp
                edges.
            type = 'Southwell'
                This type uses the kernel given by equation 5 in the
                reference paper. The gradient is the difference between
                current point and the +1 pt. Because the kernel is
                different from the MATLAB 'gradient' function, it cannot
                undo that calculation. However the kernel does not skip
                points, and therefore produces smoother output. [default]

    OUTPUT:
        surface: 2D array of numbers representing the integrated slope data.
        For every non-NaN gradient point, there will be a surface point.
    """

    # Validate input
    if slopeX.ndim != 2 or slopeY.ndim != 2:
        raise ValueError('SlopeX and slopeY must both be 2-dimensional arrays.')
    if slopeX.shape != slopeY.shape:
        raise ValueError('Dimensions of slopeX and slopeY must match.')

    # Create data mask for output
    thresholdOn = 0  # Use 0 for normal processing. Use 1 for processing that throws out data < threshold
    threshold = -5e-5  # Added by SM (-3*10^-5 for GMT3)

    if thresholdOn == 1:
        badSurface = np.isnan(slopeX) | np.isnan(slopeY) | (slopeX < threshold) | (slopeY < threshold)
    else:
        badSurface = np.isnan(slopeX) | np.isnan(slopeY)

    # Type-specific pre-processing
    if type.lower() == 'southwell':
        slopeX, slopeY = SouthwellShift(slopeX, slopeY)

    # Concatenate slope data and remove NaNs to minimize memory usage
    data = np.concatenate((slopeX.ravel(), slopeY.ravel()))
    data[np.isnan(data)] = 0

    if thresholdOn == 1:
        data[data < threshold] = 0  # Added by SM

    # Size of the sparse matrix (M x N on each side)
    M, N = slopeX.shape

    # Create sparse matrix that describes the conversion from surface to gradient
    if thresholdOn == 1:
        goodSurfaceX = ~np.isnan(slopeX) & (slopeX > threshold)
        goodSurfaceY = ~np.isnan(slopeY) & (slopeY > threshold)
        sparseMatrix = CreateMatrix(goodSurfaceX, goodSurfaceY, type)
    else:
        sparseMatrix = CreateMatrix(~np.isnan(slopeX), ~np.isnan(slopeY), type)

    # Solve the linear system of equations
    surface = lsqr(sparseMatrix, data)[0]

    # Reshape the surface vector into a 2D array
    surface = surface.reshape(M, N)

    # Re-apply the mask
    surface[badSurface] = np.nan

    # Manually remove the mean value
    surface = surface - np.nanmean(surface)

    return surface


def CreateMatrix(goodDataX, goodDataY, type):
    """
    Create sparse matrix that describes the conversion from surface to gradient.

    INPUT:
        goodDataX, goodDataY: 2D array of 1s and 0s where 1s represent data to
            be processed and 0s represent bad data (NaN values). Array size
            must match dimensions of gradient image and must be same for X & Y.
        type: string indicator for the type of integration kernel to use. ('GradientInverse', or 'Southwell')

    OUTPUT:
        sparseMatrix: sparse matrix describing conversion from 2D surface array to two 2D
        gradient arrays.
    """

    M, N = goodDataX.shape
    indexLin = np.arange(M * N).reshape(M, N)

    # Define convolution kernel
    if type.lower() == 'gradientinverse':
        interiorX = [-0.5, 0, 0.5]
        interiorY = [-0.5, 0, 0.5]
        topX = [-0.5, 0, 0.5]
        topY = [0, -1, 1]
        rightX = [-1, 1, 0]
        rightY = [-0.5, 0, 0.5]
        bottomX = [-0.5, 0, 0.5]
        bottomY = [-1, 1, 0]
        leftX = [0, -1, 1]
        leftY = [-0.5, 0, 0.5]
    elif type.lower() == 'southwell':
        interiorX = [0, -1, 1]
        interiorY = [0, -1, 1]
        topX = [0, -1, 1]
        topY = [0, -1, 1]
        rightX = [-1, 1, 0]
        rightY = [0, -1, 1]
        bottomX = [0, -1, 1]
        bottomY = [-1, 1, 0]
        leftX = [0, -1, 1]
        leftY = [0, -1, 1]
    else:
        raise ValueError(f"Type '{type}' not recognized. Allowed types are 'GradientInverse' and 'Southwell'.")

    # Determine sparse matrix indices and values for every point
    indexArrayX = []
    indexArrayY = []

    # Interior points
    selectPts = indexLin[1:-1, 1:-1].ravel()
    indexArrayX.extend(CreateIndexArray(selectPts, [M, 0], interiorX, goodDataX[1:-1, 1:-1].ravel()))
    indexArrayY.extend(CreateIndexArray(selectPts, [0, 1], interiorY, goodDataY[1:-1, 1:-1].ravel()))

    # Top points (except corner)
    selectPts = indexLin[0, 1:-1].ravel()
    indexArrayX.extend(CreateIndexArray(selectPts, [M, 0], topX, goodDataX[0, 1:-1].ravel()))
    indexArrayY.extend(CreateIndexArray(selectPts, [0, 1], topY, goodDataY[0, 1:-1].ravel()))

    # Right points (except corner)
    selectPts = indexLin[1:-1, -1].ravel()
    indexArrayX.extend(CreateIndexArray(selectPts, [M, 0], rightX, goodDataX[1:-1, -1].ravel()))
    indexArrayY.extend(CreateIndexArray(selectPts, [0, 1], rightY, goodDataY[1:-1, -1].ravel()))

    # Bottom points (except corner)
    selectPts = indexLin[-1, 1:-1].ravel()
    indexArrayX.extend(CreateIndexArray(selectPts, [M, 0], bottomX, goodDataX[-1, 1:-1].ravel()))
    indexArrayY.extend(CreateIndexArray(selectPts, [0, 1], bottomY, goodDataY[-1, 1:-1].ravel()))

    # Left points (except corner)
    selectPts = indexLin[1:-1, 0].ravel()
    indexArrayX.extend(CreateIndexArray(selectPts, [M, 0], leftX, goodDataX[1:-1, 0].ravel()))
    indexArrayY.extend(CreateIndexArray(selectPts, [0, 1], leftY, goodDataY[1:-1, 0].ravel()))

    new_method = True

    if new_method:
        # Append rows to indexArrayX
        indexArrayX.extend([
            [indexLin[0, 0], indexLin[0, 0], leftX[1] * goodDataX[0, 0]],  # upper-left
            [indexLin[0, 0], indexLin[0, 0] + M, leftX[2] * goodDataX[0, 0]],  # upper-left
            [indexLin[M - 1, 0], indexLin[M - 1, 0], leftX[1] * goodDataX[-1, 0]],  # lower-left
            [indexLin[M - 1, 0], indexLin[M - 1, 0] + M, leftX[2] * goodDataX[-1, 0]],  # lower-left
            [indexLin[M - 1, N - 1], indexLin[M - 1, N - 1], rightX[1] * goodDataX[0, -1]],  # lower-right
            [indexLin[M - 1, N - 1], indexLin[M - 1, N - 1] - M, rightX[0] * goodDataX[0, -1]],  # lower-right
            [indexLin[0, N - 1], indexLin[0, N - 1], rightX[1] * goodDataX[-1, -1]],  # upper-right
            [indexLin[0, N - 1], indexLin[0, N - 1] - M, rightX[0] * goodDataX[-1, -1]]  # upper-right
        ])

        # Append rows to indexArrayY
        indexArrayY.extend([
            [indexLin[0, 0], indexLin[0, 0], topY[1] * goodDataY[0, 0]],  # upper-left
            [indexLin[0, 0], indexLin[0, 0] + 1, topY[2] * goodDataY[0, 0]],  # upper-left
            [indexLin[M - 1, 0], indexLin[M - 1, 0], bottomY[1] * goodDataY[-1, 0]],  # lower-left
            [indexLin[M - 1, 0], indexLin[M - 1, 0] - 1, bottomY[0] * goodDataY[-1, 0]],  # lower-left
            [indexLin[M - 1, N - 1], indexLin[M - 1, N - 1], bottomY[1] * goodDataY[0, -1]],  # lower-right
            [indexLin[M - 1, N - 1], indexLin[M - 1, N - 1] - 1, bottomY[0] * goodDataY[0, -1]],  # lower-right
            [indexLin[0, N - 1], indexLin[0, N - 1], topY[1] * goodDataY[-1, -1]],  # upper-right
            [indexLin[0, N - 1], indexLin[0, N - 1] + 1, topY[2] * goodDataY[-1, -1]]  # upper-right
        ])
    else:
        # Manually determine corners
        corners = [
            (indexLin[0, 0], leftX[1], leftX[2], topY[1], topY[2]),  # Upper-left
            (indexLin[-1, 0], leftX[1], leftX[2], bottomY[1], bottomY[0]),  # Lower-left
            (indexLin[-1, -1], rightX[1], rightX[0], bottomY[1], bottomY[0]),  # Lower-right
            (indexLin[0, -1], rightX[1], rightX[0], topY[1], topY[2])  # Upper-right
        ]

        for idx, lx1, lx2, ty1, ty2 in corners:
            indexArrayX.extend([[idx, idx, lx1 * goodDataX.flat[idx]], [idx, idx + M, lx2 * goodDataX.flat[idx]]])
            indexArrayY.extend(
                [[idx + M * N, idx, ty1 * goodDataY.flat[idx]], [idx + M * N, idx + 1, ty2 * goodDataY.flat[idx]]])

    # Combine index arrays
    indexArray = np.vstack((indexArrayX, indexArrayY))

    # Remove any points with zero value
    indexArray = indexArray[np.abs(indexArray[:, 2]) > 1e-10, :]

    flag = True
    if flag:
        # Debugging: Check if indices are within bounds
        max_row_index = 2 * M * N
        max_col_index = M * N
        invalid_rows = indexArray[:, 0] >= max_row_index
        invalid_cols = indexArray[:, 1] >= max_col_index

        if np.any(invalid_rows) or np.any(invalid_cols):
            print("Warning: Some indices are out of bounds. Filtering them out.")
            indexArray = indexArray[~(invalid_rows | invalid_cols), :]

    # Create sparse matrix
    sparseMatrix = csr_matrix((indexArray[:, 2], (indexArray[:, 0], indexArray[:, 1])), shape=(2 * M * N, M * N))

    return sparseMatrix


def CreateIndexArray(indices, shift, template, goodData):
    """
    Create index array for sparse matrix construction.

    INPUT:
        indices: 1D array of matrix indices at center of pixel
        shift: 2-element array specifying how much offset to handle +/-
        template: 3-element array with the -shift, center, and +shift values
        goodData: Array of 1s (good) and 0s (bad) describing where the data is
            useful for processing. Must have same number of elements as
            'indices' input.

    OUTPUT:
        indexArray: Nx3 array where columns are [row index, col index, value].
    """

    indexCenter = np.column_stack((indices, indices, template[1] * goodData))

    indexPlusX = []
    indexMinusX = []
    if shift[0] > 0:
        indexPlusX = np.column_stack((indices, indices + shift[0], template[2] * goodData))
        indexMinusX = np.column_stack((indices, indices - shift[0], template[0] * goodData))

    indexPlusY = []
    indexMinusY = []
    if shift[1] > 0:
        indexPlusY = np.column_stack((indices, indices + shift[1], template[2] * goodData))
        indexMinusY = np.column_stack((indices, indices - shift[1], template[0] * goodData))

    if len(indexPlusX) == 0:
        indexArray = np.vstack((indexCenter, indexPlusY, indexMinusY))
    elif len(indexPlusY) == 0:
        indexArray = np.vstack((indexCenter, indexPlusX, indexMinusX))
    else:
        indexArray = np.vstack((indexCenter, indexPlusX, indexMinusX, indexPlusY, indexMinusY))

    indexArray = indexArray[np.abs(indexArray[:, 2]) > 1e-10, :]

    return indexArray


def SouthwellShift(slopeXIn, slopeYIn):
    """
    Shift coordinates by half-pixel to make surface map match slope map.

    INPUT:
        slopeXIn: 2D array representing the x-slope data to be integrated.
        slopeYIn: 2D array representing the y-slope data to be integrated.

    OUTPUT:
        slopeXOut: shifted slope such that after integration, the surface point
            will align with the original x-slope data.
        slopeYOut: same as slopeX, but for the y-slope data.
    """

    slopeXOut = slopeXIn.copy()
    slopeXOut[:, :-1] = (slopeXIn[:, 1:] + slopeXIn[:, :-1]) / 2
    slopeXOut[:, -1] = slopeXOut[:, -2]

    slopeYOut = slopeYIn.copy()
    slopeYOut[:-1, :] = (slopeYIn[1:, :] + slopeYIn[:-1, :]) / 2
    slopeYOut[-1, :] = slopeYOut[-2, :]

    return slopeXOut, slopeYOut

def trim_maps_to_square(regularSlopeX, regularSlopeY):

    N,M = regularSlopeX.shape
    min_size = np.min([N,M])
    start_N = (N-min_size)//2
    start_M = (M-min_size)//2

    regularSlopeX = regularSlopeX[start_N:start_N+min_size, start_M:start_M+min_size]
    regularSlopeY = regularSlopeY[start_N:start_N+min_size, start_M:start_M+min_size]

    return regularSlopeX, regularSlopeY

def SouthwellIntegration(slopeX, slopeY, type='Southwell'):
    """
    Perform Southwell least squares integration to reconstruct a surface from slope data.

    Parameters:
        slopeX (numpy.ndarray): 2D array of slopes in the X direction.
        slopeY (numpy.ndarray): 2D array of slopes in the Y direction.
        type (str): Type of integration kernel to use. Options: 'Southwell' (default) or 'GradientInverse'.

    Returns:
        numpy.ndarray: Reconstructed surface.
    """
    # Validate input
    if slopeX.shape != slopeY.shape:
        raise ValueError("slopeX and slopeY must have the same dimensions.")
    if slopeX.ndim != 2 or slopeY.ndim != 2:
        raise ValueError("slopeX and slopeY must be 2D arrays.")

    M, N = slopeX.shape  # Dimensions of the slope arrays

    # Create a mask for bad data (NaN values)
    badData = np.isnan(slopeX) | np.isnan(slopeY)
    slopeX[badData] = 0
    slopeY[badData] = 0

    # Type-specific pre-processing
    if type.lower() == 'southwell':
        slopeX, slopeY = SouthwellShifting(slopeX, slopeY)

    # Flatten the slope arrays into a single vector
    data = np.concatenate((slopeX.ravel(), slopeY.ravel()))

    # Create the sparse matrix for the integration
    sparseMatrix = CreateIntegrationMatrix(M, N, type)

    # Solve the linear system using least squares
    surface = lsqr(sparseMatrix, data)[0]

    # Reshape the surface into a 2D array
    surface = surface.reshape(M, N)

    # Re-apply the mask for bad data
    surface[badData] = np.nan

    # Remove the mean value (arbitrary constant of integration)
    surface = surface - np.nanmean(surface)

    return surface


def CreateIntegrationMatrix(M, N, type):
    """
    Create the sparse matrix for Southwell least squares integration.

    Parameters:
        M (int): Number of rows in the slope arrays.
        N (int): Number of columns in the slope arrays.
        type (str): Type of integration kernel to use.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix for integration.
    """
    # Initialize lists to store row indices, column indices, and values
    row_indices = []
    col_indices = []
    values = []

    # Define the kernel based on the type
    if type.lower() == 'southwell':
        kernelX = [0, -1, 1]  # Southwell kernel for X direction
        kernelY = [0, -1, 1]  # Southwell kernel for Y direction
    elif type.lower() == 'gradientinverse':
        kernelX = [-0.5, 0, 0.5]  # Gradient inverse kernel for X direction
        kernelY = [-0.5, 0, 0.5]  # Gradient inverse kernel for Y direction
    else:
        raise ValueError("Invalid type. Use 'Southwell' or 'GradientInverse'.")

    # Create the sparse matrix for X slopes
    for i in range(M):
        for j in range(N):
            if j < N - 1:  # Right neighbor
                row_indices.append(i * N + j)
                col_indices.append(i * N + j)
                values.append(kernelX[1])
                row_indices.append(i * N + j)
                col_indices.append(i * N + (j + 1))
                values.append(kernelX[2])

    # Create the sparse matrix for Y slopes
    for i in range(M):
        for j in range(N):
            if i < M - 1:  # Bottom neighbor
                row_indices.append(M * N + i * N + j)
                col_indices.append(i * N + j)
                values.append(kernelY[1])
                row_indices.append(M * N + i * N + j)
                col_indices.append((i + 1) * N + j)
                values.append(kernelY[2])

    # Create the sparse matrix
    sparseMatrix = csr_matrix((values, (row_indices, col_indices)), shape=(2 * M * N, M * N))

    return sparseMatrix


def SouthwellShifting(slopeX, slopeY):
    """
    Shift the slope arrays by half a pixel for Southwell integration.

    Parameters:
        slopeX (numpy.ndarray): 2D array of slopes in the X direction.
        slopeY (numpy.ndarray): 2D array of slopes in the Y direction.

    Returns:
        tuple: Shifted slopeX and slopeY arrays.
    """
    slopeX_shifted = np.zeros_like(slopeX)
    slopeX_shifted[:, :-1] = (slopeX[:, 1:] + slopeX[:, :-1]) / 2
    slopeX_shifted[:, -1] = slopeX_shifted[:, -2]

    slopeY_shifted = np.zeros_like(slopeY)
    slopeY_shifted[:-1, :] = (slopeY[1:, :] + slopeY[:-1, :]) / 2
    slopeY_shifted[-1, :] = slopeY_shifted[-2, :]

    return slopeX_shifted, slopeY_shifted

def prepare_image(file_name, rotation = None, xyr = None, output_plots = False):

    if file_name.endswith('.bmp'):
        # Read the image
        image_color = cv2.imread(file_name)
        # Check if the image was loaded successfully
        if image_color is None:
            raise FileNotFoundError(f"Image file not found at {file_name}")

        # Convert the image to grayscale
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    elif file_name.endswith('.fits') or file_name.endswith('.fit'):
        hdul = fits.open(file_name)
        image_gray = hdul[0].data

    # Crop the image using the CropImage function
    im_crop = crop_image(image_gray, xyr)

    # Optional: Display the cropped image
    if output_plots:
        plt.imshow(im_crop)
        plt.title('Cropped image')
        plt.show()

    if rotation:
        image = rotate(im_crop, rotation, reshape=False)
    else:
        image, rotation = improve_image_rotation(im_crop)

    return image, rotation

def define_reference(folder_path, xyr=None, output_plots = False):
    refX_holder = []
    refY_holder = []
    magnification_holder = []
    nominalSpot_holder = []
    rotation_holder = []
    for file in os.listdir(folder_path):
        if file.endswith('.bmp') or file.endswith('.fits') or file.endswith('.fit'):
            file_path = os.path.join(folder_path, file)
            image, rotation = prepare_image(file_path,xyr=xyr,output_plots=output_plots)
            xyr_cropped = [image.shape[0]/2,image.shape[1]/2,xyr[2]]
            referenceX, referenceY, magnification, nominalSpot = GetGrid(image, xyr_cropped, output_plots)
            refX_holder.append(referenceX)
            refY_holder.append(referenceY)
            magnification_holder.append(magnification)
            nominalSpot_holder.append(nominalSpot)
            rotation_holder.append(rotation)

    return np.mean(refX_holder), np.mean(refY_holder), np.mean(magnification_holder), np.mean(nominalSpot_holder,0), np.mean(rotation_holder)

def jupiter_pupil_merit_function(xyr, thresh_image, inside_pupil_weight=2, outside_pupil_weight = 1):
    #The premise of this optimizer is to define a circle that contains as many "good" pixels (containing starlight)
    #as possible while minimizing as many "bad" pixels that do not contain starlight
    negative_image = np.subtract(thresh_image.astype(float), np.max(thresh_image.astype(float)))*-1
    X,Y = np.meshgrid(np.arange(thresh_image.shape[0]),np.arange(thresh_image.shape[1]))
    proposed_pupil = np.sqrt(np.square(X-xyr[0]) + np.square(Y-xyr[1])) < xyr[2]
    spots_inside_pupil = np.sum(thresh_image[proposed_pupil]).astype(float)
    spots_outside_pupil = np.sum(thresh_image[~proposed_pupil]).astype(float)
    spaces_inside_pupil = np.sum(negative_image[proposed_pupil]).astype(float)
    spaces_outside_pupil = np.sum(negative_image[~proposed_pupil]).astype(float)

    good_pupil = spots_inside_pupil**inside_pupil_weight + spaces_outside_pupil
    bad_pupil = spots_outside_pupil**outside_pupil_weight + spaces_inside_pupil

    merit = (bad_pupil - good_pupil)/(np.sum(thresh_image) + np.sum(negative_image))

    if False:
        #Plot the image to check how the optimizer is doing
        fig,ax = plt.subplots()
        ax.imshow(thresh_image)
        artist = Circle(xyr[:2],xyr[-1],fill=False,color='r')
        ax.add_artist(artist)
        fig.suptitle(np.round(xyr[-1],3))
        plt.show()
    elif False:
        print(np.round(merit,3))

    return merit

def average_folder_of_images(path):
    image_holder = []
    for file in os.listdir(path):
        if file.lower().endswith(".jpg") or file.lower().endswith(".png") or file.lower().endswith(".bmp"):
            image_holder.append(io.imread(path + file))
        elif file.lower().endswith(".fits") or file.lower().endswith(".fit"):
            hdul = fits.open(path+file)
            image_holder.append(hdul[0].data)
    image = np.mean(image_holder,0)
    return image

def define_pupil_from_extended_object(image,thresh=127):
    #Name is a misnomer: originally used for extended objects but revised for star images
    thresh_image = cv2.threshold(image, thresh, 1, cv2.THRESH_BINARY)[1]
    
    #Starting value for optimizer: centerpoint of image
    xyr = [int(thresh_image.shape[0]/2), int(thresh_image.shape[1]/2), int(np.max(thresh_image.shape)/4)]
    res = minimize(jupiter_pupil_merit_function, xyr, args=thresh_image, method='Nelder-Mead', maxiter=500)
    return res.x

def define_xyr_for_boolean_pupil(pupil):
    X,Y = np.meshgrid(np.arange(pupil.shape[1]),np.arange(pupil.shape[0]))
    cent_x = np.sum(X * pupil) / np.sum(pupil)
    cent_y = np.sum(Y * pupil) / np.sum(pupil)
    radius = (len(np.where(pupil[int(cent_y),:])[0]) + len(np.where(pupil[:,int(cent_x)])[0])) / 4
    return [cent_x, cent_y, radius]

def choose_how_much_to_crop(pupil):
    # crop out columns / rows of nans surrounding the edge of the pupil
    left = 0
    while np.all(np.isnan(pupil[left, :])):
        left = left + 1
    right = pupil.shape[0]-1
    while np.all(np.isnan(pupil[right,:])):
        right = right - 1
    top = 0
    while np.all(np.isnan(pupil[:,top])):
        top = top + 1
    bottom = pupil.shape[0]-1
    while np.all(np.isnan(pupil[:,bottom])):
        bottom = bottom - 1

    return left, right+1, top, bottom+1

def griddata_interpolater(starting_im, ending_im,clear_aperture_outer, clear_aperture_inner):
    left, right, top, bottom = choose_how_much_to_crop(starting_im)
    begin_im = starting_im[left:right, top:bottom]
    left, right, top, bottom = choose_how_much_to_crop(ending_im)
    finish_im = ending_im[left:right, top:bottom]
    x,y = np.meshgrid(np.linspace(-clear_aperture_outer,clear_aperture_outer,begin_im.shape[1]), np.linspace(-clear_aperture_outer,clear_aperture_outer,begin_im.shape[0]))
    X,Y = np.meshgrid(np.linspace(-clear_aperture_outer,clear_aperture_outer,finish_im.shape[1]), np.linspace(-clear_aperture_outer,clear_aperture_outer,finish_im.shape[0]))
    begin_valid = ~np.isnan(begin_im)
    grid_z = griddata((x[begin_valid],y[begin_valid]),begin_im[begin_valid],(X,Y),method='cubic')
    center_coordinate = grid_z.shape[0]/2
    distance_from_center = np.sqrt(np.square(X) + np.square(Y))
    end_OD = distance_from_center < clear_aperture_outer
    end_ID = distance_from_center > clear_aperture_inner
    end_valid = end_OD * end_ID
    grid_z[~end_valid] = np.nan

    req_pad = ending_im.shape[0]-grid_z.shape[0]
    grid_pad = np.pad(grid_z,int(req_pad/2),mode='constant',constant_values=np.nan)
    return grid_pad