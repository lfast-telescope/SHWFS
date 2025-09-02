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
import random

pwd = os.getcwd()
sys.path.extend([pwd.split('SHWFS')[0] + 'primary_mirror'])
from LFAST_wavefront_utils import *
from LFAST_TEC_output import *

base_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/'
sh_path = base_path + 'on-sky/20250604/SHWFS/'  # path pointing to folder of SH images for current night
reference_path = os.path.join(sh_path, '002045/')  # path to set of images used for pupil definition
folder_path = reference_path
results_path = sh_path + 'variance_results/'
averaged_image_path = sh_path + 'averaged_images/'
averaged_lenslet_image_results_path = sh_path + 'averaged_results/'
avg_cropped_reconstructions_path = sh_path + 'avg_cropped_reconstructions/'

redefine_pupil = False
output_plots = False
xyr, extend_image = xyr_pupil_definition(folder_path, sh_path, redefine_pupil=redefine_pupil)  # Define location of pupil within SH image
referenceX, referenceY, magnification, nominalSpot, rotation = lenslet_definition(folder_path, sh_path, xyr, output_plots, recompute_rotation=redefine_pupil)

percentage_crop_test = 1 / np.sqrt(2)
#%%

list_of_tests = ['avg_reconstructions','avg_lenslets','crop_reconstructions','crop_lenslets']
chosen_test = list_of_tests[2]

if chosen_test == 'avg_lenslets':
    mean_image = average_folder_of_images(folder_path)
    np.save(averaged_image_path + 'mean_image.npy', mean_image)
    mean_surface = compute_wavefront_with_averaged_lenslet_images(averaged_image_path, 'mean_image.npy', sh_path, redefine_pupil, output_plots)
    np.save(averaged_lenslet_image_results_path + 'mean_lenslet_wf.npy', mean_surface)
else:
    mean_surface = full_SHWFS_reconstruction(sh_path, reference_path, redefine_pupil=False)

#%% Evaluate the effect of averaging different reconstructions
Z = General_zernike_matrix(44, int(15 * 25.4 * 1e3), int(3 * 25.4 * 1e3), mean_surface.shape[0])

number_trials = 30
list_of_files = os.listdir(reference_path)
fits_files = [x for x in list_of_files if x.endswith('.fits')]

for group_avg_size in np.arange(1,61):
    rms_holder = []
    difference_holder = []
    group_half_size = group_avg_size // 2
    starting_candidates = fits_files[group_half_size:-group_half_size-1]

    for test_number in np.arange(number_trials):
        print('Now running test number {}'.format(test_number) + ' for group size {}'.format(group_avg_size))
        file_choice = random.choice(starting_candidates)
        index_choice = fits_files.index(file_choice)

        if group_avg_size % 2 == 1:
            subset_of_fits_files = fits_files[index_choice - group_half_size:index_choice + group_half_size+1]
        else:
            subset_of_fits_files = fits_files[index_choice - group_half_size:index_choice + group_half_size]

        if chosen_test == 'avg_lenslets':
            subset_image = average_folder_of_images(folder_path, subset_of_fits_files)
            averaged_file_name = 'subset_image.npy'
            np.save(averaged_image_path + averaged_file_name, subset_image)
            subset_mean_surface = compute_wavefront_with_averaged_lenslet_images(averaged_image_path, averaged_file_name, sh_path, redefine_pupil, output_plots)
        else:
            subset_mean_surface = full_SHWFS_reconstruction(sh_path, reference_path, redefine_pupil=False, subset_list = subset_of_fits_files)

        difference = subset_mean_surface - mean_surface

        M,C = get_M_and_C(difference, Z)

        updated_difference = remove_modes(M, C, Z, [0, 1, 2, 4])

        if chosen_test == 'crop_reconstructions':
            X, Y = np.meshgrid(np.arange(updated_difference.shape[0]), np.arange(updated_difference.shape[1]))
            X = X - np.mean(X)
            Y = Y - np.mean(Y)
            crop_index = np.max(X) * percentage_crop_test
            proposed_pupil = np.sqrt(np.square(X) + np.square(Y)) < crop_index
            updated_difference[~proposed_pupil] = np.nan

        vals = updated_difference[~np.isnan(updated_difference)]
        rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals))
        difference_holder.append(updated_difference)
        rms_holder.append(rms)

    if chosen_test == 'avg_lenslets':
        np.save(averaged_lenslet_image_results_path + 'avg' + str(group_avg_size) + '_difference_holder.npy',difference_holder)
        np.save(averaged_lenslet_image_results_path + 'avg' + str(group_avg_size) + '_rms_holder.npy', rms_holder)
    elif chosen_test == 'avg_reconstructions':
        np.save(results_path + 'avg' + str(group_avg_size) + '_difference_holder.npy', difference_holder)
        np.save(results_path + 'avg' + str(group_avg_size) + '_rms_holder.npy', rms_holder)
    elif chosen_test == 'crop_reconstructions':
        np.save(avg_cropped_reconstructions_path + 'avg' + str(group_avg_size) + '_difference_holder.npy', difference_holder)
        np.save(avg_cropped_reconstructions_path + 'avg' + str(group_avg_size) + '_rms_holder.npy', rms_holder)

    mean = np.mean(rms_holder)
    stddev = np.std(rms_holder)
    print('Group average size {}'.format(group_avg_size) + ' has mean = ' + str(round(mean, 2)) + ' and stddev = ' + str(round(stddev, 2)))

#%%
label_list = ['Averaging reconstructed wavefronts', 'Averaging lenslet images', 'Averaging reconstructions and crop outer 30%']
fig, ax = plt.subplots()

for num, test_path in enumerate([results_path, averaged_lenslet_image_results_path, avg_cropped_reconstructions_path]):
    mean_holder = []
    stddev_holder = []
    group_size_holder = []
    for file in os.listdir(test_path):
        if file.endswith('rms_holder.npy'):
            group_avg_size = int(file.split('avg')[1].split('_')[0])
            rms_holder = np.load(test_path + file)
            mean = np.mean(rms_holder)
            stddev = np.std(rms_holder)
            mean_holder.append(mean)
            stddev_holder.append(stddev)
            group_size_holder.append(group_avg_size)

    sorted_group_index = np.argsort(group_size_holder)
    plot_rms = [mean_holder[i] for i in sorted_group_index]
    plot_std = [stddev_holder[i] for i in sorted_group_index]
    plot_group = [group_size_holder[i] for i in sorted_group_index]

    log_rms = np.log10(plot_rms)
    log_group = np.log10(plot_group)

    slope, intercept = np.polyfit(log_group, log_rms, 1)
    fitted_val = slope * log_group + intercept
    fitted_real = np.power(10, fitted_val)

    #plt.plot(group_size_holder,mean_holder)
    #plt.errorbar(plot_group,plot_rms, plot_std, label=label_list[num], capsize=1.5)
    p = ax.scatter(plot_group, plot_rms, label = label_list[num], marker='.')
    ax.plot(plot_group, fitted_real, color = p.get_facecolor())
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)

plt.xlabel('N: number maps averaged')
plt.ylabel('Rms error (nm)')
plt.legend()
plt.title('Estimate of residual noise in the average of N measurements')
plt.show()



#See how subaperture reconstruction affects these numbers in two ways: either by masking the lenslet spots beforehand, or else by shrinking the reconstructed pupil
