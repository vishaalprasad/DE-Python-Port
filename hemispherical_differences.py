import numpy as np
import matplotlib.pyplot as plt
import os

from pylearn2.utils import serial

import fft_analyze as fft
import radialProfile
from de.datasets import VanHateren
DATA_DIR = VanHateren.DATA_DIR

if __name__ == '__main__':

    print("Loading the training set...")
    train_img = os.path.join(DATA_DIR, 'train.pkl')
    train_set = serial.load(train_img)
    patch_size = (32, 32)

    # Load both models
    print("Loading the model...")
    model_path = 'left.pkl'
    left_model = serial.load(model_path)

    model_path = 'right.pkl'
    right_model = serial.load(model_path)

    # Create empty arrays to hold images
    list_of_images = []
    list_of_left_reconstructed = []
    list_of_right_reconstructed = []

    print("Beginning the fft analysis...")

    # Go through all the images and add them to the lists
    for ii in np.arange(train_set.X.shape[0]):
        img_vector = train_set.denormalize_image(train_set.X[ii, :])
        img_patch = img_vector.reshape(patch_size)
        list_of_images.append(img_patch)

        # Part for reconstructed
        [tensor_var] = left_model.reconstruct([img_vector])

        reconstructed_vector = tensor_var.eval()
        reconstructed_patch = train_set.denormalize_image(reconstructed_vector)
        reconstructed_patch = reconstructed_patch.reshape(patch_size)

        list_of_left_reconstructed.append(reconstructed_patch)

        [tensor_var] = right_model.reconstruct([img_vector])

        reconstructed_vector = tensor_var.eval()
        reconstructed_patch = train_set.denormalize_image(reconstructed_vector)
        reconstructed_patch = reconstructed_patch.reshape(patch_size)

        list_of_right_reconstructed.append(reconstructed_patch)

    # Run 2D Analysis
    average_frequency = fft.fft2AverageOnImageSet(list_of_images)
    average_left_reconstructed = fft.fft2AverageOnImageSet(
        list_of_left_reconstructed
        )
    average_right_reconstructed = fft.fft2AverageOnImageSet(
        list_of_right_reconstructed
        )

    # Run 1D Analysis
    psd1D = radialProfile.azimuthalAverage(average_frequency)
    reconstructed_left_psd1D = radialProfile.azimuthalAverage(
        average_left_reconstructed
        )
    reconstructed_right_psd1D = radialProfile.azimuthalAverage(
        average_right_reconstructed
        )

    # Plot 3 1D images: The Original, the reconstructions from LH and RH
    fg = plt.figure()
    fg.add_subplot(1, 3, 1)
    plt.plot(np.log(psd1D))
    plt.title('Original')

    fg.add_subplot(1, 3, 2)
    plt.plot(np.log(reconstructed_left_psd1D))
    plt.title('LH Reconstructed')

    fg.add_subplot(1, 3, 3)
    plt.plot(np.log(reconstructed_right_psd1D))
    plt.title('RH Reconstructed')

    plt.savefig('power.png')
    os.system('eog power.png &')

    # Get difference of differences
    left_difference = abs(reconstructed_left_psd1D - psd1D)
    right_difference = abs(reconstructed_left_psd1D - psd1D)
    total_difference = abs(left_difference-right_difference)
    plt.figure()
    plt.plot(total_difference)
    plt.title('Difference in power differences')

    plt.savefig('differences.png')
    os.system('eog differences.png &')
