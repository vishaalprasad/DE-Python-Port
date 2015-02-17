from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import os
import radialProfile

from pylearn2.utils import serial

from de.datasets import VanHateren


def fft2(image):
    freq = fftpack.fft2(image)
    shifted = fftpack.fftshift(freq)
    return np.abs(shifted)


def fft2AverageOnImageSet(images):
    # Requires images to be of the same shape.
    total_frequency = np.zeros((images[0].shape))
    for image in images:
        total_frequency = np.add(total_frequency, fft2(image))
    return total_frequency / float(len(images))


# A function that performs an fft analysis of an image and its reconstruction
# and plots the analyses for purposes of visualization.
def singleImageAnalysis(model_path):

    print("Loading the training set...")
    train_img = os.path.join(VanHateren.DATA_DIR, 'train.pkl')
    train_set = serial.load(train_img)
    patch_size = (32, 32)

    # Run the model and visualize
    print("Loading the model...")
    model = serial.load(model_path)

    list_of_images = []
    list_of_reconstructed = []

    print("Beginning the fft analysis...")

    for ii in np.arange(train_set.X.shape[0]):
        img_vector = train_set.denormalize_image(train_set.X[ii, :])
        img_patch = img_vector.reshape(patch_size)
        list_of_images.append(img_patch)

        # Part for reconstructed
        [tensor_var] = model.reconstruct([img_vector])

        reconstructed_vector = tensor_var.eval()
        reconstructed_patch = train_set.denormalize_image(reconstructed_vector)
        reconstructed_patch = reconstructed_patch.reshape(patch_size)

        list_of_reconstructed.append(reconstructed_patch)

    average_frequency = fft2AverageOnImageSet(list_of_images)
    average_reconstructed = fft2AverageOnImageSet(list_of_reconstructed)

    # Show the 2D Power Analysis
    fh = plt.figure()
    fh.add_subplot(1, 2, 1)
    plt.imshow(np.log(average_frequency))
    plt.title('Original Images')

    fh.add_subplot(1, 2, 2)
    plt.imshow(np.log(average_reconstructed))
    plt.title('Reconstructed Images')

    plt.savefig('fft2D.png')
    os.system('eog fft2D.png &')

    psd1D = radialProfile.azimuthalAverage(average_frequency)
    reconstructed_psd1D = radialProfile.azimuthalAverage(average_reconstructed)

    fg = plt.figure()
    fg.add_subplot(1, 2, 1)
    plt.plot(np.log(psd1D))
    plt.title('Original Images')

    fg.add_subplot(1, 2, 2)
    plt.plot(np.log(reconstructed_psd1D))
    plt.title('Reconstructed Images')

    plt.savefig('fft1D.png')
    os.system('eog fft1D.png &')


# A function that helps visualize the differences between each
# hemispherical representation of a set of images.
def hemisphericalDifferences(left_model_path, right_model_path):
    print("Loading the training set...")
    train_img = os.path.join(VanHateren.DATA_DIR, 'train.pkl')
    train_set = serial.load(train_img)
    patch_size = (32, 32)

    # Load both models
    print("Loading the models...")
    left_model = serial.load(left_model_path)
    right_model = serial.load(right_model_path)

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
    average_frequency = fft2AverageOnImageSet(list_of_images)
    average_left_reconstructed = fft2AverageOnImageSet(
        list_of_left_reconstructed
        )
    average_right_reconstructed = fft2AverageOnImageSet(
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


if __name__ == '__main__':
    # singleImageAnalysis('savedata.pkl')
    hemisphericalDifferences('left.pkl', 'right.pkl')
