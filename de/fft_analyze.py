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
    import pdb; pdb.set_trace()
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
        img_vector = train_set.X[ii, :]

        # Part for reconstructed
        [tensor_var] = model.reconstruct([img_vector])
        reconstructed_vector = tensor_var.eval()

        # Save off human-viewable image patches
        img_patch = train_set.denormalize_image(img_vector)
        img_patch = img_patch.reshape(patch_size)
        list_of_images.append(img_patch)

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
def hemisphericalDifferences(left_model_path, right_model_path, plotting=None):
    """
    """
    print("Loading the training set...")
    train_img = os.path.join(VanHateren.DATA_DIR, 'train.pkl')
    train_set = serial.load(train_img)
    patch_size = (32, 32)

    # Load both models
    print("Loading the models...")
    left_model = serial.load(left_model_path)
    right_model = serial.load(right_model_path)

    # Create empty arrays to hold images
    image_patches = {
        'orig': [],
        left_model: [],
        right_model: [], }

    print("Beginning the fft analysis...")

    # Go through all the images and add them to the lists
    for ii in np.arange(train_set.X.shape[0]):
        img_vector = train_set.X[ii, :]
        for model in [left_model, right_model]:

            # Part for reconstructed
            [tensor_var] = model.reconstruct([img_vector])
            reconstructed_vector = tensor_var.eval()

            reconstructed_patch = train_set.denormalize_image(reconstructed_vector)
            reconstructed_patch = reconstructed_patch.reshape(patch_size)
            image_patches[model].append(reconstructed_patch)

        # Save off results
        img_patch = train_set.denormalize_image(img_vector)
        img_patch = img_patch.reshape(patch_size)
        image_patches['orig'].append(img_patch)


    # Run 2D Analysis
    average_frequency = fft2AverageOnImageSet(image_patches['orig'])
    average_left_reconstructed = fft2AverageOnImageSet(
        image_patches[left_model])
    average_right_reconstructed = fft2AverageOnImageSet(
        image_patches[right_model])

    # Run 1D Analysis
    psd1D = radialProfile.azimuthalAverage(average_frequency)
    reconstructed_left_psd1D = radialProfile.azimuthalAverage(
        average_left_reconstructed)
    reconstructed_right_psd1D = radialProfile.azimuthalAverage(
        average_right_reconstructed)


    # Get difference of differences
    left_difference = abs(reconstructed_left_psd1D - psd1D)
    right_difference = abs(reconstructed_right_psd1D - psd1D)
    total_difference = left_difference - right_difference   # RH better: > 0

    # Plot 3 1D images: The Original, the reconstructions from LH and RH
    if plotting:
        fig = plt.figure(figsize=(12, 6))
        fig.add_subplot(1, 2, 1)
        plt.plot(np.asarray([np.log(psd1D),
                             np.log(reconstructed_left_psd1D),
                             np.log(reconstructed_right_psd1D), ]).T)
        plt.legend(['Original',
                    'LH (\sigma=%.2f)' % np.asarray(left_model.sigma).max(),
                    'RH (\sigma=%.2f)' % np.asarray(right_model.sigma).max()])
        plt.xlabel('Spatial frequency')
        plt.ylabel('Power (log(amplitude))')

        fig.add_subplot(1, 2, 2)
        plt.plot(total_difference)
        plt.hold(True)
        plt.plot(np.ndarray(len(total_difference)),
                 np.zeros(total_difference.shape))  # show X-axis
        plt.title('Closeness in power differences (RH - LH)')
        plt.xlabel('Spatial frequency')
        plt.ylabel('Power difference')

        if isinstance(plotting, basestring):
            plt.savefig(plotting)
        else:
            plt.show()

    return total_difference


if __name__ == '__main__':
    # singleImageAnalysis('savedata.pkl')
    import sys
    plotting_param = sys.argv[1] if len(sys.argv) > 1 else True
    hemisphericalDifferences('left.pkl', 'right.pkl', plotting=plotting_param)
