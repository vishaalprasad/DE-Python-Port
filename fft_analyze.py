from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import os
import radialProfile

from pylearn2.utils import serial

from vanhateren import DATA_DIR, VANHATEREN


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


if __name__ == '__main__':

    print("Loading the training set...")
    train_img = os.path.join(DATA_DIR, 'train.pkl')
    train_set = serial.load(train_img)
    patch_size = (32, 32)

    # Run the model and visualize
    print("Loading the model...")
    model_path = 'savedata.pkl'
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
