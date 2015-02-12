from scipy import fftpack
import numpy as np
import array
import matplotlib.pyplot as plt
import glob
import os
import radialProfile

from pylearn2.utils import serial

from vanhateren import DATA_DIR, VH_WIDTH, VH_HEIGHT, VANHATEREN
from vanhateren import read_iml, get_patch

def fft2(image):
    freq = fftpack.fft2(image)
    shifted = fftpack.fftshift(freq)
    return np.abs(shifted)


def fft2AverageOnImageSet (images):
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

    list_of_images = []
    for ii in np.arange(train_set.X.shape[0]):
        img_vector = train_set.denormalize_image(train_set.X[ii, :])
        img_patch = img_vector.reshape(patch_size)
        list_of_images.append(img_patch)
    average_frequency = fft2AverageOnImageSet(list_of_images)

    plt.imshow(np.log(average_frequency))
    plt.savefig('fft2.png')
    os.system('eog fft2.png &')

    psd1D = radialProfile.azimuthalAverage(average_frequency)
