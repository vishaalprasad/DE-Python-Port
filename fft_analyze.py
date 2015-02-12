from scipy import fftpack
import numpy as np
import array
import matplotlib.pyplot as plt
import glob
import os
import radialProfile

from vanhateren import DATA_DIR, VH_WIDTH, VH_HEIGHT
from vanhateren import read_iml, get_patch

def fft2 (image):
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
    all_files = glob.glob(os.path.join(DATA_DIR, '*.iml'))
    list_of_images = []
    for a_file in all_files:
        list_of_images.append(read_iml(a_file))
    average_frequency = fft2AverageOnImageSet(list_of_images)

    plt.imshow(np.log(average_frequency))
    plt.savefig('fft2.png')
    os.system('eog fft2.png &')

    psd1D = radialProfile.azimuthalAverage(average_frequency)
