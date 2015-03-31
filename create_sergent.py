import array
import os

import numpy as np
import scipy.io
from pylearn2.utils import serial

file_path = os.path.join(os.path.dirname(__file__),
        "data_sergent.mat")
which_set = "train"
mat = scipy.io.loadmat(file_path)
images = mat[which_set][0][0][0]
image_size = (25, 34)
patch_size = (32,32)
height = image_size[1]
width = image_size[0]
reshaped_list = []

classifier_outputs = mat[which_set][0][0][5]

classifier_labels = mat[which_set][0][0][6]
classifier_labels = np.asarray([lbl[0][0] for lbl in classifier_labels])
image = images[0]

for image in images.T:
    reshaped = np.reshape(image, image_size)
    reshaped = reshaped.T[1:height-1]
    square = np.zeros(patch_size)
    square[:,3:28] = reshaped
    reshaped_list.append(square.flatten())

serial.save("sergent.pkl", np.asarray(reshaped_list))
