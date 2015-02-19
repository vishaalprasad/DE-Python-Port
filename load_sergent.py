import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('data_sergent.mat')

images = mat['train'][0][0][0]
imageSize = (25, 34)
height = imageSize[1]
width = imageSize[0]
reshaped_list = []

classifier_outputs = mat['train'][0][0][5]

classifier_labels = mat['train'][0][0][6]

image = images[0]


for image in images.T:
    reshaped = np.reshape(image, imageSize)
    reshaped_list.append(reshaped.T[1:height-1])


plt.imshow(reshaped_list[1])
plt.savefig('img.png')
import os
os.system('eog img.png &')

