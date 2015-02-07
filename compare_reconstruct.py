from pylearn2.utils import serial
from pylearn2.models.model import Model
import numpy, array
import matplotlib.pyplot as plt
import matplotlib.cm as cm

filename = "/raid/vprasad/data/lisa/data/vanhateren/imk03739.iml" #arbitrary

with open(filename, 'rb') as handle:
        s = handle.read()
arr = array.array('H', s)
arr.byteswap()
img = numpy.array(arr, dtype='uint16').reshape(1024, 1536)
width = 1536
height = 1024
left_margin = (width-32)/2
top_margin = (height-32)/2
crop_img = img[top_margin: top_margin+32, left_margin:left_margin+32]
plt.imshow(crop_img, cmap = cm.Greys_r)
plt.savefig('original.png')
model = serial.load('savedata.pkl')
[tensor_var] = model.reconstruct([crop_img.flatten()])
plt.imshow(tensor_var.eval().reshape((32,32)), cmap=cm.Greys_r)
plt.savefig('reconstruct.png')
import os
os.system('eog original.png &')
os.system('eog reconstruct.png &')

