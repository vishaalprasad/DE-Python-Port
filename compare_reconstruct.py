import array
import glob
import numpy
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pylearn2.utils import serial

from vanhateren import DATA_DIR, VH_WIDTH, VH_HEIGHT
from vanhateren import read_iml, get_patch


width = VH_WIDTH
height = VH_HEIGHT
patch_size = (32, 32)

# Grab an image patch
all_files = glob.glob(os.path.join(DATA_DIR, '*.iml'))
file_path = all_files[0]
img = read_iml(file_path, width=width, height=height)
img_patch = get_patch(img, patch_size=patch_size)

# Show the patch
fh = plt.figure()
fh.add_subplot(1, 2, 1)
plt.imshow(img_patch, cmap=cm.Greys_r)

# Run the model and visualize
model = serial.load('savedata.pkl')
[tensor_var] = model.reconstruct([img_patch.flatten()])

# Show the reconstructed patch
fh.add_subplot(1, 2, 2)
plt.imshow(tensor_var.eval().reshape(patch_size), cmap=cm.Greys_r)

# Display both to screen
#plt.savefig('model_performance.png')
#os.system('eog model_performance.png &')
plt.show()
