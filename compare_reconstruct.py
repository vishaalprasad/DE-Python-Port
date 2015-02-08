import array
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pylearn2.utils import serial, string_utils

from vanhateren import DATA_DIR, VH_WIDTH, VH_HEIGHT
from vanhateren import read_iml, get_patch


def compare_reconstruction(model_path='savedata.pkl', img_file_path=None, img_idx=None, plt_out=None):
    width = VH_WIDTH
    height = VH_HEIGHT
    patch_size = (32, 32)

    # Grab an image patch
    if img_idx is not None:
        print("Grabbing an image patch from the training set...")
        train_img = os.path.join(DATA_DIR, 'train.pkl')
        train_set = serial.load(train_img)
        img_vector = train_set.X[img_idx, :]
        img_patch = img_vector.reshape(patch_size)

    else:
        # If no path specified, grab the second one found.
        if img_file_path is None:
            all_files = glob.glob(os.path.join(DATA_DIR, '*.iml'))
            img_file_path = all_files[1]
        print("Grabbing an image patch from %s..." % img_file_path)
        img = read_iml(img_file_path, width=width, height=height)
        img_patch = get_patch(img, patch_size=patch_size)
        img_vector = img_patch.ravel()

    # Show the patch
    fh = plt.figure()
    fh.add_subplot(1, 2, 1)
    plt.imshow(img_patch, cmap=cm.Greys_r)

    # Run the model and visualize
    print("Running the model...")
    model = serial.load(model_path)
    [tensor_var] = model.reconstruct([img_vector])

    # Show the reconstructed patch
    print("Plotting...")
    fh.add_subplot(1, 2, 2)
    plt.imshow(tensor_var.eval().reshape(patch_size), cmap=cm.Greys_r)

    # Display both to screen
    if plt_out is None:
        plt.show()
    else:
        plt.savefig(plt_out)
        os.system('eog "%s.png" &' % plt_out)


if __name__ == '__main__':
    compare_reconstruction(model_path='savedata.pkl')
