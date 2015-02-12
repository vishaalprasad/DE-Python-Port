import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pylearn2.utils import serial

from vanhateren import DATA_DIR, VANHATEREN


def compare_reconstruction(model_path='savedata.pkl', img_file_path=None,
                           img_idx=None, plt_out=None):
    patch_size = (32, 32)

    print("Loading the training set...")
    train_img = os.path.join(DATA_DIR, 'train.pkl')
    train_set = serial.load(train_img)

    # Grab an image patch
    if img_idx is None:
        img_idx = 4

    print("Grabbing an image patch from the training set...")
    img_vector = train_set.denormalize_image(train_set.X[img_idx, :])
    img_patch = img_vector.reshape(patch_size)

    # Show the patch
    fh = plt.figure()
    fh.add_subplot(1, 2, 1)
    plt.imshow(img_patch, cmap=cm.Greys_r)
    plt.axis('off')
    plt.title('Original image')

    # Run the model and visualize
    print("Running the model...")
    model = serial.load(model_path)
    [tensor_var] = model.reconstruct([img_vector])

    # Show the reconstructed patch
    print("Plotting...")
    fh.add_subplot(1, 2, 2)

    reconstructed_vector = tensor_var.eval()
    reconstructed_patch = train_set.denormalize_image(reconstructed_vector)
    reconstructed_patch = reconstructed_patch.reshape(patch_size)
    plt.imshow(reconstructed_patch, cmap=cm.Greys_r)
    plt.axis('off')
    plt.title('Reconstructed image')

    # Display both to screen
    if plt_out is None:
        plt.show()
    else:
        plt.savefig(plt_out)
        os.system('eog "%s.png" &' % plt_out)


if __name__ == '__main__':
    import sys
    compare_reconstruction(model_path='savedata.pkl')
