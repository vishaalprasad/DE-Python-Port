import array
import glob
import os

import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial, string_utils


DATA_DIR = string_utils.preprocess('${PYLEARN2_DATA_PATH}/vanhateren')
ALL_DATASETS = ['train', 'test', 'valid']
VH_WIDTH = 1536
VH_HEIGHT = 1024


def read_iml(filename, width=VH_WIDTH, height=VH_HEIGHT, dtype='uint16'):
    """Reads an IML file and returns as an ndarray."""

    with open(filename, 'rb') as handle:
        s = handle.read()
    arr = array.array('H', s)
    arr.byteswap()

    if width and height:
        arr = np.array(arr, dtype=dtype).reshape(height, width)

    return arr


def get_patch(img, patch_size=(32, 32), width_slice=None, height_slice=None):
    width = img.shape[1]
    height = img.shape[0]
    left_margin = (width-patch_size[1])/2
    top_margin = (height-patch_size[0])/2
    img_patch = img[top_margin: top_margin+patch_size[0],
                    left_margin:left_margin+patch_size[1]].flatten()
    return img_patch.reshape(patch_size)


class VANHATEREN(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, axes=('b', 0, 1, 'c'),
                 img_shape=[32, 32], img_dir=None, ntrain=200,
                 ntest=50, nvalid=0):

        assert which_set in ALL_DATASETS, \
            "Set specified is not a valid set. Please use 'train' or " \
            "'test' or 'valid'"

        self.axes = axes
        self.img_shape = img_shape
        self.img_dir = img_dir or DATA_DIR

        # We define here:
        self.img_size = np.prod(self.img_shape)

        # Get files
        nimages = ntrain + ntest + nvalid
        images = glob.glob(os.path.join(self.img_dir, '*.iml'))
        if len(images) < nimages:
            raise Exception("%d images needed for dataset; %d found in %s" % (
                nimages,
                len(images),
                self.img_dir))

        trainX = np.empty(shape=(ntrain, self.img_size))
        testX = np.empty(shape=(ntest, self.img_size))
        validX = np.empty(shape=(nvalid, self.img_size))

        # Take 250 images, convert to 32x32, store in X
        width = VH_WIDTH
        height = VH_HEIGHT
        for ii, image_file in enumerate(images[:(ntrain + ntest + nvalid)]):
            img = read_iml(image_file, width=width, height=height)
            img_patch = get_patch(img, patch_size=(32, 32))
            if ii < ntrain:
                trainX[ii] = img_patch
            elif ii < (ntrain + ntest):
                testX[ii-ntrain] = img_patch
            elif ii:
                validX[ii-ntrain-ntest] = img_patch

        # Post-processing
        validMax = validX.max() if validX.shape[0] else 0
        trainMax = trainX.max() if trainX.shape[0] else 0
        testMax = testX.max() if testX.shape[0] else 0

        img_max = np.max([validMax, trainMax, testMax])
        trainX = trainX / float(img_max)
        testX = testX / float(img_max)
        validX = validX / float(img_max)

        view_converter = dense_design_matrix.DefaultViewConverter(
            (32, 32, 1),
            axes)
        if which_set == 'train':
            X = trainX
        elif which_set == 'test':
            X = testX
        elif which_set == 'valid':
            X = validX

        super(VANHATEREN, self).__init__(
            X=X,
            view_converter=view_converter,
            axes=axes)

    @classmethod
    def create_datasets(cls, datasets=ALL_DATASETS, overwrite=False,
                        img_dir=DATA_DIR, output_dir=DATA_DIR):
        """Creates the requested datasets, and writes them to disk.
        """
        serial.mkdir(output_dir)

        for dataset_name in list(datasets):
            file_path_fn = lambda ext: os.path.join(
                output_dir,
                '%s.%s' % (dataset_name, ext))

            output_files = dict([(ext, file_path_fn(ext))
                                 for ext in ['pkl', 'npy']])
            files_missing = np.any([not os.path.isfile(f)
                                    for f in output_files.values()])

            if overwrite or np.any(files_missing):
                print("Loading the %s data" % dataset_name)
                dataset = cls(which_set=dataset_name, img_dir=img_dir)

                print("Saving the %s data" % dataset_name)
                dataset.use_design_loc(output_files['npy'])
                serial.save(output_files['pkl'], dataset)


if __name__ == "__main__":
    VANHATEREN.create_datasets(overwrite=True)
