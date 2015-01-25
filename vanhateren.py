import array
import glob
import os

import numpy as np
from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import contains_nan
from pylearn2.utils import serial
from pylearn2.utils import string_utils
from theano.compat.six.moves import xrange


class VANHATEREN(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, axes=('b', 0, 1, 'c')):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        #(as it is here)
        self.axes = axes
        #we define here:
        dtype = 'uint16'
        ntrain = 200
        nvalid = 0  # artefact, we won't use it
        ntest = 50
        self.img_shape = (32, 32)
        self.img_size = np.prod(self.img_shape)
        #Get files
        files = '/home/vprasad/data/vanhateren/*.iml'
        images = glob.glob(files)

        trainX = np.empty(shape=(ntrain, self.img_size))
        testX = np.empty(shape=(ntest, self.img_size))

        i = 0
        #take 250 images, convert to 32x32, store in X
        for image in images:
            with open(image, 'rb') as handle:
                s = handle.read()
            arr = array.array('H', s)
            arr.byteswap()

            width = 1536
            height = 1024

            img = np.array(arr, dtype='uint16').reshape(height, width)
            left_margin = (width-32)/2
            top_margin = (height-32)/2
            if (i < ntrain):
                trainX[i] = img[top_margin: top_margin+32, left_margin:left_margin+32].flatten()
                
            else:
                testX[i-ntrain] = img[top_margin: top_margin+32, left_margin:left_margin+32].flatten()
            i = i + 1
        testMax = testX.max()
        trainMax = trainX.max()
        trainX = trainX / 1.
        trainX = trainX / max(testMax, trainMax)
        testX = testX / 1.
        testX = testX / max(testMax, trainMax)
        #call super on X

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 1), axes)
        if which_set == 'train':
            super(VANHATEREN, self).__init__(X=trainX,view_converter = view_converter, axes = axes)
        else: 
            super(VANHATEREN, self).__init__(X=testX, view_converter = view_converter, axes = axes)



