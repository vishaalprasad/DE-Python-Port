import functools

import numpy as np

from pylearn2.models.autoencoder import DenoisingAutoencoder
from pylearn2.models.model import Model
from pylearn2.linear.matrixmul import MatrixMul



class CustomDenoisingAutoencoder(DenoisingAutoencoder):
    """
    A denoising autoencoder

    Parameters
    ----------
    See: DenoisingAutoencoder's parameters
    nhid = number of hidden units
    numCons = number of connections per hidden unit
    sigma = standard deviation matrix
    imageSize = size of an image
        """
    def __init__(self, nhid, numCons, sigma, imageSize, corruptor,
            act_enc, act_dec, nvis, tied_weights=False, irange=1e-3, rng=9001):

        super(CustomDenoisingAutoencoder, self).__init__(
                corruptor,
                nvis,
                nhid,
                act_enc,
                act_dec,
                tied_weights,
                irange,
                rng
        )
        hiddenUnitLocs = self._createHiddenUnits(nhid, imageSize)
        matrix = self._createConnectionMatrix(imageSize, hiddenUnitLocs, numCons, sigma)
        self.mask = matrix.transpose()


    def _createHiddenUnits(self, numHiddenUnits, imageSize):
        return np.array([[16, 16]])

    def _createConnectionMatrix(self, imageSize, hiddenUnitLocs, numConnections, sigma):

        ## Define some useful local variables for sake of clarity ##
        imgHeight = imageSize[0]
        imgLength = imageSize[1]
        numPixels = imgHeight * imgLength
        numHiddenUnits = len(hiddenUnitLocs)

        ## Initialize the Connection Matrix to all zeroes ##
        connectionMatrix = np.zeros(shape=(numHiddenUnits,numPixels))
        currHiddenUnit = 0 # index to keep track of which hidden unit we're working on.

        ## Create a variance parameter by squaring each element in sigma, used in Gaussian ##
        variance = [[elem * elem for elem in inner] for inner in sigma]

        ## Loop through the Hidden Units to Create Samples ##
        for k in hiddenUnitLocs:

            i = 0
            while i < numConnections:

                # Get random Gaussian sample which returns an array of tuples
                [[x, y]] = np.random.multivariate_normal(k, variance, 1)

                # Round the sample to nearest integer
                x = round(x, 0)
                y = round(y, 0)

                # Check to see if it's out of bounds.
                if (x >= imgLength) or (y >= imgHeight) or (x < 0) or (y < 0):
                    continue

                # Calculate which pixel number it is to add to the map.
                pixelLoc = (y) * imgLength + (x)

                if (connectionMatrix[currHiddenUnit][pixelLoc] == 1):
                    continue

                connectionMatrix[currHiddenUnit][pixelLoc] = 1
                i += 1

            currHiddenUnit += 1

        return connectionMatrix



    @functools.wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        W = self.weights
        if W in updates:
            updates[W] = updates[W] * self.mask


if __name__ == "__main__":
    from pylearn2.utils import string_utils
    data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/vanhateren')

    import os
    if os.path.isfile(data_dir+'/valid.pkl') is False \
        or os.path.isfile(data_dir+'/test.pkl') is False \
        or os.path.isfile(data_dir+'/train.pkl') is False:
        import dataset
        dataset.create_datasets()

    from pylearn2.scripts.train import train
    train(config="custom.yaml")


