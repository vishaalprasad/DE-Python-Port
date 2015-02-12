import functools

import numpy as np

from pylearn2.models.autoencoder import DenoisingAutoencoder
from pylearn2.models.model import Model


class SparseRFAutoencoder(DenoisingAutoencoder):
    """
    A denoising autoencoder with sparse local receptive fields.
    """

    def __init__(self, nhid, numCons, sigma, imageSize, **kwargs):
        """
        Parameters:
        ----------

        nhid: number of hidden units

        numCons: number of connections per hidden unit

        sigma: standard deviation matrix

        imageSize = size of an image

        """
        super(SparseRFAutoencoder, self).__init__(nhid=nhid, **kwargs)
        self.numCons = numCons
        self.sigma = sigma
        self.imageSize = np.array(imageSize)

        self.hiddenUnitLocs = self._set_hidden_unit_locations()
        self.mask = self._create_connection_mask()

    def __str__(self):
        props_to_print = dict([(prop_name, getattr(self, prop_name))
                               for prop_name in ['nhid', 'numCons', 'sigma']])

        return "%s(%s)" % (self.__class__.__name__, props_to_print)

    def _set_hidden_unit_locations(self):
        """
        This resizes the image to an image with as many
        hidden units as there are pixels to place.
        Then the code expands the reduced image back out to
        compute the hidden unit locations.

        sets self.hiddenUnitLocs to [nhidden x 2]
        """
        imgHeight = self.imageSize[0]
        imgLength = self.imageSize[1]
        numPixels = imgHeight * imgLength

        # Determine the scaling

        scalefactor = numPixels/self.nhid
        assert scalefactor >= 1., \
            'nhid or hidden_units_per_layer is off;' \
            '%d units requested in %d locations/pixels!' % \
            (self.nhid, numPixels)

        newgrid = np.array(
            np.round(self.imageSize / np.sqrt(scalefactor)),
            dtype=int)
        assert np.prod(newgrid) == self.nhid, \
            "can't fit; npixels / nhid must be a square (4, 9, 16, etc.)"

        # Set up the hidden unit positions in the smaller grid
        [X, Y] = np.meshgrid(range(newgrid[1]), range(newgrid[0]))
        X = X * np.sqrt(scalefactor)
        Y = Y * np.sqrt(scalefactor)

        # Expand the smaller grid back out to the larger grid
        X = X + (np.sqrt(scalefactor)) / 2 - 0.5
        Y = Y + (np.sqrt(scalefactor)) / 2 - 0.5

        # Turn into rounded column vectors
        X = np.round(X).astype(int).flatten(1)
        Y = np.round(Y).astype(int).flatten(1)

        # Create the outputs from the grids
        connection_matrix = np.zeros(self.imageSize, dtype=bool)
        connection_matrix[Y, X] = True
        if False:  # debug plotting code
            import matplotlib.pyplot as plt
            plt.imshow(connection_matrix)
            plt.title('connection matrix')
            plt.show()
        assert np.count_nonzero(connection_matrix) == self.nhid, \
            '# of requested locations must match the # of provided locations!'

        return np.asarray(np.nonzero(connection_matrix)).T

    def _create_connection_mask(self):
        # Define some useful local variables for sake of clarity
        imgHeight = self.imageSize[0]
        imgLength = self.imageSize[1]
        numPixels = imgHeight * imgLength
        numHiddenUnits = self.hiddenUnitLocs.shape[0]

        # Initialize the Connection Matrix to all zeroes
        connectionMatrix = np.zeros(shape=(numPixels, numHiddenUnits))
        currHiddenUnit = 0  # index to keep track of which hidden unit

        # Create a variance parameter by squaring each element in sigma,
        # used in Gaussian
        variance = [[elem * elem for elem in inner] for inner in self.sigma]

        # Loop through the Hidden Units to Create Samples
        for k in self.hiddenUnitLocs:

            i = 0
            while i < self.numCons:

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

                if (connectionMatrix[pixelLoc][currHiddenUnit] == 1):
                    continue

                connectionMatrix[pixelLoc][currHiddenUnit] = 1
                i += 1

            currHiddenUnit += 1

        return connectionMatrix

    @functools.wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        W = self.weights
        if W in updates:
            updates[W] = updates[W] * self.mask
        return super(SparseRFAutoencoder, self)._modify_updates(updates)

if __name__ == "__main__":

    # Create the dataset
    from vanhateren import VANHATEREN
    VANHATEREN.create_datasets()

    # Train the network.
    from pylearn2.scripts.train import train
    train(config="custom.yaml")

    # Visualize the weights
    from pylearn2.scripts.show_weights import show_weights
    show_weights(model_path="savedata.pkl", border=True)

    # Visualize the reconstruction
    from compare_reconstruct import compare_reconstruction
    compare_reconstruction(model_path="savedata.pkl")
