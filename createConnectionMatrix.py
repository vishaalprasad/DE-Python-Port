import numpy as np

#==============================================================================#
#  Function Name: createConnectionMatrix                                       #
#                                                                              #
#  Description: Given its parameters, this function populates a connectivity   #
#  matrix which is a h x p matrix, where h is the number of hidden units and p #
#  is the number of pixels on the image. It creates ten unique connections for #
#  each hidden unit using the Gaussian function. Note that if the picture is   #
#  an m x n pixel map, then pixel[i][k] is represented in the connectivity     #
#  matrix by "i * imgWidth + k", i.e. view the pixels from left to right and   #
#  top to bottom.                                                              #
#                                                                              #
#  Parameters:  imageSize is a tuple with positive integers.                   #
#               hiddenUnitLocs is an n x 2 numpy array                         #
#               numConnections is the number of connections per hidden unit    #
#               sigma is the standard deviation matrix used in the Gaussian    #
#                                                                              #
#  Return Value: This function returns the connectivity matrix, which is an    #
#  h x p matrix, which has zeroes everywhere except for the locations with a   #
#  connection between the hidden unit and the pixel, where the value is 1.     #
#                                                                              #
#  Example Input:                                                              #
#  createConnectionMatrix([21, 21], np.array([[1,2], [9,10], [21,21]]),        #
#                         10, [[2, 0], [0,2]])                                 #
#==============================================================================#

def createConnectionMatrix(imageSize, hiddenUnitLocs, numConnections, sigma):

    ## Define some useful local variables for sake of clarity ##
    imgHeight = imageSize[0]
    imgLength = imageSize[1]
    numPixels = imgHeight * imgLength
    numHiddenUnits = len(hiddenUnitLocs)

    ## Initialize the Connection Matrix to all zeroes ##
    connectionMatrix = np.zeros(shape=(numHiddenUnits,numPixels))
    currHiddenUnit = 0 # index to keep track of which hidden unit we're working on.

    ## Loop through the Hidden Units to Create Samples ##
    for k in hiddenUnitLocs:

        i = 0
        while i < numConnections:

            # Get random Gaussian sample which returns an array of tuples
            [[x, y]] = np.random.multivariate_normal(k, sigma, 1)

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
            print "Using sample (%2d, %2d) from Gaussian with mean %s, std %s" % (x, y, k, sigma)
            connectionMatrix[currHiddenUnit][pixelLoc] = 1
            i += 1

        currHiddenUnit += 1

    return connectionMatrix

## Basic function to check equality of floating point numbers ##
def approx_equal(a, b, epsilon=0.000000001):
     return abs(a - b) < epsilon

## Basic tester that makes sure that the connectionMatrix has the
## correct amount of connections for now. Rest of checking was done manually.
def testConnectionMatrix(matrix, numConnection, numHiddenUnit):
    connectionCounter = 0
    for (r,c), value in np.ndenumerate(matrix):
        if approx_equal(value, 1.0):
            print "Hidden unit # %2d: Connection at %s" % (r, np.unravel_index(c, imageSize),)

            connectionCounter += 1

    assert numConnection * numHiddenUnit == connectionCounter


if __name__ == "__main__":

    # Three test cases:
    # 1.

    # 1. square image, single unit at center.
    nConns = 10
    imageSize = (20, 20)
    hiddenUnitLocs = np.asarray(((11, 11),))
    sigma = [[20, 0],[0, 20]]
    mat = createConnectionMatrix(imageSize, hiddenUnitLocs, nConns, sigma)

    testConnectionMatrix(mat, nConns, 1)