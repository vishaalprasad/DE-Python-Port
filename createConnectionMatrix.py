import numpy as np
import matplotlib.pyplot as plt

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
#               variance is the covariance matrix used in the Gaussian         #
#                                                                              #
#  Return Value: This function returns the connectivity matrix, which is an    #
#  h x p matrix, which has zeroes everywhere except for the locations with a   #
#  connection between the hidden unit and the pixel, where the value is 1.     #
#                                                                              #
#  Example Input:                                                              #
#  createConnectionMatrix([21, 21], np.array([[1,2], [9,10], [21,21]]),        #
#                         10, [[2, 0], [0,2]])                                 #
#==============================================================================#

def createConnectionMatrix(imageSize, hiddenUnitLocs, numConnections, variance):

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
            #print "Using sample (%2d, %2d) from Gaussian with mean %s, std %s" % (x, y, k, variance)

            connectionMatrix[currHiddenUnit][pixelLoc] = 1
            i += 1

        currHiddenUnit += 1

    return connectionMatrix

## Basic function to check equality of floating point numbers ##
def approx_equal(a, b, epsilon=0.000000001):
     return abs(a - b) < epsilon

## Basic tester that makes sure that the connectionMatrix has the
## correct amount of connections for now. Rest of checking was done manually.
def testConnectionMatrix(matrix, numConnection, numHiddenUnit, imageSize):
    connectionCounter = 0
    for (r,c), value in np.ndenumerate(matrix):
        if approx_equal(value, 1.0):
            print "Hidden unit # %2d: Connection at %s" % (r, np.unravel_index(c, imageSize),)

            connectionCounter += 1
    try:
        assert True
        assert numConnection * numHiddenUnit == connectionCounter
    except AssertionError:
        print "Expected num of connections: %d.\n Received: %d." % (numConnection * numHiddenUnit, connectionCounter)
        exit(1)

# Graph a matrix on the Cartesian Plane, with a marking at any location there's a connection or a hidden unit
def graphConnectionMatrix(matrix, imageSize, hiddenUnitLocs):
    plt.figure() #create new window for graph

    # Plot all of the hidden unit locations
    for x,y in hiddenUnitLocs:
        plt.plot(x,y,'ro');

    # Plot all the connection points
    for (r,c), value in np.ndenumerate(matrix):
        if approx_equal(value, 1.0):
            [x,y] = np.unravel_index(c, imageSize)
            plt.plot(x,y,'bx');
            #print [x, y]

    plt.axis([-1, imageSize[0], -1, imageSize[1]]);
    plt.show()



def doTest(testName, nConns, imageSize, hiddenUnitLocs, variance):

    print "---------------------------\n"
    print "\n%s:\n" % testName

    mat = createConnectionMatrix(imageSize, hiddenUnitLocs, nConns, variance)

    print "\nTesting and Graphing Resulting Matrix:\n"
    testConnectionMatrix(mat, nConns, len(hiddenUnitLocs), imageSize)
    graphConnectionMatrix(mat, imageSize, hiddenUnitLocs)


def doRatioTest(testName, nConns, imageSize, hiddenUnitLocs, variance):

    print "---------------------------\n"
    print "\n%s:\n" % testName

    ratioOfConnections = 0.0
    numTrials = 100
    for i in range(0, numTrials):
        mat = createConnectionMatrix(imageSize, hiddenUnitLocs, nConns, variance)
        ratioOfConnections += getPropOfHUwithSample(mat, hiddenUnitLocs, imageSize)
    ratioOfConnections = ratioOfConnections / numTrials
    print "Ratio is: " + str(ratioOfConnections)


# Given a matrix and the hidden unit locations, find the proportion of hidden
# units that also have a sample at their locaiton.
def getPropOfHUwithSample(matrix, hiddenUnitLocs, imageSize):
    numHULocs = len(hiddenUnitLocs)
    numSampleAtHULoc = 0 #counter
    for (r,c), value in np.ndenumerate(matrix):
        if approx_equal(value, 1.0):
            [x,y] = np.unravel_index(c, imageSize) #current sample
            b = hiddenUnitLocs[r] #get the current hidden unit

            if (x == b[1] and y == b[0]):

                numSampleAtHULoc += 1
    return numSampleAtHULoc / float(numHULocs) #get the proportion


if __name__ == "__main__":

    myTest = doTest #variable to easily switch between tests

    # Four test cases:

    # 1. square image, single unit at center.
    myTest(testName="single unit at center",
        nConns=10,
        imageSize=(20, 20),
        hiddenUnitLocs=np.asarray(((10, 10),)),
        variance = [[20, 0],[0, 20]])


    # 2. square image, four units symmetrical about the center
    myTest(testName="four units symmetrical about the center",
        nConns=24,
        imageSize=(80, 80),
        hiddenUnitLocs = np.array([[14, 39], [39, 14], [39, 64], [64, 39]]),
        variance = [[15, 0], [0, 15]])

    # 3. square image, four units at each corner
    myTest(testName="four units at each corner",
        nConns = 15,
        imageSize = (50, 50),
        hiddenUnitLocs = np.array([[0, 0], [49, 0], [0, 49], [49, 49]]), #four corners
        variance = [[10, 0], [0, 10]])

    # 4. small image, three units, 12 connections, 12 pixels. each pixel maps to each hidden unit
    if (myTest != doRatioTest):
        myTest(testName="3 hidden units, connecting to all pixels",
            nConns = 12,
            imageSize = (4, 3),
            hiddenUnitLocs = np.array([[0,0], [2,2], [3,2]]),
            variance = [[1, 0], [0, 1]])
