
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
#==============================================================================#

def createConnectionMatrix(imageSize, hiddenUnitLocs, numConnections, sigma):    
    ## Define some useful local variables for sake of clarity ##
    imgHeight = imageSize[0]
    imgLength = imageSize[1]
    numPixels = imgHeight * imgLength 
    numHiddenUnits = hiddenUnitLocs.shape[1] + 1 #zero-based so add one
    
    
    ## Initialize the Connection Matrix ##
    connectionMatrix = np.zeros(shape=(numHiddenUnits,numPixels))
    
    currHiddenUnit = 0
    ## Loop through the Hidden Units to Create Samples ##
    for k in hiddenUnitLocs:
        i = 0
        # 10 Samples for each Matrix
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
            pixelLoc = (x-1) * imgLength + (y-1)
            
            if (connectionMatrix[currHiddenUnit][pixelLoc] == 1):
                continue

            connectionMatrix[currHiddenUnit][pixelLoc] = 1
            i += 1
        
        currHiddenUnit += 1        
    
    return connectionMatrix
