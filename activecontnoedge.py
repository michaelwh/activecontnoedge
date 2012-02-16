from __future__ import division
import matplotlib.pyplot as plt
import Image
import scipy.ndimage
import numpy as np
import sys
from optparse import OptionParser



if __name__ == '__main__':
    parser = OptionParser()
    #parser.add_option("-n", "--no-dir", help="don't plot direction on the image", action="store_true", default=False)
    (options, args) = parser.parse_args()
    
    imgin = Image.open(args[0])
	
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point
    
    #############################################################
    
    ## following lines initialise the level set function (Initial Level-set Function (ILS)) with an initial square contour in the middle
    c0 = 2 # from [1]
    ils = np.ones(img.shape)*c0 # from [1]
    ils[50:img.shape[0] - 50, 50:img.shape[1] - 50] = -c0 # from [1]
    
    ## set up the rest of the variables, we use the values from [1] 
    lambda1 = 1 # weighting of the internal force, it is fixed to one as described in [2]
    lambda2 = 1 # weighting of the external force, it is fixed to one as described in [2]
    timestep = 0.1 # from [1]
    v = 1 # v is a constraint on the area inside the curve, value of 1 taken from from [1], but in [2] it is set to zero #################### SHOULD THIS BE SET TO ZERO    
    mu = 1 # mu is a constraint upon the length of the curve, value of 1 taken from [1]
    epsilon = 1 # epsilon is a parameter of the regularisation of the heaviside function and dirac delta function value taken from [1]
    
    
    
    
    #############################################################
    
    plt.set_cmap(plt.cm.gray)
    plt.imshow(img)
    plt.show()

# References:
# [1] - http://www.mathworks.com/matlabcentral/fileexchange/34548-active-contour-without-edge
# [2] - Chan, T.F.; Vese, L.A.; , "Active contours without edges," Image Processing, IEEE Transactions on , vol.10, no.2, pp.266-277, Feb 2001 URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=902291&isnumber=19508

