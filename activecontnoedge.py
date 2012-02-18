from __future__ import division
import matplotlib.pyplot as plt
import Image
import scipy.ndimage
import numpy as np
import sys
from optparse import OptionParser


# taken from [1], which was taken from [3] 
def neumann_bound(f):
    nrow, ncol = f.shape
    g = f
    g([0, nrow-1],[0, ncol-1]) = g([2, nrow-3],[2, ncol-3])
    g([0, nrow-1],1:-2) = g([2, nrow-3],1:-2)
    g(1:-2,[0, ncol-1]) = g(1:-2,[2, ncol-3])
    return g

# taken from [1], which was taken from [3]
def curvature(u):
    ux, uy = np.gradient(u)
    normDu = np.sqrt((ux**2)+(uy**2)+1e-10)

    Nx = ux/normDu
    Ny = uy/normDu
    nxx, junk = np.gradient(Nx)
    junk, nyy = np.gradient(Ny)
    k = nxx + nyy
    return k
    

def reg_heaviside(x, epsilon):
    """Regularized heaviside step function from equation 8 in [4]"""
    return 0.5*(1 + (2 / np.pi) * arctan(x / epsilon))
    
def reg_dirac(x, epsilon):
    """Regularized delta dirac function from equation 9 in [4], is 
    the derivative of the regularized Heaviside step function """
    return (1 / np.pi) * (epsilon / (epsilon**2 + x**2))
    
def evolve(img, u, num_iterations, epsilon, thresh=0.5):
    """From [1]"""
    for i in range(num_iterations):
        u = neumann_bound(u)
        k = curvature(u)
        
        delta = reg_dirac(u, epsilon)
        heaviside = reg_heaviside(u, epsilon)
        
        inside_i = np.where(heaviside < thresh)
        outside_i = np.where(heaviside >= thresh)
        c1 = img[inside_i].mean()
        c2 = img[outside_i].mean()
        

if __name__ == '__main__':
    parser = OptionParser()about:startpage
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
# [3] - http://www.mathworks.com/matlabcentral/fileexchange/12711-level-set-for-image-segmentation
# [4] - Chunming Li; Chiu-Yen Kao; Gore, J.C.; Zhaohua Ding; , "Minimization of Region-Scalable Fitting Energy for Image Segmentation," Image Processing, IEEE Transactions on , vol.17, no.10, pp.1940-1949, Oct. 2008 URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4623242&isnumber=4623174
