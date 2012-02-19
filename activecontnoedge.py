from __future__ import division
import matplotlib.pyplot as plt
import Image
import scipy.ndimage
import scipy.spatial
import numpy as np
import sys
from optparse import OptionParser
from mpl_toolkits.mplot3d import Axes3D

# taken from [1], which was taken from [3] 
def neumann_bound(f):
    nrow, ncol = f.shape
    g = f.copy()
    g[[0, nrow-1],[0, ncol-1]] = g[[2, nrow-3],[2, ncol-3]]
    g[[0, nrow-1], 1:-2] = g[[2, nrow-3],1:-2]
    g[1:-2,[0, ncol-1]] = g[1:-2,[2, ncol-3]]
    return g

# taken from [1], which was taken from [3]
def central_curvature_old(u):
    ux, uy = np.gradient(u)
    normDuX = np.sqrt((ux**2)+(uy**2)+1e-10)
    normDuY = np.sqrt((ux**2)+(uy**2)+1e-10)

    Nx = ux / normDuX
    Ny = uy / normDuY
    nxx, junk = np.gradient(Nx)
    junk, nyy = np.gradient(Ny)
    k = nxx + nyy
    return k
    
def central_curvature(u):
    # this is just calculating div(grad(u)/norm(grad(u))) - look up definition of divergance
    grad_u = np.gradient(u)
    norm_grad_u = np.linalg.norm(grad_u)

    divided = grad_u / norm_grad_u
    
    k = divided[0] + divided[1]
    return k

def reg_heaviside(x, epsilon):
    """Regularized heaviside step function from equation 8 in [4]"""
    return 0.5*(1 + (2 / np.pi) * np.arctan(x / epsilon))
    
def reg_dirac(x, epsilon):
    """Regularized delta dirac function from equation 9 in [4], is 
    the derivative of the regularized Heaviside step function """
    return (epsilon / np.pi) / ((epsilon**2) + (x**2))
    
def evolve(u0, img, timestep, epsilon, mu, v, lambda1, lambda2, pc, thresh=0.5):
    """To help us understand the algorithm the following code has been ported from [1].
    The algorithm seems to use the central finite differences to solve the PDEs, following
    the approach taken in [4]. Gradient descent is used to mimimise the contour energy."""
    u = u0.copy()
    u = neumann_bound(u)
    K = central_curvature(u) # div(grad(u)/mod(grad(u)))
    
    delta = reg_dirac(u, epsilon)
    heaviside = reg_heaviside(u, epsilon)
    
    # BUG FOUND HERE: The inequalities here were the wrong way around
    inside_i = np.where(heaviside >= thresh)
    outside_i = np.where(heaviside < thresh)
    #inside_i = np.where(heaviside <= thresh)
    #outside_i = np.where(heaviside > thresh)
    
    
    c1 = img[inside_i].mean()
    c2 = img[outside_i].mean()
    
    #inout = np.zeros(img.shape)
    #inout[inside_i] = 1
    #inout[outside_i] = -1
    #plt.imshow(K)
    #plt.colorbar()
    #plt.show()
    
    
    
    euler_lagrange_eqn = delta * (mu * K - v - lambda1 * ((img - c1)**2) + lambda2 * ((img - c2)**2)) # from [1] and [2] (eqn 9)
    
    
    # this next term is described as a "distance regularation term" in [1] and can be seen in equation 15 of [4] where it is constant * (laplacian(u) - K)
    # in [1] 4*del2(u) is the finite difference approximation of laplaces differential operator according to [5]
    # in numpy we can use scipy.ndimage.filters.laplace
    P = pc * (scipy.ndimage.filters.laplace(u) - K)
    #P = pc * (4 * del2(u) - K)
    #P = 0
    
    
    u = u + timestep * (euler_lagrange_eqn + P) # gradient descent
    return u
        

if __name__ == '__main__':
    parser = OptionParser()
    #parser.add_option("-n", "--no-dir", help="don't plot direction on the image", action="store_true", default=False)
    parser.add_option("-n", "--num-iterations", action="store", type="int", default=10)
    parser.add_option("-m", "--median-filter-size", action="store", type="int", default=0)
    ## set up the rest of the variables, we use the values from [1] 
    #lambda1 = 1.0 # weighting of the internal force, it is fixed to one as described in [2]
    parser.add_option("--lambda1", action="store", type="float", default=1.0) # weighting of the internal force, it is fixed to one as described in [2]
    #lambda2 = 1.0
    parser.add_option("--lambda2", action="store", type="float", default=1.0) # weighting of the internal force, it is fixed to one as described in [2]
    #timestep = 0.1 # from [1]
    parser.add_option("--timestep", action="store", type="float", default=0.1) # from [1]
    #v = 1.0 # v is a constraint on the area inside the curve, value of 1 taken from from [1], but in [2] it is set to zero #################### SHOULD THIS BE SET TO ZERO    
    parser.add_option("--v", action="store", type="float", default=1.0) # v is a constraint on the area inside the curve, value of 1 taken from from [1], but in [2] it is set to zero #################### SHOULD THIS BE SET TO ZERO    
    #mu = 1.0 # mu is a constraint upon the length of the curve, value of 1 taken from [1]
    parser.add_option("--mu", action="store", type="float", default=1.0) # mu is a constraint upon the length of the curve, value of 1 taken from [1]
    #epsilon = 1.0 # epsilon is a parameter of the regularisation of the heaviside function and dirac delta function value taken from [1]
    parser.add_option("--epsilon", action="store", type="float", default=1.0) # epsilon is a parameter of the regularisation of the heaviside function and dirac delta function value taken from [1]
    #pc = 1.0 # [1] describes this as the "penalty coefficient" and it is used to avoid reinitialization according to [4]
    parser.add_option("--pc", action="store", type="float", default=1.0) # [1] describes this as the "penalty coefficient" and it is used to avoid reinitialization according to [4]
    #thresh = 0.5
    parser.add_option("--thresh", action="store", type="float", default=0.5) # threshold used for determining which areas are inside and outside the contour
    parser.add_option("-p", "--hide-progress-image", action="store_true", default=False)
    parser.add_option("-w", "--use-webcam", action="store_true", default=False)
    parser.add_option("--noise", action="store", type="float", default=0.0)
    
    
    
    (options, args) = parser.parse_args()
    
    if options.use_webcam:
        from VideoCapture import Device
        cam = Device()
        imgin = cam.getImage()
    else:
        imgin = Image.open(args[0])
	
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point
    
    
    noise = np.random.randn(img.shape[0], img.shape[1]) * options.noise
    img += noise
    
    if options.median_filter_size != 0:
        img = scipy.ndimage.filters.median_filter(img, options.median_filter_size)
    
    fig = plt.figure()
    selpoints = []
    def onclick(event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)
        selpoints.append((int(event.xdata), int(event.ydata)))
        if len(selpoints) >= 2:
            plt.close('all')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.imshow(img)
    plt.show()
    
    #############################################################
    
    ## following lines initialise the level set function (Initial Level-set Function (ILS)) with an initial square contour in the middle
    c0 = 2 # from [1]
    ils = np.ones(img.shape) * c0 # from [1]
    #ils[50:img.shape[0] - 50, 50:img.shape[1] - 50] = -c0 # from [1]
    print selpoints
    ils[selpoints[0][1]:selpoints[1][1], selpoints[0][0]:selpoints[1][0]] = -c0 # from [1]
    
    plt.imshow(ils)
    plt.show()
    u = ils.copy()

    if not options.hide_progress_image:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.set_cmap(plt.cm.gray)
    
    for i in range(options.num_iterations):
        u = evolve(u, img, options.timestep, options.epsilon, options.mu, options.v, options.lambda1, options.lambda2, options.pc, options.thresh)
        print i, "of", options.num_iterations
        
        if not options.hide_progress_image:
            ax.clear()
            #ax.imshow(u)
            ax.imshow(img)
            cont = ax.contour(u, [0, 0], colors='r')
            plt.show(block=False)
            plt.draw()
                    
    #############################################################
    
    if options.hide_progress_image:
        plt.imshow(img)
        plt.colorbar()
        plt.set_cmap(plt.cm.gray)
        cont = plt.contour(u, [0, 0], colors='r')
        
    #ax3d = fig.add_subplot(111, projection='3d')
    
    #fig2 = plt.figure()
    #plt.imshow(u)
    #plt.colorbar()
    #cont = plt.contour(u, [0, 0], colors='r')
    #plt.clabel(cont, inline=1, fontsize=10)
    
    
    

    
    plt.show()
    
    
# References:
# [1] - http://www.mathworks.com/matlabcentral/fileexchange/34548-active-contour-without-edge
# [2] - Chan, T.F.; Vese, L.A.; , "Active contours without edges," Image Processing, IEEE Transactions on , vol.10, no.2, pp.266-277, Feb 2001 URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=902291&isnumber=19508
# [3] - http://www.mathworks.com/matlabcentral/fileexchange/12711-level-set-for-image-segmentation
# [4] - Chunming Li; Chiu-Yen Kao; Gore, J.C.; Zhaohua Ding; , "Minimization of Region-Scalable Fitting Energy for Image Segmentation," Image Processing, IEEE Transactions on , vol.17, no.10, pp.1940-1949, Oct. 2008 URL: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2720140/
# [5] - http://www.mathworks.co.uk/help/techdoc/ref/del2.html
