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

    #imgin = Image.open("/home/mh23g08/susanedge/susanedge/test_data/fish_image_small.jpg")
    imgin = Image.open(args[0])
	
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point
    
    
    plt.set_cmap(plt.cm.gray)
    plt.imshow(img)
    plt.show()
