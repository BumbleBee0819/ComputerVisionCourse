
# Local Feature Stencil Code
# CS 589 Computater Vision, American Uniersity, Bei Xiao
# Adapted from James Hayes's MATLAB starter code for Project.2

# % This script 
# % (1) Loads and resizes images
# % (2) Finds interest points in those images                 (you code this)
# % (3) Describes each interest point with a local feature    (you code this)
# % (4) Finds matching features                               (you code this)
# % (5) Visualizes the matches
# % (6) Evaluates the matches based on ground truth correspondences

# % There are numerous other image sets in the data sets folder uploaded. 
# % You can simply download images off the Internet, as well. However, the
# % evaluation function at the bottom of this script will only work for this
# % particular image pair (unless you add ground truth annotations for other
# % image pairs). It is suggested that you only work with these two images
# % until you are satisfied with your implementation and ready to test on
# % additional images. 

# A single scale pipeline works fine for these two
# images (and will give you full credit for this project), but you will
# need local features at multiple scales to handle harder cases.


# % You don't have to work with grayscale images. Matching with color
# % information might be helpful.

import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from get_interest_points import get_interest_points
from get_features import get_features
from match_features import match_features
from show_correspondence import show_correspondence 
from evaluate_correspondence import evaluate_correspondence
from math import sqrt


# read in the notre dame images

image1 = cv2.imread('1 church1.jpg')
image2 = cv2.imread('1 church2.jpg')

#image1 = cv2.imread('2 mountain1.jpg')
#image2 = cv2.imread('2 mountain2.jpg')

#image1 = cv2.imread('3 Capricho Gaudi2.jpg')
#image2 = cv2.imread('3 Capricho Gaudi1.jpg')



# convert to grayscale
image1 =  cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 =  cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

scale_factor = 0.3; #make images smaller to speed up the algorithm

#height1, width1 = image1.shape[:2]
#height2, width2 = image2.shape[:2]
#image1 = cv2.resize(image1,(width1/2, height1/2), interpolation = cv2.INTER_CUBIC)
#image2 = cv2.resize(image2,(width2/2,height1/2), interpolation = cv2.INTER_CUBIC)

feature_width = 64; #width and height of each local feature, in pixels. 

# %% Find distinctive points in each image. Szeliski 4.1.1
# % !!! You will need to implement get_interest_points. !!!
def matlab_style_gauss2D(shape,sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h =np.exp(-(x*x + y*y)/(2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
    
# construct derivative kernals from scratch
def gauss_derivative_kernels(size, sizey=None):
    """ returns x and y derivatives of a 2D 
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]

    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    gy = - y * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 

    return gx,gy    
    
# computing derivatives     
def gauss_derivatives(im, n, ny=None):
    """ returns x and y derivatives of an image using gaussian 
        derivative filters of size n. The optional argument 
        ny allows for a different size in the y direction."""

    gx,gy = gauss_derivative_kernels(n, sizey=ny)

    imx = signal.convolve(im,gx, mode='same')
    imy = signal.convolve(im,gy, mode='same')

    return imx,imy
    
#gx,gy = gauss_derivative_kernels(3)


def compute_harris_response(image):
    """ compute the Harris corner detector response function 
        for each pixel in the image"""

    #derivatives
    imx,imy = gauss_derivatives(image, 5)

    #kernel for blurring
    gauss = matlab_style_gauss2D((5,5),1)

    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')

    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr

def get_harris_points(harrisim,min_dist, threshold):
    # find top corner candiates above a threshold
    corner_threshold =  harrisim.max()*threshold
    harrisim_t = harrisim > corner_threshold
    
    #get the coordinates, all the non-zero components 
    coords = np.array(harrisim_t.nonzero()).T
    
    # ...add their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    
    # sort candidates in descending order of corner responses
    index = np.argsort(candidate_values)
    
    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    
    # select the best points taking min_dist into account
    filtered_coords = [] 
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0     
    return filtered_coords

# plotting the corners onto the image
def plot_harris_points(image,filtered_coords):
    plt.figure()
    plt.imshow(image,cmap='gray')
    plt.plot([p[1]for p in filtered_coords], [p[0] for p in filtered_coords],'*')
    plt.show()


# calling the harris corner detector
def get_interest_points(im, feature_width):
    
    harrisim = compute_harris_response(im)
    ''' change feature detector threshold here'''
    filtered_coords = get_harris_points(harrisim,12,0.1)
    plot_harris_points(im,filtered_coords)
    #print filtered_coords
    x=np.zeros((len(filtered_coords)))
    y=np.zeros((len(filtered_coords)))
    for i in range(len(filtered_coords)):
        [y[i],x[i]]=filtered_coords[i]

    return [x,y]
    
[x1, y1] = get_interest_points(image1, feature_width)
[x2, y2] = get_interest_points(image2, feature_width)



# %% Create feature vectors at each interest point. Szeliski 4.1.2
# % !!! You will need to implement get_features. !!!
def fdegree(dx,dy):
    n=8
    
    r2=dx**2+dy**2
    if dy**2/r2<=0.5 and dy>=0 and dx>=0:
        n=0
    if dx==0 and dy==0:
        n=0
    if dy**2/r2>0.5 and dx>=0 and dy>=0:
        n=1
    if dy**2/r2>0.5 and dx<0 and dy>0:
        n=2
    if dy**2/r2<=0.5 and dx<0 and dy>0:
        n=3
    if dy**2/r2<=0.5 and dx<=0 and dy<=0:
        n=4
    if dy**2/r2>0.5 and dx<=0 and dy<=0:
        n=5
    if dy**2/r2>0.5 and dx>0 and dy<0:
        n=6
    if dy**2/r2<=0.5 and dx>0 and dy<0:
        n=7
    return n

def gradient(imdx,imdy,x,y,width):
    bx=np.zeros((width,width))
    by=np.zeros((width,width))
    
    a,b=imdx.shape

    #print x,y
    #print imdx.shape 
    for i in range(width):
        for j in range(width):
            if x-width/2+i<b and x-width/2+i>=0 and y-width/2+j<a and y-width/2+j>=0:
                bx[i,j]=imdx[y-width/2+j,x-width/2+i]
                by[i,j]=imdy[y-width/2+j,x-width/2+i]
            else:
                bx[i,j]=0
                by[i,j]=0
    
    
    h=matlab_style_gauss2D((width,width),50)
    bx=bx*h
    by=by*h
    bins=np.zeros((width*width/16,8))
    
    for i in range(width):
        for j in range(width):
            n=fdegree(bx[i,j],by[i,j])
            bins[i/(width/4)*(width/4)+j/(width/4),n]=bins[i/(width/4)*(width/4)+j/(width/4),n]+sqrt(bx[i,j]**2+by[i,j]**2)


    #normalize
    bins=bins/sqrt((bins*bins).sum())
    for i in range(width*width/16):
        #print bins[i,:].min()
        for j in range(8):
            if bins[i,j]>0.2:
                bins[i,j]=0.2
    bins=bins/sqrt((bins*bins).sum())
    f=bins.flatten()
    return f

                


def get_features(image, x, y, feature_width):
    sobelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    imdx = signal.convolve(image, sobelx, mode='same') # horizontal derivative
    imdy = signal.convolve(image, sobely, mode='same')  # vertical derivative
    
    
    
    features = np.zeros((len(x), feature_width*feature_width/2));
    for i in range(len(x)):
        f=gradient(imdx,imdy,x[i],y[i],feature_width)
        for j in range(feature_width*feature_width/2):
            features[i,j]=f[j]
            
    return features 

image1_features = get_features(image1, x1, y1, feature_width)
image2_features = get_features(image2, x2, y2, feature_width)


# %% Match features. Szeliski 4.1.3

def ssd(f1,f2):
    r=0
    for i in range(len(f1)):
        r=r+(f1[i]-f2[i])**2
    return sqrt(r)


def ratio2(d):
    ratio=np.zeros((len(d[:,0])))
    #for i in range(len(d[:,0])):
    ratio[:]=d[:,0]/d[:,1]
    return ratio
    
    

def match_features(features1, features2):

    # % This function does not need to be symmetric (e.g. it can produce
    # % different numbers of matches depending on the order of the arguments).
    # % To start with, simply implement the "ratio test", equation 4.18 in
    # % section 4.1.3 of Szeliski. For extra credit you can implement various
    # % forms of spatial verification of matches.
    # % Placeholder that you can delete. Random matches and confidences
    num_features1 = features1.shape[0]
    num_features2 = features2.shape[0]
    # this is annoying for Python, if you want the number to be integer, you must specify its data type
    matches = np.zeros((num_features1, 2))
    d2=np.zeros((num_features1, num_features2))
    for i in range(num_features1):
        for j in range(num_features2):
            d2[i,j]=ssd(features1[i],features2[j])
    
    
    d2pie=np.sort(d2)
    ratio=ratio2(d2pie)
    
    
    for i in range(num_features1):
        matches[i,0]=i
        
        for j in range(num_features2):
            if d2[i,j]==d2[i,:].min():
                matches[i,1]=j
                break
    

    return matches,ratio
    

    




# % !!! You will need to implement get_features. !!!
[matches, confidences] = match_features(image1_features, image2_features)

# % You might want to set 'num_pts_to_visualize' and 'num_pts_to_evaluate' to
# % some constant once you start detecting hundreds of interest points,
# % otherwise things might get too cluttered. You could also threshold based
# % on confidence.
num_pts_to_visualize = matches.shape[0]

#show_correspondence(image1, image2, x1[matches[0:num_pts_to_visualize,0:1]],y1[matches[0:num_pts_to_visualize,0:1]],x2[matches[0:num_pts_to_visualize,1:2]],y2[matches[0:num_pts_to_visualize,1:2]])

show_correspondence(image1,image2,x1,y1,x2,y2,matches,confidences)

#num_pts_to_evaluate = matches.shape[0]

# you can also end your code by this:

#fig.savefig('vis.jpg')
#print 'Saving visualization to vis.jpg'

# # % All of the coordinates are being divided by scale_factor because of the
# # % imresize operation at the top of this script. This evaluation function
# # % will only work for the particular Notre Dame image pair specified in the
# # % starter code. You can simply comment out
# # % this function once you start testing on additional image pairs.







