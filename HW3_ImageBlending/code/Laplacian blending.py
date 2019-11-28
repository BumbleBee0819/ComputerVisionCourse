# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 12:57:42 2015

@author: wenyanbi


"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

'''
note: with different images, make change to 
1) line 34 (m=m=2**n)----change padding area
2) line  239-240 (a,b value in "move" function)-----change position
'''

##############################################################################
#==============================================================================
# ###############     Method 1: Lalacian blending   #####################
#==============================================================================
##############################################################################




#==============================================================================
# ################     input & size/color  manipulation    #####################
#==============================================================================
#img1 background
img1 = misc.imread('2.jpg')
img2 = misc.imread('1.jpg')
img3 = misc.imread('mask.jpg',flatten=1)


# seperate r,g,b for original image
data=np.array(img1)
r1,g1,b1=data[:,:,0], data[:,:,1], data[:,:,2]

data=np.array(img2)
r2,g2,b2=data[:,:,0], data[:,:,1], data[:,:,2]


# m,n should be 2**n (2**8=256, 2**9=512, 2**10=1024)
# change if the image is larger than 1024 or smaller than 512
m=n=512
img11=Image.new('RGB',(m,n),'white')
img22=Image.new('RGB',(m,n),'white')
img33=Image.new('L',(m,n),'black')

pixel1=img11.load()
pixel2=img22.load()
pixel3=img33.load()


# mirror padding (so that the size of padding image is 2**n)
for i in range(m):
    for j in range(n):
        if i<img1.shape[0] and j<img1.shape[1]:
            pixel1[j,i]=r1[i,j],g1[i,j],b1[i,j]
            # note: mask is not mirror padding, instead, only padding black
        else:
            k=img1.shape[0]-1-abs(i-img1.shape[0]+1)
            l=img1.shape[1]-1-abs(j-img1.shape[1]+1)
            pixel1[j,i]=pixel1[l,k]
            

for i in range(m):
    for j in range(n):
        if i<img2.shape[0] and j<img2.shape[1]:
            pixel2[j,i]=r2[i,j],g2[i,j],b2[i,j]
            pixel3[j,i]=img3[i,j]
            # note: mask is not mirror padding, instead, only padding black
        else:
            k=img2.shape[0]-1-abs(i-img2.shape[0]+1)
            l=img2.shape[1]-1-abs(j-img2.shape[1]+1)
            pixel2[j,i]=pixel2[l,k]
   
     
# r,g,b for padding image
data=np.array(img11)
r1,g1,b1=data[:,:,0], data[:,:,1], data[:,:,2]

data=np.array(img22)
r2,g2,b2=data[:,:,0], data[:,:,1], data[:,:,2]     

data3=np.array(img33) # matrix for mask


'''
# check: padding images
plt.imshow(img11)
plt.show()
plt.imshow(img22)
plt.show()
plt.imshow(img33,cmap='gray')
plt.show()
'''


########  create a  Binomial (5-tap) filter ########
kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])
plt.imshow(kernel)
plt.title ('kernel')
plt.show()







#==============================================================================
# #######################    Define Functions    ###############################
#==============================================================================


##=========================================================== 
#  1
def decimate(image):
    """
    Decimates at image with downsampling rate r=2.
    First filtering, then subsample
    """
    image_blur = ndimage.filters.convolve(image,kernel, mode='constant')
    #1/2
    return image_blur[::2, ::2]                                
               
    
##===========================================================        
# 2
def interpolate(image):
    """
    Interpolates an image with upsampling rate r=2.
    """
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    image_up[::2, ::2] = image
    return ndimage.filters.convolve(image_up,4*kernel, mode='constant')
    


##===========================================================    
#  3
def pyramids(image):
    """
    Constructs Gaussian and Laplacian pyramids.
    Parameters :
        image  : the original image (i.e. base of the pyramid)
    Returns :
        G   : the Gaussian pyramid
        L   : the Laplacian pyramid
    """
    ## Initialize pyramids
    G = [image, ]
    L = []
    rows, cols = image.shape
    
    ## Build the Gaussian pyramid to maximum depth
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        G.append(image)
             
    
        
    ## Build the Laplacian pyramid
    for i in range(len(G) - 1):
        L.append(G[i] - interpolate(G[i + 1]))
          
        
      
    ## show Gaussian pyramid
    composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
    composite_image[:rows, :cols] = G[0]
    
    i_row = 0
    
    for p in G[1:]:
         n_rows, n_cols = p.shape[:2]
         composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
         i_row += n_rows
   
    fig, ax = plt.subplots()
    ax.imshow(composite_image,cmap='gray')
    plt.title('Gaussian Pyramid')
    plt.show()
 

   
   ## show Laplacian pyramid
    composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
    composite_image[:rows, :cols] = L[0]

    i_row = 0
    for p in L[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    fig, ax = plt.subplots() 
    ax.imshow(composite_image,cmap='gray')
    plt.title('Laplacian Pyramid')
    plt.show()
   

    return G[:-1], L



##=============================================================    
#  4
def image_buildup (L): 
   for i in range(len(L) - 1):
       L[len(L)-2-i] = L[len(L)-2-i] + L[len(L)-1-i]
   return L[0] 




##=============================================================    
#  5 change the position of the image (down: a; right: b)
# if you want to move the position of the target image slightly 
# and you don't want to bother re-creating the mask
def move(p1):
    m,n=p1.shape
    p2=np.zeros((m,n))
    a=0
    b=0
    for i in range(m):
        for j in range(n):
            if p1[i,j]!=0 and i+a<m and j+b<n:
                p2[i+a,j+b]=p1[i,j]   
    return p2


##=============================================================    
#  6 normalize the mask
def norm(matrix):
    m,n=matrix.shape
    ma=matrix.max()
    if ma>1:
        matrix=1.0*matrix/255
    return matrix


##============================================================    
#  7
def blend(lpr_white,lpr_black,gauss_pyr_mask,move_mask):
    blended_pyr = []
    k = len(gauss_pyr_mask)
    for i in range(0,k):
        j=i
        while j>0:
            #interpolate all L-pyramids img1, img2 to size l[0]
            lpr_black[i]=interpolate(lpr_black[i]) 
            lpr_white[i]=interpolate(lpr_white[i])
            j=j-1
        gauss_pyr_mask[i]=norm(gauss_pyr_mask[i])
        move_mask[i]=norm(move_mask[i])
        # blend        
        p1= gauss_pyr_mask[i]*lpr_black[i]
        p1=move(p1)
        p2=(1-move_mask[i])*lpr_white[i] 
        blended_pyr.append( p1+p2 )
    return blended_pyr,k 



   
    

#==============================================================================
# ###################    Build G and L pyramids     ##################
#==============================================================================
[G1r,L1r] = pyramids(r1)
[G1g,L1g] = pyramids(g1)
[G1b,L1b] = pyramids(b1)
[G2r,L2r] = pyramids(r2)
[G2g,L2g] = pyramids(g2)
[G2b,L2b] = pyramids(b2)
[G3,L3] = pyramids(data3)
[G4,L4] = pyramids(move(data3))

#==============================================================================
# ###########################    Blend    #################################
#==============================================================================


# interpolate all G-pyramid masks to size G[0]
for i in range(len(G3)):
    j=i
    while j>0:
        G3[i]=interpolate(G3[i])
        G4[i]=interpolate(G4[i])
        j=j-1

blendedr,k=blend(L1r,L2r,G3,G4)
blendedg,k=blend(L1g,L2g,G3,G4)
blendedb,k=blend(L1b,L2b,G3,G4)




# Reconstruct r,g,b
i=k-2
while i >-1:
    blendedr[i]=blendedr[i]+blendedr[i+1]
    i=i-1

i=k-2
while i >-1:
    blendedg[i]=blendedg[i]+blendedg[i+1]
    i=i-1
   
i=k-2
while i >-1:
    blendedb[i]=blendedb[i]+blendedb[i+1]
    i=i-1





#==============================================================================
# #### Check: Collapse Laplacian pyramids (should get the original image)  ####
#==============================================================================
# a. image 1
r11=image_buildup (L1r)
g11=image_buildup(L1g)
b11=image_buildup(L1b)

reconstruct_img11=Image.new('RGB',(m,n),'white')
pixel=reconstruct_img11.load()

for i in range(m):
    for j in range(n):
        pixel[j,i]=int(r11[i,j]),int(g11[i,j]),int(b11[i,j])


plt.subplot(121)
plt.imshow(reconstruct_img11)
plt.title('check_reconstructed image1')
plt.subplot(122)
plt.imshow(img11)
plt.title('original image1')
plt.show()


# b. image 2
r22=image_buildup (L2r)
g22=image_buildup(L2g)
b22=image_buildup(L2b)

reconstruct_img22=Image.new('RGB',(m,n),'black')
pixel=reconstruct_img22.load()

for i in range(m):
    for j in range(n):
        pixel[j,i]=int(r22[i,j]),int(g22[i,j]),int(b22[i,j])

plt.subplot(121)
plt.imshow(reconstruct_img22)
plt.title('check_reconstructed image2')
plt.subplot(122)
plt.imshow(img22)
plt.title('original image2')
plt.show()






#==============================================================================
#########  Combine R, G, B
#########  cut the padding area
#==============================================================================
blimg=Image.new('RGB',(img1.shape[1],img1.shape[0]),'white')
pixel=blimg.load()
r=blendedr[0]
g=blendedg[0]
b=blendedb[0]

for i in range(img1.shape[1]):
    for j in range(img1.shape[0]):
        pixel[i,j]=int(r[j,i]),int(g[j,i]),int(b[j,i])


plt.imshow(blimg)
misc.imsave('blend.jpg',blimg)

