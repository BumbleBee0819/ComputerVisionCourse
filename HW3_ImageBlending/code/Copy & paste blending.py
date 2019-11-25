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
1) (m=m=2**n)----change padding area
2) (a,b value in "move" function)-----change position
'''

##############################################################################
#==============================================================================
# ###############     Method 2: Copy & Paste blending   #####################
#==============================================================================
##############################################################################




#==============================================================================
# ################     input & size/color  manipulation    #####################
#==============================================================================

# img2 is background

img1 = misc.imread('1.jpg')
img2 = misc.imread('2.jpg')
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
            pixel3[j,i]=img3[i,j]
            # note: mask is not mirror padding, instead, only padding black
        else:
            k=img1.shape[0]-1-abs(i-img1.shape[0]+1)
            l=img1.shape[1]-1-abs(j-img1.shape[1]+1)
            pixel1[j,i]=pixel1[l,k]
            

for i in range(m):
    for j in range(n):
        if i<img2.shape[0] and j<img2.shape[1]:
            pixel2[j,i]=r2[i,j],g2[i,j],b2[i,j]
            
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




#==============================================================================
# #######################    Define Functions    ###############################
#==============================================================================


##=============================================================    
#  1 change the position of the image (down: a; right: b)
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



##============================================================    
#  2
def blend(img1,img2,mask):
    p1= mask*img1
    p1=move(p1)
    p2=(1-move(mask))*img2 
    blended=p1+p2
    return blended 



#==============================================================================
#########  Combine R, G, B
#########  cut the padding area
#==============================================================================

blimg=Image.new('RGB',(img2.shape[1],img2.shape[0]),'white')
pixel=blimg.load()
r=blend(r1,r2,1.0*data3/255)
g=blend(g1,g2,1.0*data3/255)
b=blend(b1,b2,1.0*data3/255)

for i in range(img2.shape[1]):
    for j in range(img2.shape[0]):
        pixel[i,j]=int(r[j,i]),int(g[j,i]),int(b[j,i])

plt.imshow(blimg)
misc.imsave('C&P.jpg',blimg)

