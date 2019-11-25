# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 09:41:37 2015

@author: wenyanbi
"""


import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import math as e 

###############################################################################
############################ define functions #################################

    
############  1. define gaussian filter########################################
def gauss(shape,sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h=h/sumh
    return h
 

     
############  2. define fft for color image ##############################

def fft_color (img, filterm):
    pixels=img.load()
    m=img.size[0]
    n=img.size[1]
    
   ######## seperate r,g,b
    r=np.zeros((m,n))
    g=np.zeros((m,n))
    b=np.zeros((m,n))
    
    for i in range (m):
        for j in range (n):    
            r[i,j],g[i,j],b[i,j]=pixels[i,j]
    
   #### FFT to R,G,B, & filter seperately 
    fr = np.fft.fft2(r)
    fr = np.fft.fftshift(fr)
    fg = np.fft.fft2(g)
    fg = np.fft.fftshift(fg)
    fb = np.fft.fft2(b)
    fb = np.fft.fftshift(fb)
    ffilter=np.fft.fft2(filterm)
    ffilter=np.fft.fftshift(ffilter)
    ffilter=np.abs(ffilter)
   
   
   ### convolve:  FFT multiple
   #convolve
    red1=ffilter*fr
    green1=ffilter*fg
    blue1=ffilter*fb

   #image_back
    img_reda=np.fft.ifftshift(red1)
    img_red=np.fft.ifft2(img_reda)
    img_red=np.abs(img_red)

    img_greena=np.fft.ifftshift(green1)
    img_green = np.fft.ifft2(img_greena)
    img_green = np.abs(img_green)

    img_bluea=np.fft.ifftshift(blue1)
    img_blue = np.fft.ifft2(img_bluea)
    img_blue= np.abs(img_blue)
   
   
   ######## combine r,g,b
    finalimg=Image.new( 'RGB', (m,n), "white") 
    pixel=finalimg.load()

    for i in range(m):
        for j in range(n): 
            pixel[i,j]=int(img_red[i,j]),int(img_green[i,j]),int(img_blue[i,j])
    
    return finalimg, pixel
    
    

############  3. hybrid image      ########################################
def hybrid (img1,pixell,pixelh):
    finalimg=Image.new( 'RGB', (img1.size[0],img1.size[1]), "white")
    pixel=finalimg.load()

    for i in range(img1.size[0]):
        for j in range(img1.size[1]):    #######change with different images !!!!!!
            r1,g1,b1=pixelh[i,j]
            r2,g2,b2=pixell[i,j]
            q,w,t=pixelh[i,j]
            pixelh[i,j]=q+100,w+100,t+100
            pixel[i,j]=int(r1+r2),int(g1+g2),int(b1+b2)
    return finalimg


###############################################################################
############### do the filtering ##############################################
#filter (low/high)
filterlow=gauss ((img1.size[0],img1.size[1]),7)
filterhigh=np.zeros(shape=(img1.size[0],img1.size[1]))
filterhigh[img1.size[0]/2,img1.size[1]/2]=1                    
filterhigh=filterhigh-gauss((img1.size[0],img1.size[1]),11)


#read image
img2=Image.open('1 einstein.bmp')
img1=Image.open('1 marilyn.bmp')

'''
img2=Image.open('1 einstein.bmp')
img1=Image.open('1 marilyn.bmp')
img2=Image.open('2 cat.bmp')
img1=Image.open('2 dog.bmp')
img2=Image.open('3 plane.bmp')
img1=Image.open('3 bird.bmp')
img2=Image.open('4 fish.bmp')
img1=Image.open('4 submarine.bmp')
img2=Image.open('5 bicycle.bmp')
img1=Image.open('5 motorcycle.bmp')
'''

#filtering
img1,pixel1=fft_color (img1,filterlow)
img2,pixel2=fft_color(img2,filterhigh)
plt.imshow(img1)
plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(img2)
plt.xticks([]), plt.yticks([])
plt.show()

#hybrid
hybridimg=hybrid (img1,pixel1,pixel2)
plt.imshow(hybridimg)
plt.xticks([]), plt.yticks([])
plt.show()






 

