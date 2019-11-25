# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 21:41:38 2015

@author: Wenyan Bi
"""

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image



def gauss(shape,sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h=h/sumh
    return h
    
    
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


img1 = Image.open('1.jpg')
img2 = Image.open('2.jpg')
mask = misc.imread('mask.jpg',flatten=1)

m,n=img1.size[0],img1.size[1]
 
filterlow1=gauss ((img1.size[0],img1.size[1]),0.1)
filterhigh1=np.zeros(shape=(img1.size[0],img1.size[1]))
filterhigh1[img1.size[0]/2,img1.size[1]/2]=1                    
filterhigh1=filterhigh1-filterlow1


filterlow2=gauss ((img2.size[0],img2.size[1]),0.1)
filterhigh2=np.zeros(shape=(img2.size[0],img2.size[1]))
filterhigh2[img2.size[0]/2,img2.size[1]/2]=1                    
filterhigh2=filterhigh2-filterlow2

N=50
kernel=np.ones((N,N))
kernel=1.0*kernel/(N*N)
lmask=ndimage.filters.convolve(mask,kernel, mode='nearest')
mask=1.0*mask/255
lmask=1.0*lmask/255

img1h,pixel1h=fft_color(img1,filterhigh1)
img2h,pixel2h=fft_color(img2,filterhigh2)

img1l,pixel1l=fft_color(img1,filterlow1)
img2l,pixel2l=fft_color(img2,filterlow2)

img2band=Image.new('RGB',(m,n),'black')
pixel=img2band.load()

for i in range(m):
    for j in range(n):
        r1h,g1h,b1h=pixel1h[i,j]
        r2h,g2h,b2h=pixel2h[i,j]
        r1l,g1l,b1l=pixel1l[i,j]
        r2l,g2l,b2l=pixel2l[i,j]
        
        r=r1h*mask[j,i]+r2h*(1-mask[j,i])+r1l*lmask[j,i]+r2l*(1-lmask[j,i])
        g=g1h*mask[j,i]+g2h*(1-mask[j,i])+g1l*lmask[j,i]+g2l*(1-lmask[j,i])
        b=b1h*mask[j,i]+b2h*(1-mask[j,i])+b1l*lmask[j,i]+b2l*(1-lmask[j,i])
        
        pixel[i,j]=int(r),int(g),int(b)
        

        
plt.imshow(img2band)
plt.show()
misc.imsave('2band.jpg',img2band)
        
        
        
        
        