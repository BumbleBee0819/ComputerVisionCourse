# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:39:14 2015

@author: Wenyan Bi
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import math as e 

####################  filter #############################
def imfilter(img,filterm):
    #read size of the filter matrix
    m, n = filterm.shape
    pixels=img.load()    
    
    newimg1=Image.new( 'RGB', (img.size[0]+m,img.size[1]+n), "white")
    pixels1=newimg1.load()

    newimg2=Image.new( 'RGB', (img.size[0],img.size[1]), "white")
    pixels2=newimg2.load()
    
    #zero padding
    for i in range(img.size[0]+m):    # for every pixel:
        for j in range(img.size[1]+n):
            if (i<m/2 or j<n/2 or i>=img.size[0]+m/2 or j>=img.size[1]+n/2):
                pixels1[i,j]=0
            
            elif ((i>=m/2 and j>=n/2 and i<img.size[0]+m/2 and j<img.size[1]+n/2)):
                pixels1[i,j]=pixels[i-m/2,j-n/2]
           

    #misc.imsave('zerobird.jpg',newimg1)
    #plt.imshow(newimg1)

    #convolve
    for i in range(m/2,m/2+img.size[0]):    
        for j in range(n/2,n/2+img.size[1]):
            sumr=0
            sumg=0
            sumb=0
            for k in range(m):
                for l in range(n):
                    r, g, b = pixels1[i-m/2+k,j-n/2+l]
                    sumr+=r*filterm[k,l]
                    sumg+=g*filterm[k,l]
                    sumb+=b*filterm[k,l]
        
            pixels2[i-m/2,j-n/2]=int(sumr),int(sumg),int(sumb)

    #misc.imsave('filterbird.jpg',newimg2)
    #plt.imshow(newimg2)
    return newimg2,pixels2



################### gaussian filter  ###################
def gauss(shape,sigma):
    
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h=h/sumh
    return h
    
    



################  input image    #########################
img1=Image.open('1 einstein.bmp')
img2=Image.open('1 marilyn.bmp')
img3=Image.open('2 cat.bmp')
img4=Image.open('2 dog.bmp')
img5=Image.open('3 plane.bmp')
img6=Image.open('3 bird.bmp')
img7=Image.open('4 fish.bmp')
img8=Image.open('4 submarine.bmp')
img9=Image.open('5 bicycle.bmp')
img10=Image.open('5 motorcycle.bmp')



###########   define parameter  ########################

## 1) for marilyn & einstein 
## sigma_low=7,  m=n=35
## sigma_high=1, m=n=35

#m=n=35
#filterlow=gauss((m,n),7)
#m=n=35
#filterhigh=np.zeros(shape=(m,n))
#filterhigh[m/2,n/2]=1                    
#filterhigh=filterhigh-gauss((m,n),1)




## 2) for cat & dog
## sigma_low=11,  m=n=25
## sigma_high=7, m=n=25

#m=n=25
#filterlow=gauss((m,n),11)
#m=n=25
#filterhigh=np.zeros(shape=(m,n))
#filterhigh[m/2,n/2]=1                    
#filterhigh=filterhigh-gauss((m,n),7)




## 3) for plane & bird
## sigma_low=11,  m=n=25
## sigma_high=5, m=n=25

# low filtering
m=n=25
filterlow=gauss((m,n),11)
# high filtering
m=n=25
filterhigh=np.zeros(shape=(m,n))
filterhigh[m/2,n/2]=1                    
filterhigh=filterhigh-gauss((m,n),5)





## 4) for fish & submarine
## sigma_low=9,  m=n=25
## sigma_high=3, m=n=25

#m=n=25
#filterlow=gauss((m,n),9)
#m=n=25
#filterhigh=np.zeros(shape=(m,n))
#filterhigh[m/2,n/2]=1                    
#filterhigh=filterhigh-gauss((m,n),3)




## 5) for bicycle & motorcycle
## sigma_low=15,  m=n=41
## sigma_high=3, m=n=31

#m=n=41
#filterlow=gauss((m,n),15)
#m=n=31
#filterhigh=np.zeros(shape=(m,n))
#filterhigh[m/2,n/2]=1                    
#filterhigh=filterhigh-gauss((m,n),3)



##########    high&low filtering   ###########################

highfreimg,pixelh=imfilter(img5, filterhigh)   ###change with different images!!!!!!
lowfreimg,pixell=imfilter(img6, filterlow)


###############  combine two images  ###########################


finalimg=Image.new( 'RGB', (img5.size[0],img5.size[1]), "white") ### change with different images!!!!!!
pixel=finalimg.load()


for i in range(img5.size[0]):
    for j in range(img5.size[1]):    #######change with different images !!!!!!
        r1,g1,b1=pixelh[i,j]
        r2,g2,b2=pixell[i,j]
        q,w,t=pixelh[i,j]
        pixelh[i,j]=q+100,w+100,t+100
        #print pixelh[i,j]
        pixel[i,j]=int(r1+r2),int(g1+g2),int(b1+b2)




#################    FFT magnitude   ###########################

fh1 = np.fft.fft2(img1.convert('L'))
fshifth1 = np.fft.fftshift(fh1)
magnitudeh1 = 30*np.log(np.abs(fshifth1))
print fshifth1.max(),fshifth1.min()
plt.imshow(magnitudeh1)
plt.title('magnitude FFT high pass_before filtering')
plt.show()

fh = np.fft.fft2(highfreimg.convert('L'))
fshifth = np.fft.fftshift(fh)
magnitudeh = 30*np.log(np.abs(fshifth))
print fshifth.max(),fshifth.min()
plt.imshow(magnitudeh)
plt.title('magnitude FFT high pass')
plt.show()

fl1 = np.fft.fft2(img2.convert('L'))
fshiftl1 = np.fft.fftshift(fl1)
magnitudel1 = 30*np.log(np.abs(fshiftl1))
print fshiftl1.max(),fshiftl1.min()
plt.imshow(magnitudel1)
plt.title('magnitude FFT low pass_before filtering')
plt.show()

fl = np.fft.fft2(lowfreimg.convert('L'))
fshiftl = np.fft.fftshift(fl)
magnitudel = 30*np.log(np.abs(fshiftl))
print fshiftl.max(),fshiftl.min()
plt.imshow(magnitudel)
plt.title('magnitude FFT low pass')
plt.show()


f = np.fft.fft2(finalimg.convert('L'))
fshift = np.fft.fftshift(f)
magnitude = 30*np.log(np.abs(fshift))
print fshift.max(),fshift.min()
plt.imshow(magnitude)
plt.title('magnitude FFT hybrid image')
plt.show()






##############   save &  plot    #################################

misc.imsave('final.jpg',finalimg)
misc.imsave('fina1high.jpg', highfreimg)
misc.imsave('finallow.jpg',lowfreimg)

plt.imshow(highfreimg)
plt.title('high frequency image')
plt.show()
plt.imshow(lowfreimg)
plt.title('low frequency image')
plt.show()
plt.imshow(finalimg)
plt.title('hybrid image')
plt.show()





################### cut-off frequency_magnitude  #################################

#######  print Fgausslow

m=n=25   #change with different images!!!!!!!!!!!
filterlow=gauss((m,n),11)   ###### low sigma: change with different images!!!!!!!!!
Fgausslow=np.fft.fft2(filterlow)
Fgausslow=np.fft.fftshift(Fgausslow)
rows, cols = Fgausslow.shape
crow, ccol = rows/2 , cols/2


x=[i-m/2 for i in range(crow, rows)]
y=[abs(Fgausslow[i,ccol]/Fgausslow[crow,ccol]) for i in range(crow, rows)] 
plt.scatter(x,y)   

x=np.arange(0,17,0.01)
y=np.exp( -(x*x) / (2.*0.36*0.36) )    # change with different images!!!!!!!!!
### kearnal size/(sigma*2pi)
#pair 1: y=np.exp( -(x*x) / (2.*0.8*0.8) ) 
#pair 2: y=np.exp( -(x*x) / (2.*0.36*0.36) )
#pair 3: y=np.exp( -(x*x) / (2.*0.36*0.36) )
#pair 4: y=np.exp( -(x*x) / (2.*0.44*0.44) )
#pair 5: y=np.exp( -(x*x) / (2.*0.44*0.44) )
plt.plot(x,y)



#####  print Fgausshigh

m=n=25             ############ low sigma: change with different images!!!!!
filterhigh=np.zeros(shape=(m,n))
filterhigh[m/2,n/2]=1                   
filterhigh=filterhigh-gauss((m,n),5)    ############ change with different images!!!!!

Fgausshigh=np.fft.fft2(filterhigh)
Fgausshigh=np.fft.fftshift(Fgausshigh)
rows, cols = Fgausshigh.shape
crow, ccol = rows/2 , cols/2


x=[i-crow for i in range(crow, rows)]
y=[abs((Fgausshigh[i,ccol])/Fgausshigh[rows-1,ccol]) for i in range(crow, rows)] 
plt.scatter(x,y)   

x=np.arange(0,17,0.01)
y=1-np.exp( -(x*x) / (2.*0.8*0.8) ) 
 #### kearnal size/(sigma*2pi)
#pair 1: y=1-np.exp( -(x*x) / (2.*5.57*5.57) ) 
#pair 2: y=1-np.exp( -(x*x) / (2.*0.57*0.57) ) 
#pair 3: y=1-np.exp( -(x*x) / (2.*0.8*0.8) ) 
#pair 4: y=1-np.exp( -(x*x) / (2.*1.33*1.33) ) 
#pair 5: y=1-np.exp( -(x*x) / (2.*1.65*1.65) ) 
plt.plot(x,y)


plt.title('filters in frequency domain')
plt.ylabel('magnitude')
plt.xlabel('frequency')
plt.show()



