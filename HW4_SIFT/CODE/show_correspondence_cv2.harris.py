import cv2
import numpy as np
import scipy.io
from matplotlib import pyplot as plt


def show_correspondence(image1, image2, X1, Y1, X2, Y2, matches,ratio):
    fig = plt.figure(figsize=(18,14))

	#mngr = plt.get_current_fig_manager()
	# to put it into the upper left corner for example:
	#mngr.window.setGeometry(50,100,640, 545)
    #plt.figure(figsize=(18,14))
    plt.subplot(1,2,1)
    plt.imshow(image1,cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(image2,cmap='gray')
    plt.xticks([]), plt.yticks([])
    
    j=0
    for i in range(len(X1)):
        cur_color = np.random.rand(3,1)
		#print X1[i]
        if ratio[i]<=0.8 and ratio[i]>0:
            plt.subplot(1,2,1)
            plt.plot(X1[matches[i,0]],Y1[matches[i,0]], marker='o', ms=12, mec = 'k', mfc=cur_color,lw=2.0)
            #plt.plot(Y1[i],X1[i], marker='o', ms=4, mec = 'k', mfc=cur_color,lw=2.0)
            plt.subplot(1,2,2)
            plt.plot(X2[matches[i,1]],Y2[matches[i,1]], marker='o', ms=12, mec = 'k', mfc=cur_color,lw=2.0)
            #plt.plot(Y2[i],X2[i], marker='o', ms=4, mec = 'k', mfc=cur_color,lw=2.0)
            j=j+1
            
		# #cur_color = np.random.randint(255, size=3)
		    

    fig.savefig('vis3.jpg')
    print j
    print 'Saving visualization to vis.jpg'
	
    return fig





