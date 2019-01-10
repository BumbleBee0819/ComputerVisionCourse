# calling the harris corner detector
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