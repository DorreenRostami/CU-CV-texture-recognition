from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist
import cv2
import os

def computeHistogram(img_file, F, textons):

    ### YOUR CODE HERE
    img = io.imread(img_file, as_gray=True)

    img_responses = []
    for filter in F:
        filtered_img = cv2.filter2D(src=img, ddepth=-1, kernel=filter)
        img_responses.append(filtered_img.flatten())
    responses_arr = np.array(img_responses).T

    dist = np.sqrt(np.sum((responses_arr[:, None, :] - textons[None, :, :])**2, axis=2)) #shapes: response (22500, 1, 49), textons (1,50,49)
    close_idx = np.argmin(dist, axis=1) #indices of closest texton for each
    histogram, _ = np.histogram(close_idx, bins=len(textons))
    histogram_norm = histogram / np.sum(histogram) #normalize
    
    return histogram_norm
    ### END YOUR CODE
    
def createTextons(F, file_list, K):

    ### YOUR CODE HERE
    sample_pixels_count = 100
    all_responses = []

    for fname in file_list:
        img = io.imread(fname, as_gray=True)
        img_responses = []
        for filter in F:
            filtered_img = cv2.filter2D(src=img, ddepth=-1, kernel=filter)
            img_responses.append(filtered_img.flatten())
        responses_arr = np.array(img_responses).T  #shape is (49, img_h*img_w), transpose to get (h*w, 49)

        indices = np.random.choice(responses_arr.shape[0], sample_pixels_count, replace=False) #random 100 pixels
        sample = responses_arr[indices]
        all_responses.append(sample)

    vertically_stacked_responses = np.vstack(all_responses) #shape is (700, 49) since we have 7 images and 100 pixels were sampled from each

    os.environ["OMP_NUM_THREADS"] = "3" #KMeans is known to have a memory leak on Windows with MKL
    os.environ["MKL_NUM_THREADS"] = "3"
    
    kmeans = sklearn.cluster.KMeans(n_clusters=K, random_state=0)
    kmeans.fit(vertically_stacked_responses)
    textons = kmeans.cluster_centers_

    return textons
    ### END YOUR CODE
