# reference : https://github.com/MarcoForte/knn-matting.git
# https://github.com/CassiaCai/Building-AI-coursework-Elements-of-AI/blob/main/ex16-nearest-neighbor
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree

# configurations
K = 10 # KNN K neighbors
KNN_METHOD = 'kdtree' # 'custom', 'sklearn', 'kdtree', 'balltree'

def custom_knn(image, k):
    h, w, c = image.shape
    a, b = np.unravel_index(np.arange(h*w), (h, w))
    feature_vec = np.append(np.transpose(image.reshape(h*w,c)), [ a, b]/np.sqrt(h*h + w*w), axis=0).T
    
    knns = np.zeros((h * w, k), dtype=int)
    
    for i in range(h * w):
        distances = np.sqrt(np.sum((feature_vec - feature_vec[i])**2, axis=1))
        nearest_indices = np.argsort(distances)
        knns[i] = nearest_indices[1:k+1]  
    
    return knns

def custom_knn_kdtree(image, k):
    h, w, c = image.shape
    a, b = np.unravel_index(np.arange(h*w), (h, w))
    feature_vec = np.append(np.transpose(image.reshape(h*w,c)), [ a, b]/np.sqrt(h*h + w*w), axis=0).T
    
    tree = KDTree(feature_vec)
    
    dist, ind = tree.query(feature_vec, k=k+1)
    
    knns = ind[:, 1:]
    
    return knns

def custom_knn_balltree(image, k):
    h, w, c = image.shape
    a, b = np.unravel_index(np.arange(h*w), (h, w))
    feature_vec = np.append(np.transpose(image.reshape(h*w,c)), [ a, b]/np.sqrt(h*h + w*w), axis=0).T
    
    tree = BallTree(feature_vec)
    
    dist, ind = tree.query(feature_vec, k=k+1)
    
    knns = ind[:, 1:]
    
    return knns

def knn_matting(image, trimap, my_lambda=100):
    [h, w, c] = image.shape
    image, trimap = image / 255.0, trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)
    all_constraints = foreground + background
    ####################################################
    # TODO: find KNN for the given image
    ####################################################
    a, b = np.unravel_index(np.arange(h*w), (h, w))
    feature_vec = np.append(np.transpose(image.reshape(h*w,c)), [ a, b]/np.sqrt(h*h + w*w), axis=0).T
    if KNN_METHOD == 'custom':
        knns = custom_knn(image, K)
    elif KNN_METHOD == 'sklearn':
        nbrs = NearestNeighbors(n_neighbors=K, n_jobs=4).fit(feature_vec)
        knns = nbrs.kneighbors(feature_vec)[1]
    elif KNN_METHOD == 'kdtree':
        knns = custom_knn_kdtree(image, K)
    elif KNN_METHOD == 'balltree':
        knns = custom_knn_balltree(image, K)
    
    ####################################################
    # TODO: compute the affinity matrix A
    #       and all other stuff needed
    ####################################################
    row_inds = np.repeat(np.arange(h*w), K)
    col_inds = knns.reshape(h*w*K)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1)/(c+2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)),shape=(h*w, h*w))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script-A
    D = scipy.sparse.diags(np.ravel(all_constraints[:, :, 0]))
    v = np.ravel(foreground[:,:,0])
    c = 2*my_lambda*np.transpose(v)
    H = 2*(L + my_lambda*D)
    ####################################################
    # TODO: solve for the linear system,
    #       note that you may encounter en error
    #       if no exact solution exists
    ####################################################

    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(h, w)
        pass
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(h, w)
        pass

    return alpha


if __name__ == '__main__':
    img_dir = ['bear','gandalf','woman']

    for img in img_dir:
        image = cv2.imread('./image/'+img+'.png')
        trimap = cv2.imread('./trimap/'+img+'.png')
        print('Processing image: ', img)
        alpha = knn_matting(image, trimap)
        alpha = alpha[:, :, np.newaxis]

        ####################################################
        # TODO: pick up your own background image, 
        #       and merge it with the foreground
        ####################################################
        background = cv2.imread('./background/'+img+'.png')
        background = cv2.resize(background, (image.shape[1], image.shape[0]))

        result = []
        result = image * alpha + (1 - alpha) * background
        cv2.imwrite('./result/'+img+'.png', result)
        print(img, 'finished')
