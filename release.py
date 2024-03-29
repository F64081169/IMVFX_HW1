# reference : https://github.com/MarcoForte/knn-matting.git
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import cv2


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
    nbrs = NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]
    ####################################################
    # TODO: compute the affinity matrix A
    #       and all other stuff needed
    ####################################################
    row_inds = np.repeat(np.arange(h*w), 10)
    col_inds = knns.reshape(h*w*10)
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

    image = cv2.imread('./image/woman.png')
    trimap = cv2.imread('./trimap/woman.png')

    alpha = knn_matting(image, trimap)
    alpha = alpha[:, :, np.newaxis]

    ####################################################
    # TODO: pick up your own background image, 
    #       and merge it with the foreground
    ####################################################
    background = cv2.imread('./background/woman.png')
    background = cv2.resize(background, (image.shape[1], image.shape[0]))

    result = []
    result = image * alpha + (1 - alpha) * background
    cv2.imwrite('./result/woman.png', result)
