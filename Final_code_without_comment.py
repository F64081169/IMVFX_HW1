"""
reference code: 
    1. knn mattng : https://github.com/MarcoForte/knn-matting.git
    2. KNN implementation: https://github.com/CassiaCai/Building-AI-coursework-Elements-of-AI/blob/main/ex16-nearest-neighbor
    3. knn balltree : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
    4. knn kdtree : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
    5. poisson matting : https://github.com/MarcoForte/poisson-matting/tree/master
    6. closed form matting : https://stackoverflow.com/questions/55353735/how-to-do-alpha-matting-in-python
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import cv2

from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
import multiprocessing as mp


# configurations
K = 10 # KNN K neighbors

def closed_form_laplacian(image, epsilon=1e-7, r=1):
    h,w = image.shape[:2]
    window_area = (2*r + 1)**2
    n_vals = (w - 2*r)*(h - 2*r)*window_area**2
    k = 0
    # data for matting laplacian in coordinate form
    i = np.empty(n_vals, dtype=np.int32)
    j = np.empty(n_vals, dtype=np.int32)
    v = np.empty(n_vals, dtype=np.float64)

    # for each pixel of image
    for y in range(r, h - r):
        for x in range(r, w - r):

            # gather neighbors of current pixel in 3x3 window
            n = image[y-r:y+r+1, x-r:x+r+1]
            u = np.zeros(3)
            for p in range(3):
                u[p] = n[:, :, p].mean()
            c = n - u

            # calculate covariance matrix over color channels
            cov = np.zeros((3, 3))
            for p in range(3):
                for q in range(3):
                    cov[p, q] = np.mean(c[:, :, p]*c[:, :, q])

            # calculate inverse covariance of window
            inv_cov = np.linalg.inv(cov + epsilon/window_area * np.eye(3))

            # for each pair ((xi, yi), (xj, yj)) in a 3x3 window
            for dyi in range(2*r + 1):
                for dxi in range(2*r + 1):
                    for dyj in range(2*r + 1):
                        for dxj in range(2*r + 1):
                            i[k] = (x + dxi - r) + (y + dyi - r)*w
                            j[k] = (x + dxj - r) + (y + dyj - r)*w
                            temp = c[dyi, dxi].dot(inv_cov).dot(c[dyj, dxj])
                            v[k] = (1.0 if (i[k] == j[k]) else 0.0) - (1 + temp)/window_area
                            k += 1
        print("generating matting laplacian", y - r + 1, "/", h - 2*r)

    return i, j, v

def make_system(L, trimap, constraint_factor=100.0):
    # split trimap into foreground, background, known and unknown masks
    is_fg = (trimap > 0.9).flatten()
    is_bg = (trimap < 0.1).flatten()
    is_known = is_fg | is_bg
    is_unknown = ~is_known

    # diagonal matrix to constrain known alpha values
    d = is_known.astype(np.float64)
    D = scipy.sparse.diags(d)

    # combine constraints and graph laplacian
    A = constraint_factor*D + L
    # constrained values of known alpha values
    b = constraint_factor*is_fg.astype(np.float64)

    return A, b
def closed_form_matting(image, trimap):
    i,j,v = closed_form_laplacian(image)
    h,w = trimap.shape
    L = scipy.sparse.csr_matrix((v, (i, j)), shape=(w*h, w*h))
    A, b = make_system(L, trimap)
    alpha = scipy.sparse.linalg.spsolve(A, b).reshape(h, w)
    return alpha

def computeAlphaJit(alpha, b, unknown):
    h, w = unknown.shape
    alphaNew = alpha.copy()
    alphaOld = np.zeros(alphaNew.shape)
    threshold = 0.1
    n = 1
    while (n < 50 and np.sum(np.abs(alphaNew - alphaOld)) > threshold):
        alphaOld = alphaNew.copy()
        for i in range(1, h-1):
            for j in range(1, w-1):
                if(unknown[i,j]):
                    alphaNew[i,j] = 1/4  * (alphaNew[i-1 ,j] + alphaNew[i,j-1] + alphaOld[i, j+1] + alphaOld[i+1,j] - b[i,j])
        n +=1
    return alphaNew

def poisson_matting(image, trimap):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape
    fg = trimap == 255
    bg = trimap == 0
    unknown = True ^ np.logical_or(fg,bg)
    fg_img = gray_img*fg
    bg_img = gray_img*bg
    alphaEstimate = fg + 0.5 * unknown

    approx_bg = cv2.inpaint(bg_img.astype(np.uint8),(unknown+fg).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)*(np.logical_not(fg)).astype(np.float32)
    approx_fg = cv2.inpaint(fg_img.astype(np.uint8),(unknown+bg).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)*(np.logical_not(bg)).astype(np.float32)

    # Smooth F - B image
    approx_diff = approx_fg - approx_bg
    approx_diff = scipy.ndimage.filters.gaussian_filter(approx_diff, 0.9)

    dy, dx = np.gradient(gray_img)
    d2y, _ = np.gradient(dy/approx_diff)
    _, d2x = np.gradient(dx/approx_diff)
    b = d2y + d2x

    alpha = computeAlphaJit(alphaEstimate, b, unknown)
    
    alpha = np.minimum(np.maximum(alpha,0),1).reshape(h,w)
    return alpha

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

def _knn_chunk(chunk_feature_vec):
  nbrs = NearestNeighbors(n_neighbors=K).fit(chunk_feature_vec)
  knns = nbrs.kneighbors(chunk_feature_vec)[1]
  return knns

def parallel_knn(feature_vec, n_jobs=4):

    n_rows = feature_vec.shape[0]
    chunksize = int(n_rows / n_jobs) + 1

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(_knn_chunk, np.array_split(feature_vec, n_jobs))

    knns = np.concatenate(results)
    return knns

def knn_matting(image, trimap, my_lambda=100, knn_method='sklearn'):
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
    if knn_method == 'custom':
        knns = custom_knn(image, K)
    elif knn_method == 'sklearn':
        nbrs = NearestNeighbors(n_neighbors=K, n_jobs=4).fit(feature_vec)
        knns = nbrs.kneighbors(feature_vec)[1]
    elif knn_method == 'kdtree':
        knns = custom_knn_kdtree(image, K)
    elif knn_method == 'balltree':
        knns = custom_knn_balltree(image, K)
    elif knn_method == 'parallel':
        knns = parallel_knn(feature_vec, n_jobs=4)
    else:
        nbrs = NearestNeighbors(n_neighbors=K, n_jobs=4).fit(feature_vec)
        knns = nbrs.kneighbors(feature_vec)[1]
    
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
        ## Exact solution
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(h, w)
        ## Approximate solution
        # eigvals, eigvecs = scipy.sparse.linalg.eigsh(H, 1)
        # max_eigval = eigvals[-1]
        # alpha = np.minimum(np.maximum(c / max_eigval, 0), 1).reshape(h, w)
        pass
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(h, w)
        pass

    return alpha


if __name__ == '__main__':
    img_dir = ['bear','gandalf','woman']
    methods = ['knn_matting']#['knn_matting', 'closed_form_matting', 'poisson_matting']
    knn_methods = ['sklearn']#['sklearn', 'kdtree', 'balltree', 'parallel']
    k_params = [5, 10, 20, 30, 40, 50, 100]
    for img in img_dir:
        # for method in methods:
        for method in methods:
            image = cv2.imread('./image/'+img+'.png')
            trimap = cv2.imread('./trimap/'+img+'.png')
            start_time = cv2.getTickCount()
            print('Processing image: ', img,method)
            if method == 'knn_matting':
                alpha = knn_matting(image, trimap)
            elif methods == 'closed_form_matting':
                alpha = closed_form_matting(image, trimap)
            elif methods == 'poisson_matting':
                alpha = poisson_matting(image, trimap)
            else:
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
            cv2.imwrite('./result/'+img+'_'+method+'.png', result)
            end_time = cv2.getTickCount()
            print(img,method, 'finished')
            print('Time elapsed: ', (end_time - start_time) / cv2.getTickFrequency(), 's')
            print('---------------------------')
