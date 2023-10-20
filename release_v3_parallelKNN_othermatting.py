# reference : https://github.com/MarcoForte/knn-matting.git
# https://github.com/CassiaCai/Building-AI-coursework-Elements-of-AI/blob/main/ex16-nearest-neighbor
# https://github.com/MarcoForte/closed-form-matting.git
# 
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import cv2

from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from scipy.fft import fft2, ifft2
from scipy.sparse.linalg import cg
import multiprocessing as mp


# configurations
K = 10 # KNN K neighbors
KNN_METHOD = 'parallel' # 'custom', 'sklearn', 'kdtree', 'balltree', 'parallel',default=sklearn
MATTING_METHOD = 'knn_matting' # 'knn_matting', 'closed_form_matting', 'poisson_matting',default=knn_matting

def closed_form_matting(image, trimap):

  h, w, c = image.shape
  A = scipy.sparse.lil_matrix((h*w, h*w))  
  for y in range(h):
    for x in range(w):
      if x > 0:  
        A[y*w + x, y*w + x - 1] = -1
      if y > 0:
        A[y*w + x, (y-1)*w + x] = -1
      A[y*w + x, y*w + x] = 4
  
  D = scipy.sparse.lil_matrix((h*w, h*w))
  D.setdiag(np.ravel(trimap[:,:,0]))
  
  laplacian = scipy.sparse.csr_matrix(A)
  alpha = np.minimum(np.maximum(spsolve(laplacian + D), 0), 1).reshape(h,w)
  
  return alpha

def poisson_matting(image, trimap):

  h, w, c = image.shape
  
  indicies = [(0,1), (-w,w), (-1,0), (w,-w)]
  regular_indicies = [(0, 0)]
  regular_vals = [-4 for i in range(h*w)]
  laplacian_vals = [1 for i in range(4*h*w)]

  mat_indicies = np.array(regular_indicies + indicies * h * w)
  mat_vals = np.array(regular_vals + laplacian_vals)

  L = coo_matrix((mat_vals, mat_indicies), shape=(h*w, h*w)).tocsr()
  
  foreground = (trimap > 0).ravel()
  background = (trimap == 0).ravel() 
  alpha = np.zeros((h*w))
  alpha[foreground] = 1
  alpha[background] = 0
  
  f = fft2(1 - alpha)
  x = cg(L, ifft2(f).real)[0]
  alpha = np.clip(x.reshape(h, w), 0, 1)  

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
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(h, w)
        pass
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(h, w)
        pass

    return alpha


if __name__ == '__main__':
    img_dir = ['bear','gandalf','woman']
    methods = ['knn_matting', 'closed_form_matting', 'poisson_matting']
    knn_methods = ['sklearn', 'kdtree', 'balltree', 'parallel']
    k_params = [5, 10, 20, 30, 40, 50, 100]
    for img in img_dir:
        # for method in methods:
        for method in knn_methods:
            image = cv2.imread('./image/'+img+'.png')
            trimap = cv2.imread('./trimap/'+img+'.png')
            start_time = cv2.getTickCount()
            print('Processing image: ', img,method)
            if method == 'knn_matting':
                alpha = knn_matting(image, trimap,method=method)
            elif methods == 'closed_form_matting':
                alpha = closed_form_matting(image, trimap)
            elif methods == 'poisson_matting':
                alpha = poisson_matting(image, trimap,method=method)
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
