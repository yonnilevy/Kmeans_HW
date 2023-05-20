import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    random_indicies =  np.random.choice(X.shape[0], size = k, replace = False)
    centroids = X[random_indicies, :]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    temp = X[:,None,:] - centroids[None,:,:]
    temp = np.abs(power(temp,p))
    temp = np.sum(temp, axis =2)
    distances = (temp**(1/p)).T
    #print(distances.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances


def power(x,p):
    res = 1
    for i in range(p):
       res *= x
    return res


def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    #print(centroids)
    new_centroids = np.zeros((k,3))
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        #print(distances)
        classes = np.argmin(distances, axis = 0)
        for i in range(k):
            cluster_i = X[classes == i]
            new_centroids[i,:] = np.mean(cluster_i, axis = 0)
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = np.zeros((k,3))
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    i = np.random.choice(X.shape[0], size=1, replace=False)
    centroids[0,:] = X[i,:]
    new_X = X.copy()
    for j in range(k-1):
      new_X = np.delete(new_X, (i), axis = 0)
      distances = lp_distance(new_X, centroids,p)
      min_distances = np.min(distances, axis = 0)
      probs = min_distances / min_distances.sum()
      i = np.random.choice(new_X.shape[0], size=1, replace=False, p=probs)
      centroids[j+1,:] = new_X[i, :]


    new_centroids = np.zeros((k,3))

    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        #print(distances)
        classes = np.argmin(distances, axis = 0)
        for i in range(k):
            cluster_i = X[classes == i]
            new_centroids[i,:] = np.mean(cluster_i, axis = 0)
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

