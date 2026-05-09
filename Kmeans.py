__authors__ = ["1707361"]
__group__ = '87'

import numpy as np
import utils
import random


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        X = np.array(X)
        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1]).astype(float)
        self.X = X.astype(float)
        
        
            

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        if self.options['km_init'] == 'first':
            centroids = []
            for points in self.X:
                added = False
                for c in centroids:
                    if np.array_equal(points, c):
                        added = True
                if not added:
                    centroids.append(points)
                if len(centroids) >= self.K:
                    break
            self.centroids = np.array(centroids)
            
        elif self.options['km_init'] == 'random':
            self.centroids = np.array(random.sample(list(self.X), self.K))

        elif self.options['km_init'].lower() == 'custom':
            min_vals = np.min(self.X, axis=0)
            max_vals = np.max(self.X, axis=0)
            self.centroids = np.linspace(min_vals, max_vals, self.K)

        self.old_centroids = np.zeros_like(self.centroids)



    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distances = distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = self.centroids.copy()

        auxCentroids = np.zeros((self.K, self.X.shape[-1]))
        numPoints = np.zeros(self.K)
        for i, points in enumerate(self.labels):
            auxCentroids[points] += self.X[i]
            numPoints[points] += 1
        
        numPoints[numPoints == 0] = 1
        self.centroids = auxCentroids / numPoints[:, None]
        


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        diff = np.array(self.centroids - self.old_centroids)
        return np.all(np.linalg.norm(diff, axis=1) <= self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()
        self.num_iter = 0
    
        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1

            if self.converges():
                break

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
  
        dist = distance(self.X, self.centroids)
        min_dist = np.min(dist, axis=1)
        return np.mean(min_dist ** 2) 
    
        


    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        km = KMeans(self.X, 1, self.options)
        km.fit()
        prevWCD = km.withinClassDistance()
        found = False
        for i in range(max_K - 1):
            km = KMeans(self.X, i + 2, self.options)
            km.fit()
            actualWCD = km.withinClassDistance()
            if (100 - 100 * actualWCD / prevWCD < 20):
                found = True
                self.K = i + 1
                break
            prevWCD = actualWCD

        if not found: 
            self.K = max_K



def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    
    return np.linalg.norm(X[:, None] - C[None, :], axis=2)
            


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    probs = utils.get_color_prob(centroids)
    index = np.argmax(probs, axis=1)
    colors = [utils.colors[i] for i in index]
    return colors
