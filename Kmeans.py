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
            for points in self.X: #iterates over all the points, in order, if that point its not on the centroids list add it, otherwise ignore
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
            self.centroids = np.array(random.sample(list(self.X), self.K)) #takes random points for init centroids

        elif self.options['km_init'].lower() == 'custom':
            min_vals = np.min(self.X, axis=0)
            max_vals = np.max(self.X, axis=0)
            self.centroids = np.linspace(min_vals, max_vals, self.K)  #inits centroids equally distributed over the sample
        
        elif self.options['km_init'].lower() == 'custom2':
            #we will use kmeans++
            centroids = []
            num_samples = self.X.shape[0]
            centroids.append(self.X[np.random.randint(num_samples)]) #choose the first random point

            for i in range(1, self.K):
                distances = distance(self.X, np.array(centroids)) #calc distance from every point to the centroid
                min_distance = np.min(distances, axis = 1) #returns the distance of every point to their neareast centroid

                probs = min_distance ** 2 
                probs = probs / np.sum(probs) #gives a weight to every point, depending on their distance

                centroids.append(self.X[np.random.choice(num_samples, p=probs)]) #chooses next centroid randomly, but given preference
                #to farther points
            self.centroids = np.array(centroids)

        self.old_centroids = np.zeros_like(self.centroids)



    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distances = distance(self.X, self.centroids) #calc every point distance to the centroids
        self.labels = np.argmin(distances, axis=1) #get the arg (label) of every point by the neareast centroid
        #returns a list of labels, each point has a label, for example, self.labels[0] is the label of self.X[0] and so on

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
        for i, points in enumerate(self.labels): #iterates over every label and sums the points location and number of points of each label
            auxCentroids[points] += self.X[i] #auxCentroids stores for every label auxCentroids[points] the sum of their points
            numPoints[points] += 1 #numPoints stores for every label the number of points
        
        numPoints[numPoints == 0] = 1
        self.centroids = auxCentroids / numPoints[:, None] #then, calculate the average to get the new centroids
        


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        diff = np.array(self.centroids - self.old_centroids)
        return np.all(np.linalg.norm(diff, axis=1) <= self.options['tolerance']) #if the change of centroids position from prev iteration is lower than a tolerance we can say its done.

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
    
        while self.num_iter < self.options['max_iter']: #maing algorithm of KMEANS
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
  
        dist = distance(self.X, self.centroids) #get distance of every point to every centroid
        min_dist = np.min(dist, axis=1) #get the lowest distance for every point (which is the distance of that point to his class centroid)
        return np.mean(min_dist ** 2) #calc mean of the distance squared so we get WCD.
    
    def interClassDistanceCentroids(self):
        dist = distance(self.centroids, self.centroids) #calcu dist from every centroid to every centroid
        return np.mean(np.triu(dist, k=1)) #eliminate everything below the diagonal of the matrix, since its symmetric
    
    def interClassDistancePoints(self):
        dist = distance(self.X, self.X)
        total = 0
        count = 0

        for i in range(len(self.X)):
            for j in range(i+1, len(self.X)):
                if self.labels[i] != self.labels[j]:
                    total += dist[i][j]
                    count += 1
        return total/count

    def fisherDiscriminant(self):
        return self.withinClassDistance()/self.interClassDistanceCentroids()

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
        for i in range(max_K - 1): #do Kmeans with diferent values of K until we find, by a drop of WCD parameter, which is the best K.
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
    return colors #given a list of centroids, returns another list but with the colours associated to each centroid
