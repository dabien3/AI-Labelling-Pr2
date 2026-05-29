__authors__ = ["1707361"]
__group__ = '87'

import numpy as np
import cv2
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels, options=None):
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        self._init_options(options)
        self.train_data = self._init_train(train_data)
        self.neighbors = None

    def _init_options(self, options=None):
        if options is None:
            options = {}
        if 'knn_size_data' not in options:
            options['knn_size_data'] = 'default'
        if "f_space" not in options:
            options['f_space'] = 'default'
        if "quadrants" not in options:
            options['quadrants'] = 'default'
        if "dist" not in options:
            options["dist"] = 'euclidean'
        self.options = options

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        P = train_data.shape[0]
        features = []

        # RESIZE 

        if self.options["knn_size_data"] != "default":
            w, h = self.options["knn_size_data"]

            resized = []

            for img in train_data:
                img_resized = cv2.resize(img, (w, h)) #resize baja la resolucion imagen
                resized.append(img_resized) 

            train_data = np.array(resized) #matriz de P imagenes, con w x h features

        quadrants = self.options["quadrants"]        # full | quadrants
        feature = self.options['f_space']    # raw | mean | mean_var


        # QUADRANTS + DEFINE FEATURE SPACE

        for img in train_data:

            # DEFINE QUADRANTS
            if quadrants != "default":
                regions = []
                n = self.options["quadrants"]
                h, w = img.shape[:2]
                step_h = h // n
                step_w = w // n

                for i in range(n): #para cada imagen lo dividimos en n regiones
                    for j in range(n):

                        region = img[
                            i * step_h : (i + 1) * step_h,
                            j * step_w : (j + 1) * step_w
                        ]

                        regions.append(region)
            else:
                regions = [img] #si no se quiere cuadrantes simplemente pasamos la imagen

            # GET FEATURES (RAW, MEAN, OR VARIANCE)
            vec = []

            for r in regions:

                # RAW
                if feature == "default":
                    vec.extend(r.reshape(-1)) #los features por default es la escala de grises 

                # MEAN
                elif feature == "mean":
                    vec.append(np.mean(r)) #para unificar los pixeles de una region hacemos el promedio

                # MEAN + VAR
                elif feature == "mean_var": #para cada region calcula su media y varianza
                    vec.append(np.mean(r))
                    vec.append(np.var(r))

            features.append(vec)

        return np.array(features).astype(float)
            

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        N = test_data.shape[0]
        test_data_reshaped = test_data.reshape(N, -1).astype(float)
        metric = self.options["dist"]
        
        list_top_k = [] #list to store top k labels for each object
        dist = cdist(test_data_reshaped, self.train_data, metric=metric) #calculate euclidean dist to every train point from every test point
        for row in dist:
            top_k_index = np.argsort(row)[:k] #get indexs from top k lower distances for each test point
            top_k_labels = [self.labels[i] for i in top_k_index] #get the equivalent label of each index
            list_top_k.append(top_k_labels) #append top k labels from each point to the matrix
        self.neighbors = np.array(list_top_k)


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        topValues = []
        for row in self.neighbors:
            valueDic = {} #dict to store how many each value appears for very obj in test_data
            for value in row:
                if value in valueDic:
                    valueDic[value]+=1 #if the value already appeared we increase counter by one
                else:
                    valueDic[value] = 1 #otherwise the counter is set to one
            key_max = max(valueDic, key=valueDic.get) #we get the value with the max counter
            topValues.append(key_max) #we append the value (label) with the max counter for each obj
        return np.array(topValues)
    
    def get_class_prob(self, k):
    #same as get_class but we also return the percentage of the most voted class for each object in test_data
        topValues = []
        topProbs  = []
        for row in self.neighbors:
            valueDic = {}
            for value in row:
                if value in valueDic:
                    valueDic[value] += 1
                else:
                    valueDic[value] = 1

            key_max = max(valueDic, key=valueDic.get)
            porcentaje = valueDic[key_max] / k
            topValues.append(key_max)
            topProbs.append(porcentaje)

        return np.array(topValues), np.array(topProbs)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
        test_data = self._init_train(test_data)
        self.get_k_neighbours(test_data, k)
        return self.get_class_prob(k)
