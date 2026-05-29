__authors__ = ["1707361"]
__group__ = '87'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        self.train_data = self._init_train(train_data)
        self.neighbors = None

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
        return train_data.reshape(P, -1).astype(float)
        

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
        
        list_top_k = [] #list to store top k labels for each object
        dist = cdist(test_data_reshaped, self.train_data, metric='euclidean') #calculate euclidean dist to every train point from every test point
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

        self.get_k_neighbours(test_data, k)
        return self.get_class_prob(k)
