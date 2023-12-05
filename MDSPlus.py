# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import matplotlib.pyplot as plt


#To use this class, first instantiate it and add a distance matrix. Then, find
#the pq embedding. After that, the mds and mdsplus methods need to be run. Then
# all analysis methods can be run.
class mds():
    
    #Listing all class variables
    distance_matrix = None
    eigenvalues = None
    pq_embedding = None
    rhos = None
    mdsp_pos_vecs = None
    mdsp_neg_vecs = None
    mds_vecs = None
    p = 0
    q = 0
    mdsp_distance_matrix = None
    mds_distance_matrix = None
    mds_distorts = []
    mdsp_distorts = []
    
    #Constructor with the option to set distance matrix later
    def __init__(self, distance_matrix = None):
        self.distance_matrix = distance_matrix
        return
    
    def calculate_rho(self, verbose = True):
        n = len(self.distance_matrix)
        self.rhos = np.zeros((n, n))
        for i in range(n):
            if verbose:
                if i % (n // 100) == 0:
                    print(i // (n // 100))
            for j in range(i + 1, n):
                difference = self.pq_embedding[i] - self.pq_embedding[j]
                magnitude = np.linalg.norm(difference)
                rho = self.distance_matrix[i][j] / magnitude
                self.rhos[i][j] = rho
                self.rhos[j][i] = rho
        return
    
    def graph_eigenvalues(self):
        fig, ax = plt.subplots(2)
        ax[0].hist(self.eigenvalues, np.arange(np.floor(np.min(self.eigenvalues)),
                                               np.ceil(np.max(self.eigenvalues)),1), log=True)
        plt.show()
    
    def mdsp_distance_matrix(self, verbose = True):
        n = len(self.distance_matrix)
        self.mdsp_distance_matrix = np.zeros((n, n))
        for i in range(n):
            if verbose:
                if i % (n // 100) == 0:
                    print(i // (n // 100))
            for j in range(i+1, n):
                mdsp_distance = np.linalg.norm(self.mdsp_pos_vecs[j] - self.mdsp_pos_vecs[i])**2
                mdsp_distance -= np.linalg.norm(self.mdsp_neg_vecs[j] - self.mdsp_neg_vecs[i])**2
                if mdsp_distance < 0:
                    mdsp_distance = -np.sqrt(np.absolute(mdsp_distance))
                else:
                    mdsp_distance = np.sqrt(mdsp_distance)
                self.mdsp_distance_matrix[i][j] = mdsp_distance
                self.mdsp_distance_matrix[j][i] = mdsp_distance
    
    def mds_distance_matrix(self, verbose = True):
        n = len(self.distance_matrix)
        self.mds_distance_matrix = np.zeros((n, n))
        for i in range(n):
            if verbose:
                if i % (n // 100) == 0:
                    print(i // (n // 100))
            for j in range(i+1, n):
                mds_distance = np.linalg.norm(self.mds_vecs[j] - self.mds_vecs[i])
                self.mds_distance_matrix[i][j] = mds_distance
                self.mds_distance_matrix[j][i] = mds_distance
    
    def mult_distortion_analysis(self):
        n = len(self.distance_matrix)
        mdsp_distorts =[]
        mds_distorts = []
        num_negative = 0
        for i in range(n):
            for j in range(i+1, n):
                if self.mdsp_distance_matrix[j][i] < 0:
                    num_negative += 1
                    continue
                mdsp_distorts.append(self.distance_matrix[j][i]/self.mdsp_distance_matrix[j][i])
                mds_distorts.append(self.distance_matrix[j][i]/self.mds_distance_matrix[j][i])
        mdsp_median = np.median(mdsp_distorts)
        mds_median = np.median(mds_distorts)
        for i in range(len(mdsp_distorts)):
            mdsp_distorts[i] /= mdsp_median
            if mdsp_distorts[i] < 1:
                mdsp_distorts[i] = 1.0/mdsp_distorts[i]
        for i in range(len(mds_distorts)):
            mds_distorts[i] /= mds_median
            if mds_distorts[i] < 1:
                mds_distorts[i] = 1.0/mds_distorts[i]
        self.mdsp_distorts = mdsp_distorts
        self.mds_distorts = mds_distorts
        print("MDSPlus Highest Distortion: " + str(max(mdsp_distorts)))
        print("MDS Highest Distortion: " + str(max(mds_distorts)))
        print("MDSPlus Geometric Average Distortion: " + str(10**np.average(np.log10(mdsp_distorts))))
        print("Number Distances Negative: " + str(num_negative))
        print("MDS Geometric Average Distortion: " + str(10**np.average(np.log10(mds_distorts))))
        
    #Currently unusable
    def add_mult_distortion_analysis(self, add_dist):
        n = len(self.distance_matrix)
        diameter = np.max(self.distance_matrix)
        mdsp_distorts =[]
        mds_distorts = []
        num_negative = 0
        for i in range(n):
            for j in range(i+1, n):
                if np.iscomplex(self.mdsp_distance_matrix[j][i]):
                    num_negative += 1
                    continue
                mdsp_distorts.append(self.distance_matrix[j][i]/self.mdsp_distance_matrix[j][i])
                mds_distorts.append(self.distance_matrix[j][i]/self.mds_distance_matrix[j][i])
        mdsp_median = np.median(mdsp_distorts)
        mds_median = np.median(mds_distorts)
        for i in range(len(mdsp_distorts)):
            mdsp_distorts[i] /= mdsp_median
            if mdsp_distorts[i] < 1:
                mdsp_distorts[i] = 1.0/mdsp_distorts[i]
        for i in range(len(mds_distorts)):
            mds_distorts[i] /= mds_median
            if mds_distorts[i] < 1:
                mds_distorts[i] = 1.0/mds_distorts[i]
        self.mdsp_distorts = mdsp_distorts
        self.mds_distorts = mds_distorts
        print("MDSPlus Geometric Average Distortion: " + str(10**np.average(np.log10(mdsp_distorts))))
        print("Number Distances Negative: " + str(num_negative))
        print("MDS Geometric Average Distortion: " + str(10**np.average(np.log10(mds_distorts))))
    
    def print_pq(self):
        print("P, Q: " + str(self.p) + ", " + str(self.q))
        print("R, S: " + str(self.r) + ", " + str(self.s))
    
    def analysis_graphs(self):
        fig, ax = plt.subplots(3)
        ax[0].hist(np.log10(self.mdsp_distorts), np.arange(0,3,.25), log=True)
        ax[1].hist(np.log10(self.rhos), np.arange(-3,0,.25), log=True)
        ax[2].hist(np.log10(self.mds_distorts), np.arange(0,3,.25), log=True)
    
    def mdsplus(self, target_dimension):
        n = len(self.distance_matrix)
        pos = []
        neg = []
        for i in range(n):
            if self.eigenvalues[i] < 0:
                neg.append(i)
            elif self.eigenvalues[i] > 0:
                pos.append(i)
        self.p = len(pos)
        self.q = len(neg)
        
        sorted_indices = np.argsort(np.absolute(self.eigenvalues))
        pos = np.intersect1d(pos, sorted_indices[n-target_dimension:n], assume_unique = True)
        neg = np.intersect1d(neg, sorted_indices[n-target_dimension:n], assume_unique = True)
        self.r = len(pos)
        self.s = len(neg)
        self.mdsp_pos_vecs = np.take(self.pq_embedding, pos, axis = 1)
        self.mdsp_neg_vecs= np.take(self.pq_embedding, neg, axis = 1)
        return
    
    def mdsplusmanual(self, td1, td2):
        n = len(self.distance_matrix)
        pos = []
        neg = []
        for i in range(n):
            if self.eigenvalues[i] < 0:
                neg.append(i)
            elif self.eigenvalues[i] > 0:
                pos.append(i)
        self.p = len(pos)
        self.q = len(neg)
        
        sorted_indices = np.argsort(np.absolute(self.eigenvalues))
        pos_indices = []
        neg_indices = []
        index = n
        while len(pos_indices) < td1 or len(neg_indices) < td2:
            index -= 1
            if self.eigenvalues[sorted_indices[index]] > 0 and len(pos_indices) < td1:
                pos_indices.append(sorted_indices[index])
            elif self.eigenvalues[sorted_indices[index]] < 0 and len(neg_indices) < td2:
                neg_indices.append(sorted_indices[index])
        self.r = len(pos_indices)
        self.s = len(neg_indices)
        self.mdsp_pos_vecs = np.take(self.pq_embedding, pos_indices, axis = 1)
        self.mdsp_neg_vecs= np.take(self.pq_embedding, neg_indices, axis = 1)
        return
    
    def mds(self, target_dimension):
        n = len(self.distance_matrix)
        sorted_indices = np.argsort(self.eigenvalues)
        self.mds_vecs = []
        for i in range(n):
            self.mds_vecs.append(np.take(self.pq_embedding[i], sorted_indices[n-target_dimension:n]))
        return
    
    def find_pq_embedding(self):
        n = len(self.distance_matrix)
        distance_squared_matrix = np.zeros((n, n), dtype=np.double)
        for a in range(n):
            for b in range(n):
                distance_squared_matrix[a][b] = self.distance_matrix[a][b] ** 2
        centering = np.identity(n) - 1/n*np.ones((n,n))
        gram = -1/2*np.matmul(centering, distance_squared_matrix)
        gram = np.matmul(gram, centering)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        sqrt_eigenvalues = np.sqrt(np.absolute(eigenvalues))
        for j in range(n):
            for k in range(n):
                eigenvectors[k][j] *= sqrt_eigenvalues[j]
        self.eigenvalues = eigenvalues
        self.pq_embedding = eigenvectors
        return