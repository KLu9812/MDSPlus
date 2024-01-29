# -*- coding: utf-8 -*-
import numpy as np
# import random
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
    mdsp_vecs = None
    mdsp_pos_vecs = None
    mdsp_neg_vecs = None
    mdsp_coords = None
    mds_coords = None
    mds_vecs = None
    p = 0
    q = 0
    mdsp_distance_matrix = None
    mds_distance_matrix = None
    mds_distort = 0
    mdsp_distort = 0
    mds_distorts = []
    mdsp_distorts = []
    num_negative = 0
    mdsp_additive_error = 0
    mds_additive_error = 0
    mdsp_scaled_additive_error = 0
    mds_scaled_additive_error = 0
    
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
        fig, ax = plt.subplots(1)
        step_size = (np.ceil(np.max(self.eigenvalues)) + 1 - np.floor(np.min(self.eigenvalues)))/10
        ax.hist(self.eigenvalues, np.arange(np.floor(np.min(self.eigenvalues)),
                                               np.ceil(np.max(self.eigenvalues)) + 1, step_size), log=True)
        plt.title("Eigenvalues")
        plt.show()
    
    def gen_mdsp_distance_matrix(self, verbose = True):
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
    
    def gen_mdspb_distance_matrix(self, verbose = True):
        n = len(self.distance_matrix)
        self.mdspb_distance_matrix = np.zeros((n, n))
        for i in range(n):
            if verbose:
                if i % (n // 100) == 0:
                    print(i // (n // 100))
            for j in range(i+1, n):
                mdspb_distance = np.linalg.norm(self.mdspb_pos_vecs[j] - self.mdspb_pos_vecs[i])**2
                mdspb_distance -= np.linalg.norm(self.mdspb_neg_vecs[j] - self.mdspb_neg_vecs[i])**2
                if mdspb_distance < 0:
                    mdspb_distance = -np.sqrt(np.absolute(mdspb_distance))
                else:
                    mdspb_distance = np.sqrt(mdspb_distance)
                self.mdspb_distance_matrix[i][j] = mdspb_distance
                self.mdspb_distance_matrix[j][i] = mdspb_distance
    
    def gen_mds_distance_matrix(self, verbose = True):
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
        mdspb_distorts = []
        mdsp_distorts =[]
        mds_distorts = []
        num_negative = 0
        for i in range(n):
            for j in range(i+1, n):
                if self.mdsp_distance_matrix[j][i] < 0:
                    num_negative += 1
                    continue
                if self.mdspb_distance_matrix[j][i] < 0:
                    continue
                mdspb_distorts.append(self.distance_matrix[j][i]/self.mdspb_distance_matrix[j][i])
                mdsp_distorts.append(self.distance_matrix[j][i]/self.mdsp_distance_matrix[j][i])
                mds_distorts.append(self.distance_matrix[j][i]/self.mds_distance_matrix[j][i])
        mdspb_median = np.median(mdspb_distorts)
        mdsp_median = np.median(mdsp_distorts)
        mds_median = np.median(mds_distorts)
        for i in range(len(mdspb_distorts)):
            mdspb_distorts[i] /= mdspb_median
            if mdspb_distorts[i] < 1:
                mdspb_distorts[i] = 1.0/mdspb_distorts[i]
        for i in range(len(mdsp_distorts)):
            mdsp_distorts[i] /= mdsp_median
            if mdsp_distorts[i] < 1:
                mdsp_distorts[i] = 1.0/mdsp_distorts[i]
        for i in range(len(mds_distorts)):
            mds_distorts[i] /= mds_median
            if mds_distorts[i] < 1:
                mds_distorts[i] = 1.0/mds_distorts[i]
        self.mdspb_distort = 10**np.average(np.log10(mdspb_distorts))
        self.mdsp_distort = 10**np.average(np.log10(mdsp_distorts))
        self.mds_distort = 10**np.average(np.log10(mds_distorts))
        self.mdspb_distorts = mdspb_distorts
        self.mdsp_distorts = mdsp_distorts
        self.mds_distorts = mds_distorts
        self.num_negative = num_negative
        print("MDSPlus Highest Distortion: " + str(max(mdsp_distorts)))
        print("MDS Highest Distortion: " + str(max(mds_distorts)))
        print("MDSPlus Geometric Average Distortion: " + str(10**np.average(np.log10(mdsp_distorts))))
        print("Number Distances Negative: " + str(num_negative))
        print("MDS Geometric Average Distortion: " + str(10**np.average(np.log10(mds_distorts))))
        
        
    def additive_error(self):
        mds_error = np.linalg.norm(self.mds_distance_matrix - self.distance_matrix)
        mdsp_error = np.linalg.norm(self.mdsp_distance_matrix-self.distance_matrix)
        mdspb_error = np.linalg.norm(self.mdspb_distance_matrix-self.distance_matrix)
        print("MDSPlus Additive Error: " + str(mdsp_error))
        print("MDSPlusBonus Additive Error: " + str(mdspb_error))
        print("MDS Additive Error: " + str(mds_error))
        self.mdsp_additive_error = mdsp_error
        self.mdspb_additive_error = mdspb_error
        self.mds_additive_error = mds_error
    
    def scaled_additive_error(self):
        flattened_distance = self.distance_matrix.flatten()
        flattened_mds = self.mds_distance_matrix.flatten()
        flattened_mdsp = self.mdsp_distance_matrix.flatten()
        flattened_mdspb = self.mdspb_distance_matrix.flatten()
        mds_error = np.linalg.norm(flattened_distance - np.dot(flattened_mds, flattened_distance)
                                   /np.linalg.norm(flattened_mds)**2*flattened_mds)
        mdsp_error = np.linalg.norm(flattened_distance - np.dot(flattened_mdsp, flattened_distance)
                                   /np.linalg.norm(flattened_mdsp)**2*flattened_mdsp)
        mdspb_error = np.linalg.norm(flattened_distance - np.dot(flattened_mdspb, flattened_distance)
                                   /np.linalg.norm(flattened_mdspb)**2*flattened_mdspb)
        print("MDSPlus Scaled Additive Error: " + str(mdsp_error))
        print("MDSPlusBonus Scaled Additive Error: " + str(mdspb_error))
        print("MDS Scaled Additive Error: " + str(mds_error))
        self.mdsp_scaled_additive_error = mdsp_error
        self.mdspb_scaled_additive_error = mdspb_error
        self.mds_scaled_additive_error = mds_error
    
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
        print("BR, BS: " + str(self.br) + ", " + str(self.bs))
    
    def analysis_graphs(self):
        fig, ax = plt.subplots(3)
        ax[0].hist(np.log10(self.mdsp_distorts), np.arange(0,8,.25), log=True)
        ax[1].hist(np.log10(self.rhos), np.arange(-3,0,.25), log=True)
        ax[2].hist(np.log10(self.mds_distorts), np.arange(0,8,.25), log=True)
    
    # def mdsplus(self, target_dimension):
    #     n = len(self.distance_matrix)
    #     pos = []
    #     neg = []
    #     for i in range(n):
    #         if self.eigenvalues[i] < 0:
    #             neg.append(i)
    #         elif self.eigenvalues[i] > 0:
    #             pos.append(i)
    #     self.p = len(pos)
    #     self.q = len(neg)
        
    #     sorted_indices = np.argsort(np.absolute(self.eigenvalues))
    #     pos = np.intersect1d(pos, sorted_indices[n-target_dimension:n], assume_unique = True)
    #     neg = np.intersect1d(neg, sorted_indices[n-target_dimension:n], assume_unique = True)
    #     self.r = len(pos)
    #     self.s = len(neg)
    #     self.mdsp_pos_vecs = np.take(self.pq_embedding, pos, axis = 1)
    #     self.mdsp_neg_vecs= np.take(self.pq_embedding, neg, axis = 1)
    #     return
    
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
        
        sorted_indices = np.argsort(self.eigenvalues)
        pos_selected = []
        neg_selected = []
        #c_1 = np.sum(np.square(self.eigenvalues))
        c_2l = np.sum(self.eigenvalues)
        neg_index = 0
        pos_index = len(self.eigenvalues) - 1
        for i in range(target_dimension):
            if c_2l < 0:
                neg_selected.append(sorted_indices[neg_index])
                #c_1 -= self.eigenvalues[sorted_indices[neg_index]]**2
                c_2l -= self.eigenvalues[sorted_indices[neg_index]]
                neg_index += 1
            elif c_2l >= 0:
                pos_selected.append(sorted_indices[pos_index])
                #c_1 -= self.eigenvalues[sorted_indices[pos_index]]**2
                c_2l -= self.eigenvalues[sorted_indices[pos_index]]
                pos_index -= 1
        self.r = len(pos_selected)
        self.s = len(neg_selected)
        all_selected = pos_selected + neg_selected
        self.mdsp_coords = all_selected
        self.mdsp_pos_vecs = np.take(self.pq_embedding, pos_selected, axis = 1)
        self.mdsp_neg_vecs= np.take(self.pq_embedding, neg_selected, axis = 1)
        self.mdsp_vecs = np.take(self.pq_embedding, all_selected, axis = 1)
        return
    
    def mdsplusbonus(self, target_dimension):
        n = len(self.distance_matrix)
        pos = []
        neg = []

        for i in range(n):
            if self.eigenvalues[i] < 0:
                neg.append(i)
            elif self.eigenvalues[i] > 0:
                pos.append(i)
        
        sorted_indices = np.argsort(self.eigenvalues)
        pos_selected = []
        neg_selected = []
        c_1 = np.sum(np.square(self.eigenvalues))
        c_2l = np.sum(self.eigenvalues)
        neg_index = 0
        pos_index = len(self.eigenvalues) - 1
        for i in range(target_dimension):
            pos_c_1 = c_1 - self.eigenvalues[sorted_indices[pos_index]]**2
            pos_c_2l = c_2l - self.eigenvalues[sorted_indices[pos_index]]
            neg_c_1 = c_1 - self.eigenvalues[sorted_indices[neg_index]]**2
            neg_c_2l = c_2l - self.eigenvalues[sorted_indices[neg_index]]
            if pos_c_1 + pos_c_2l**2/(target_dimension+1) > neg_c_1+neg_c_2l**2/(target_dimension+1):
                neg_selected.append(sorted_indices[neg_index])
                c_1 = neg_c_1
                c_2l = neg_c_2l
                neg_index += 1
            else:
                pos_selected.append(sorted_indices[pos_index])
                c_1 = pos_c_1
                c_2l = pos_c_2l
                pos_index -= 1
        self.br = len(pos_selected)
        self.bs = len(neg_selected)
        all_selected = pos_selected + neg_selected
        self.mdspb_coords = all_selected
        new_eigenvalues = self.eigenvalues - c_2l / (target_dimension + 1)
        new_pq_embedding = self.eigenvectors.copy()
        for j in range(n):
            for k in range(n):
                new_pq_embedding[k][j] *= math.sqrt(np.absolute(new_eigenvalues[j]))
        self.mdspb_pos_vecs = np.take(new_pq_embedding, pos_selected, axis = 1)
        self.mdspb_neg_vecs= np.take(new_pq_embedding, neg_selected, axis = 1)
        self.mdspb_vecs = np.take(new_pq_embedding, all_selected, axis = 1)
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
        all_indices = pos_indices + neg_indices
        self.mdsp_coords = all_indices
        self.r = len(pos_indices)
        self.s = len(neg_indices)
        self.mdsp_pos_vecs = np.take(self.pq_embedding, pos_indices, axis = 1)
        self.mdsp_neg_vecs= np.take(self.pq_embedding, neg_indices, axis = 1)
        self.mdsp_vecs = np.take(self.pq_embedding, all_indices, axis = 1)
        return
    
    def mds(self, target_dimension):
        n = len(self.distance_matrix)
        sorted_indices = np.argsort(self.eigenvalues)
        self.mds_vecs = []
        self.mds_coords = sorted_indices[n-target_dimension:n]
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
        self.eigenvectors = eigenvectors.copy()
        sqrt_eigenvalues = np.sqrt(np.absolute(eigenvalues))
        for j in range(n):
            for k in range(n):
                eigenvectors[k][j] *= sqrt_eigenvalues[j]
        self.eigenvalues = eigenvalues
        self.pq_embedding = eigenvectors
        return