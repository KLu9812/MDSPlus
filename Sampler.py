# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import igraph as ig
import keras

class InvalidDistanceException(Exception):
    "Raised when the distance between two points is invalid"
    def __init__(self, dist):
        print(dist)

class sampler:
    #Generates random distance matrix that satisfies triangle inequality
    def gen_random_matrix(n, lower_bound, upper_bound):
    	return_matrix = np.zeros((n,n))
    	for i in range(n):
    		for j in range(i + 1, n):

    			current_lower_bound = lower_bound
    			current_upper_bound = upper_bound

    			for i_2 in range(i):
    				new_upper_bound = return_matrix[i_2][i] + return_matrix[i_2][j]
    				if new_upper_bound < current_upper_bound:
    					current_upper_bound = new_upper_bound
    				new_lower_bound = abs(return_matrix[i_2][i] - return_matrix[i_2][j])
    				if new_lower_bound > current_lower_bound:
    					current_lower_bound = new_lower_bound

    			return_matrix[i][j] = random.uniform(current_lower_bound, current_upper_bound)
    			return_matrix[j][i] = return_matrix[i][j]
    	return return_matrix

    #Generates random similarity matrix
    def gen_similarity_matrix(n, lower_bound = .0000001, upper_bound = 1):
        return_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(i + 1, n):
                return_matrix[i][j] = random.uniform(lower_bound, upper_bound)
                return_matrix[j][i] = return_matrix[i][j]
        return return_matrix

    #Generates random similarity matrix with at least p positive eigenvalues
    def gen_random_matrix_p(n, p, upper_bound = 1, lower_bound = .0000001, verbose = True):
        return_matrix = np.zeros((n,n))
        
        #Generating Euclidean points in R^p space to find distances of
        random_euclideans = []
        uniform_upper = math.sqrt(upper_bound/p)
        for i in range(p):
            new_row = np.random.uniform(lower_bound, uniform_upper, p)
            random_euclideans.append(new_row)
        max_val = 0
        for i in range(p):
            if verbose:
                if i % (p // 100) == 0:
                    print(i // (p //100))
            for j in range(i + 1, p):
                dist = np.linalg.norm(random_euclideans[i] - random_euclideans[j])
                if dist > upper_bound or dist < lower_bound:
                    raise InvalidDistanceException(dist)
                return_matrix[i][j] = dist
                return_matrix[j][i] = dist
                if dist > max_val:
                    max_val = dist
        for i in range(p):
            for j in range(p):
                return_matrix[i][j] *= upper_bound/max_val
        for i in range(p, n):
            for j in range(0, n):
                if i == j:
                    break
                return_matrix[i][j] = random.uniform(lower_bound, upper_bound)
                return_matrix[j][i] = return_matrix[i][j]
        return return_matrix
    
    #Generates n points in euclidean space at dimension r < n
    def gen_random_low_rank_euclidean(n, r):
        space = []
        for i in range(r):
            if i == 0:
                new = np.random.randn(n)
                new /= np.linalg.norm(new)
                space.append(new)
                continue
            
            system = np.array(space)
            random_vals = np.random.randn(n - i)
            b = np.zeros(i)
            for j in range(i):
                for k in range(n - i):
                    b[j] -= system[j][k] * random_vals[k]
            new_system = system[:, -i:]
            solution = np.linalg.solve(new_system, b)
            new_ortho = np.append(random_vals, solution)
            new_ortho /= np.linalg.norm(new_ortho)
            space.append(new_ortho)
        
        n_points = []
        for i in range(n):
            random_vals = np.random.randn(r)
            new_point = np.zeros(n)
            for j in range(r):
                new_point = np.add(new_point, random_vals[j] * space[j])
            n_points.append(new_point)
        
        return_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                return_matrix[i][j] = np.linalg.norm(n_points[i] - n_points[j])
        
        return return_matrix
    
    #Generate shortest distance distance matrix for a Margulis-Gabber-Galil
    #graph on n^2 vertices
    def gen_mgg_graph(n):
        edges = []
        for i in range(n**2):
            x = i // (n)
            y = i % n
            edges.append([i, ((x + 2*y)%n)*n + y])
            edges.append([i, ((x + 2*y + 1)%n)*n+y])
            edges.append([i, x*n + ((y + 2*x)%n)])
            edges.append([i, x*n + ((y + 2*x + 1)%n)])
        graph = ig.Graph(n=n**2,edges=edges)
        return np.array(graph.distances())
    
    #Generating a random d-regular graph on n vertices
    def gen_random_reg_graph(n, d):
        graph = ig.Graph.K_Regular(n, d)
        return np.array(graph.distances())
    
    def gen_heavy_neg(n, q):
        points = []
        uniform_upper = math.sqrt(1/(n-q))
        for i in range(n):
            new_row = np.random.uniform(0, uniform_upper, n-q)
            points.append(new_row)
        return_matrix = np.zeros((n, n))
        q_uniform_upper = math.sqrt(.05/(q-1))
        neg_points = []
        for i in range(n):
            new_row = np.random.uniform(0, q_uniform_upper, q-1)
            neg_points.append(new_row)
        num_neg = 0
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(points[i] - points[j])**2
                distance -= np.linalg.norm(neg_points[i] - neg_points[j])**2
                distance -= ((j - i)*.3/n)**2
                distance = math.sqrt(distance)
                if distance < 0:
                    num_neg += 1
                return_matrix[i][j] = distance
                return_matrix[j][i] = distance
        print("Number of Negative Generated Distances: " + str(num_neg))
        return return_matrix

    #Currently Unusable    
    def mnist(n):
        mnist = keras.datasets.mnist
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
    
    