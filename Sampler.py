# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import igraph as ig
import tensorflow.keras as kt
# from sklearn.neural_network import MLPClassifier

np.random.seed(20)

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
    
    def ball_data(n):
        balls = []
        for i in range(n):
            point = np.random.uniform(0, 100, 10)
            balls.append(point)
        
        # distance_matrix = np.zeros((n, n))
        # for i in range(n):
        #     for j in range(i+1, n):
        #         distance = np.linalg.norm(balls[i] - balls[j])
        #         distance_matrix[i][j] = distance
        #         distance_matrix[j][i] = distance
        
        # radii = []
        # for i in range(n):
        #     radius = np.random.uniform(0,5)
        #     if np.random.uniform(0,1) < 0.1:
        #         radius = np.min(np.append(distance_matrix[i][:i], distance_matrix[i][i+1:])) * .4
        #     radii.append(radius)
        # for i in range(n):
        #     for j in range(n):
        #         if i == j:
        #             continue
        #         distance_matrix[i][j] -= radii[i]
        #         distance_matrix[j][i] -= radii[i]
        
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(balls[i] - balls[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        for i in range(n):
            radius = 0
            if np.random.uniform(0,1) < 0.1:
                radius = np.min(np.append(distance_matrix[i][:i], distance_matrix[i][i+1:])) * .8
            else:
                radius = np.random.uniform(0, 5)
            for j in range(n):
                if i == j:
                    continue
                distance_matrix[i][j] -= radius
                distance_matrix[j][i] -= radius
                
        
        return distance_matrix
    
    def sphere_metric_random(n, max_r):
        distance_matrix = np.zeros((n, n))
        balls = []
        for i in range(n):
            point = (np.random.uniform(0, .5, 4), np.random.uniform(0, max_r))
            balls.append(point)
        for i in range(n):
            for j in range(i, n):
                distance = np.linalg.norm(balls[i][0] - balls[j][0])**2
                distance -= np.linalg.norm(balls[i][1] + balls[j][1])**2
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        return distance_matrix
    
    #Embedding chosen data sets. Available Methods and Data Sets:
    #mnist, fashion_mnist, cifar10
    #noise, knn, missing
    def chosen_data(n, data, method, *method_args):
        working_data = []
        if data == "mnist":
            mnist = kt.datasets.mnist
            (train_x, train_y), (test_x, test_y) = mnist.load_data()
            working_data = train_x[:n]
            working_data = working_data.reshape(n, 28*28)
        
        if data == "fashion_mnist":
            fashion_mnist = kt.datasets.fashion_mnist
            (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
            working_data = train_x[:n]
            working_data = working_data.reshape(n, 28*28)
        
        if data =="cifar10":
            cifar10 = kt.datasets.cifar10
            (train_x, train_y), (test_x, test_y) = cifar10.load_data()
            working_data = train_x[:n]
            working_data = working_data.reshape(n, 32*32*3)
        
        distance_matrix = np.zeros((n, n))
        if method == "noise":
            scale = np.max(working_data) - np.min(working_data)
            scale *= np.sqrt(len(working_data[0])) / 500
            for i in range(n):
                for j in range(i + 1, n):
                    distance = np.linalg.norm(working_data[i] - working_data[j])
                    distance += np.random.normal(loc = 0, scale = scale)
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
        
        if method == "knn":
            k = method_args[0]
            edges = []
            for i in range(n):
                distances = np.zeros(n)
                for j in range(n):
                    distances[j] = np.linalg.norm(working_data[i] - working_data[j])
                k_closest = np.argsort(distances)[1:k+1]
                for j in k_closest:
                    edges.append([i, j])
                    edges.append([j, i])
            graph = ig.Graph(n=n, edges=edges)
            distance_matrix = np.array(graph.distances())
                
        if method == "missing":
            num_missing = method_args[0]
            missing = np.random.choice(np.arange(n*len(working_data[0]), dtype=np.int64), num_missing, replace = False)
            missing_matrix = np.ones((n, len(working_data[0])))
            for m in missing:
                missing_matrix[m//len(working_data[0])][m%len(working_data[0])] = 0
            for i in range(n):
                for j in range(i+1, n):
                    v1 = np.multiply(working_data[i], missing_matrix[i])
                    v1 = np.multiply(v1, missing_matrix[j])
                    v2 = np.multiply(working_data[j], missing_matrix[i])
                    v2 = np.multiply(v2, missing_matrix[j])
                    distance = np.linalg.norm(v1-v2)
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
        
        return distance_matrix, train_y[:n]
        
        
    