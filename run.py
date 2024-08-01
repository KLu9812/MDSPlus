# -*- coding: utf-8 -*-
from Sampler import sampler
from MDSPlus import mds
import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPClassifier
# import tensorflow.keras as kt
import numpy as np


def standard_one_run():
    target_dimension = 400
    distances = sampler.chosen_data(1000, "mnist", "missing", 28*28*100)
    
    mds_handler = mds(distances)
    print("Gened")
    mds_handler.find_pq_embedding()
    print("Found PQ Embedding")
    mds_handler.mdsplus(target_dimension)
    # mds_handler.mdsplusmanual(60, 40)
    print("Finished MDSPlus")
    mds_handler.mds(target_dimension)
    print("Finished MDS")
    mds_handler.graph_eigenvalues()
    print("Graphed Eigenvalues")
    mds_handler.calculate_rho()
    print("Calculated Rho")
    mds_handler.gen_mdsp_distance_matrix()
    print("Finished Calculating MDSPlus Distance Matrix")
    mds_handler.gen_mds_distance_matrix()
    print("Finished Calculating MDS Distance Matrix")
    mds_handler.additive_error()
    print("Finished Calculating Additive Error")
    mds_handler.scaled_additive_error()
    print("Finished Calculating Scaled Additive Error")
    mds_handler.mult_distortion_analysis()
    #mds_handler.add_mult_distortion_analysis()
    print("Completed Distortion Analysis")
    mds_handler.print_pq()
    print("Printed PQ")
    mds_handler.analysis_graphs()
    print("Finished Graphing")  


def generate_comparison_graphs():
    # both_distances, ys = sampler.chosen_data(2000, "mnist", "missing", 28*28*500)
    # train_y = np.ravel(ys[:1000])
    # test_y = np.ravel(ys[1000:])
    # mds_handler = mds(both_distances)
    
    distance_matrix = sampler.chosen_data(1000, "mnist", "knn", 5)[0]
    mds_handler = mds(distance_matrix)
    mds_handler.find_pq_embedding()
    mds_handler.graph_eigenvalues()
    
    mdspb_distorts = []
    mdsp_distorts = []
    mds_distorts = []
    num_negative = []
    mdspb_scaled_errors = []
    mdsp_scaled_errors = []
    mds_scaled_errors = []
    mdspb_unscaled_errors = []
    mdsp_unscaled_errors = []
    mds_unscaled_errors = []
    mdsp_c3 = []
    mdspb_c3 = []
    r = []
    s = []
    mdsp_acc = []
    mds_acc = []
    
    dimensions = list(range(4, 1000, 4))
    for target_dimension in dimensions:
        print(target_dimension)
        mds_handler.mdsplus(target_dimension)
        # mds_handler.mdsplusmanual(60, 40)
        mds_handler.mds(target_dimension)
        mds_handler.mdsplusbonus(target_dimension)
        mds_handler.gen_mdspb_distance_matrix(verbose = False)
        mds_handler.gen_mdsp_distance_matrix(verbose = False)
        mds_handler.gen_mds_distance_matrix(verbose = False)
        mds_handler.additive_error()
        mds_handler.scaled_additive_error()
        mds_handler.mult_distortion_analysis()
        mdspb_distorts.append(mds_handler.mdspb_distort)
        mdsp_distorts.append(mds_handler.mdsp_distort)
        mds_distorts.append(mds_handler.mds_distort)
        num_negative.append(mds_handler.num_negative)
        mdspb_scaled_errors.append(mds_handler.mdspb_scaled_additive_error)
        mdsp_scaled_errors.append(mds_handler.mdsp_scaled_additive_error)
        mds_scaled_errors.append(mds_handler.mds_scaled_additive_error)
        mdspb_unscaled_errors.append(mds_handler.mdspb_additive_error)
        mdsp_unscaled_errors.append(mds_handler.mdsp_additive_error)
        mds_unscaled_errors.append(mds_handler.mds_additive_error)
        mdspb_c3.append((mds_handler.mdspb_additive_error - mds_handler.c1c2)/mds_handler.mdspb_additive_error)
        mdsp_c3.append((mds_handler.mdsp_additive_error - mds_handler.c1c2o)/mds_handler.mdsp_additive_error)
        r.append(mds_handler.r)
        s.append(mds_handler.s)
        
        mds_handler.print_pq()
        print("MDSPB C1, C2: " + str(mds_handler.c1c2))
        print("MDSP C1, C2: " + str(mds_handler.c1c2o))
        
        # clf = MLPClassifier(solver='adam', hidden_layer_sizes=(10, 10, 10), random_state=1)
        # clf.fit(mds_handler.mdsp_vecs[:1000], train_y)
        # mdsp_acc.append(clf.score(mds_handler.mdsp_vecs[1000:], test_y))
        # clf = MLPClassifier(solver='adam', hidden_layer_sizes=(10, 10, 10), random_state=1)
        # clf.fit(mds_handler.mds_vecs[:1000], train_y)
        # mds_acc.append(clf.score(mds_handler.mds_vecs[1000:], test_y))
        
        
    
    # fig, ax = plt.subplots(6)
    # fig.set_size_inches(20, 90)
    # ax[0].plot(dimensions, mdsp_distorts, label = "MDSPlus Distortion")
    # ax[0].plot(dimensions, mds_distorts, label = "MDS Distortion")
    # ax[0].legend()
    # ax[1].plot(dimensions, num_negative, label = "Number of Negative Squared Distances")
    # ax[1].legend()
    # ax[2].plot(dimensions, mdsp_scaled_errors, label = "MDSPlus Scaled Additive Error")
    # ax[2].plot(dimensions, mds_scaled_errors, label = "MDS Scaled Additive Error")
    # ax[2].legend()
    # ax[3].plot(dimensions, mdsp_unscaled_errors, label = "MDSPlus Stress")
    # ax[3].plot(dimensions, mds_unscaled_errors, label = "MDS Stress")
    # ax[3].legend()
    # ax[4].plot(dimensions, r, label = "r dimension")
    # ax[4].plot(dimensions, s, label = "s dimension")
    # ax[4].legend()
    # ax[5].plot(dimensions, mdsp_acc, label = "MDSPlus Classify Accuracy")
    # ax[5].plot(dimensions, mds_acc, label = "MDS Classify Accuracy")
    # ax[5].legend()
    
    #For Generate Paper Plots
    fig, ax = plt.subplots(6, sharex=True)
    fig.set_size_inches(4, 24)
    ax[0].plot(dimensions, mdsp_distorts, label = "Neuc-MDS Distortion")
    ax[0].plot(dimensions, mdspb_distorts, label = "Generalized NE-MDS Distortion")
    ax[0].plot(dimensions, mds_distorts, label = "MDS Distortion")
    ax[0].legend()
    ax[1].plot(dimensions, mdsp_scaled_errors, label = "Neuc-MDS Scaled Additive Error")
    ax[1].plot(dimensions, mdspb_scaled_errors, label = "Generalized NE-MDS Scaled Additive Error")
    ax[1].plot(dimensions, mds_scaled_errors, label = "MDS Scaled Additive Error")
    ax[1].legend()
    ax[2].plot(dimensions, mdsp_unscaled_errors, label = "Neuc-MDS Stress")
    ax[2].plot(dimensions, mdspb_unscaled_errors, label = "Generalized NE-MDS Stress")
    ax[2].plot(dimensions, mds_unscaled_errors, label = "MDS Stress")
    ax[2].legend()
    ax[3].plot(dimensions, r, label = "r dimension")
    ax[3].plot(dimensions, s, label = "s dimension")
    ax[3].legend()
    ax[4].plot(dimensions, mdspb_c3, label = "Generalized Neuc-MDS C3/Stress")
    ax[4].legend()
    ax[5].plot(dimensions, mdsp_c3, label = "Neuc-MDS C3/Stress")
    ax[5].legend()

def landmark_mds(distance_matrix, k):
    n = len(distance_matrix)
    return_matrix = np.zeros((n, n))
    

def power_distance_testing():
    # distance_squared = sampler.gen_similarity_matrix(10, lower_bound = -1, upper_bound = 1)
    # mds_handler = mds(distance_squared)
    # mds_handler.distance_squared_matrix = distance_squared
    # mds_handler.find_pq_embedding()
    # mds_handler.graph_eigenvalues()
    k = 5
    
    
    def subtract_i_r(distance_matrix, i, r):
        return_distance_matrix = distance_matrix.copy()
        for j in range(len(distance_matrix)):
            if j == i:
                continue
            return_distance_matrix[i][j] += r
            return_distance_matrix[j][i] += r
        return return_distance_matrix
    
    count = 0
    for i in range(100000):
        distance_squared = np.zeros((k, k), dtype=int)
        ris = []
        for j in range(k):
            next_ri = np.random.randint(1, 1000, dtype=int)
            distance_squared = subtract_i_r(distance_squared, j, next_ri)
            ris.append(next_ri)
        distance_squared = np.square(distance_squared)
        if np.linalg.det(distance_squared) < 0:
            print(distance_squared)
            print(np.linalg.det(distance_squared))
            print(ris)
            count += 1
            
    
    # test_distance_squared = subtract_i_r(distance_squared, 3, .5)
    # test_distance_squared = subtract_i_r(test_distance_squared, 1, .1)
    # test_distance_squared = subtract_i_r(test_distance_squared, 2, .6)
    # test_distance_squared = subtract_i_r(test_distance_squared, 4, .7)
    # test_distance_squared = subtract_i_r(test_distance_squared, 0, .2)
    # test_distance_squared = subtract_i_r(test_distance_squared, 5, .7)
    # print(distance_squared)
    # print(test_distance_squared)
    # mds_handler1 = mds(test_distance_squared)
    # mds_handler1.distance_squared_matrix = test_distance_squared
    # mds_handler1.find_pq_embedding()
    # mds_handler1.graph_eigenvalues()
    # print(mds_handler1.eigenvalues)
    print(count)
    
    # print(mds_handler.eigenvalues - mds_handler1.eigenvalues)
# generate_comparison_graphs()
#standard_one_run()

# distance_squared = sampler.sphere_metric_random(1000, .7)
# mds_handler = mds(distance_squared)
# mds_handler.distance_squared_matrix = distance_squared
# mds_handler.find_pq_embedding()
# mds_handler.graph_eigenvalues()

power_distance_testing()