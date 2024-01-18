# -*- coding: utf-8 -*-
from Sampler import sampler
from MDSPlus import mds
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import tensorflow.keras as kt
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
    both_distances, ys = sampler.chosen_data(2000, "mnist", "missing", 28*28*500)
    train_y = np.ravel(ys[:1000])
    test_y = np.ravel(ys[1000:])
    mds_handler = mds(both_distances)
    mds_handler.find_pq_embedding()
    mds_handler.graph_eigenvalues()
    
    mdsp_distorts = []
    mds_distorts = []
    num_negative = []
    mdsp_scaled_errors = []
    mds_scaled_errors = []
    mdsp_unscaled_errors = []
    mds_unscaled_errors = []
    r = []
    s = []
    mdsp_acc = []
    mds_acc = []
    
    dimensions = list(range(1, 1001, 4))
    for target_dimension in dimensions:
        print(target_dimension)
        mds_handler.mdsplus(target_dimension)
        # mds_handler.mdsplusmanual(60, 40)
        mds_handler.mds(target_dimension)
        mds_handler.gen_mdsp_distance_matrix(verbose = False)
        mds_handler.gen_mds_distance_matrix(verbose = False)
        mds_handler.additive_error()
        mds_handler.scaled_additive_error()
        mds_handler.mult_distortion_analysis()
        mdsp_distorts.append(mds_handler.mdsp_distort)
        mds_distorts.append(mds_handler.mds_distort)
        num_negative.append(mds_handler.num_negative)
        mdsp_scaled_errors.append(mds_handler.mdsp_scaled_additive_error)
        mds_scaled_errors.append(mds_handler.mds_scaled_additive_error)
        mdsp_unscaled_errors.append(mds_handler.mdsp_additive_error)
        mds_unscaled_errors.append(mds_handler.mds_additive_error)
        r.append(mds_handler.r)
        s.append(mds_handler.s)
        
        # clf = MLPClassifier(solver='adam', hidden_layer_sizes=(10, 10, 10), random_state=1)
        # clf.fit(mds_handler.mdsp_vecs[:1000], train_y)
        # mdsp_acc.append(clf.score(mds_handler.mdsp_vecs[1000:], test_y))
        # clf = MLPClassifier(solver='adam', hidden_layer_sizes=(10, 10, 10), random_state=1)
        # clf.fit(mds_handler.mds_vecs[:1000], train_y)
        # mds_acc.append(clf.score(mds_handler.mds_vecs[1000:], test_y))
        
        
    
    fig, ax = plt.subplots(6)
    fig.set_size_inches(20, 90)
    ax[0].plot(dimensions, mdsp_distorts, label = "MDSPlus Distortion")
    ax[0].plot(dimensions, mds_distorts, label = "MDS Distortion")
    ax[0].legend()
    ax[1].plot(dimensions, num_negative, label = "Number of Negative Squared Distances")
    ax[1].legend()
    ax[2].plot(dimensions, mdsp_scaled_errors, label = "MDSPlus Scaled Additive Error")
    ax[2].plot(dimensions, mds_scaled_errors, label = "MDS Scaled Additive Error")
    ax[2].legend()
    ax[3].plot(dimensions, mdsp_unscaled_errors, label = "MDSPlus Stress")
    ax[3].plot(dimensions, mds_unscaled_errors, label = "MDS Stress")
    ax[3].legend()
    ax[4].plot(dimensions, r, label = "r dimension")
    ax[4].plot(dimensions, s, label = "s dimension")
    ax[4].legend()
    # ax[5].plot(dimensions, mdsp_acc, label = "MDSPlus Classify Accuracy")
    # ax[5].plot(dimensions, mds_acc, label = "MDS Classify Accuracy")
    # ax[5].legend()




generate_comparison_graphs()
#standard_one_run()