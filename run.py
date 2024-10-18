# -*- coding: utf-8 -*-
from Sampler import sampler
from MDSPlus import mds
import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPClassifier
# import tensorflow.keras as kt
import numpy as np


def standard_one_run(distances, target_dimension):
    mds_handler = mds(distances)
    mds_handler.find_pq_embedding()
    mds_handler.mdsplus(target_dimension)
    mds_handler.mds(target_dimension)
    mds_handler.mdsplusbonus(target_dimension)
    mds_handler.graph_eigenvalues()
    mds_handler.calculate_rho(verbose = False)
    mds_handler.gen_mdsp_distance_matrix(verbose = False)
    mds_handler.gen_mds_distance_matrix(verbose = False)
    mds_handler.gen_mdspb_distance_matrix(verbose = False)
    mds_handler.additive_error()
    mds_handler.scaled_additive_error()
    mds_handler.mult_distortion_analysis()
    mds_handler.print_pq()
    mds_handler.analysis_graphs()


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

def landmark_mds(distance_matrix, k, t):
    n = len(distance_matrix)
    distance_squared_matrix = np.square(distance_matrix)
    landmark_matrix = np.zeros((k, k))
    indices_picked = np.random.choice(n, size = k, replace = False)
    for i in range(len(indices_picked)):
        for j in range(len(indices_picked)):
            landmark_matrix[i][j] = distance_matrix[indices_picked[i], indices_picked[j]]
    
    mds_handler = mds(landmark_matrix)
    mds_handler.find_pq_embedding()
    mds_handler.mdsplus(t)
    mds_handler.gen_mdsp_distance_matrix(verbose = False)
    mean_columns = np.mean(distance_squared_matrix[indices_picked][:, indices_picked], axis = 0)
    L_kP = mds_handler.mdsp_vecs.copy()
    for i in range(len(L_kP)):
        for j in range(len(L_kP[0])):
            L_kP[i][j] /= mds_handler.eigenvalues[mds_handler.mdsp_coords[j]]
    L_kP = np.transpose(L_kP)
    
    remaining_deltas = np.take(distance_squared_matrix, indices_picked, axis = 0)
    mean_columns_spread = np.repeat(np.transpose([mean_columns]), n, axis = 1)
    x_a = -1 * np.matmul(L_kP , (remaining_deltas - mean_columns_spread)) / 2
    
    final_distance_matrix = np.zeros((n, n))
    pos_coords = len(mds_handler.mdsp_pos_vecs[0])
    for i in range(n):
        for j in range(n):
            pos_distance = np.linalg.norm(x_a[:pos_coords, i] - x_a[:pos_coords, j])**2
            neg_distance = np.linalg.norm(x_a[pos_coords:, i] - x_a[pos_coords:, j])**2
            final_distance_matrix[i][j] = pos_distance - neg_distance
    
    stress = np.linalg.norm(distance_squared_matrix - final_distance_matrix)**2
    compare_mds = mds(distance_matrix)
    compare_mds.find_pq_embedding()
    compare_mds.mdsplus(t)
    compare_mds.gen_mdsp_distance_matrix(verbose = False)
    compare_mds.additive_error()
    print("Landmark MDSP Stress: " + str(stress))
        

    
distance_matrix = sampler.gen_heavy_neg(1000, 800)
standard_one_run(distance_matrix, 500)