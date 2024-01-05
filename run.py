# -*- coding: utf-8 -*-
from Sampler import sampler
from MDSPlus import mds
import matplotlib.pyplot as plt


def standard_one_run():
    target_dimension = 75
    distances = sampler.gen_heavy_neg(10000, 9900)
    
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
    distances = sampler.gen_heavy_neg(1000, 900)
    mds_handler = mds(distances)
    mds_handler.find_pq_embedding()
    
    mdsp_distorts = []
    mds_distorts = []
    num_negative = []
    mdsp_scaled_errors = []
    mds_scaled_errors = []
    mdsp_unscaled_errors = []
    mds_unscaled_errors = []
    r = []
    s = []
    
    dimensions = list(range(1, 101, 2))
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
    
    fig, ax = plt.subplots(5)
    fig.set_size_inches(20, 75)
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






generate_comparison_graphs()
#standard_one_run()