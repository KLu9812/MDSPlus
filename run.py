# -*- coding: utf-8 -*-
from Sampler import sampler
from MDSPlus import mds



target_dimension = 1000
distances = sampler.gen_similarity_matrix(10000)

mds_handler = mds(distances)
print("Gened")
mds_handler.find_pq_embedding()
print("Found PQ Embedding")
#mds_handler.mdsplus(target_dimension)
mds_handler.mdsplusmanual(600, 400)
print("Finished MDSPlus")
mds_handler.mds(target_dimension)
print("Finished MDS")
mds_handler.graph_eigenvalues()
print("Graphed Eigenvalues")
mds_handler.calculate_rho()
print("Calculated Rho")
mds_handler.mdsp_distance_matrix()
print("Finished Calculating MDSPlus Distance Matrix")
mds_handler.mds_distance_matrix()
print("Finished Calculating MDS Distance Matrix")
mds_handler.mult_distortion_analysis()
#mds_handler.add_mult_distortion_analysis()
print("Completed Distortion Analysis")
mds_handler.print_pq()
print("Printed PQ")
mds_handler.analysis_graphs()
print("Finished Graphing")