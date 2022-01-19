import numpy as np
from sklearn.utils.graph import graph_shortest_path
from sklearn.neighbors import NearestNeighbors

from src.model.DR.MethodFromScratch.mds import mds
def isomap(data, n_components=2, n_neighbors=6):
    """
    Dimensionality reduction with isomap algorithm
    :param data: input image matrix of shape (n,m) if dist=False, square distance matrix of size (n,n) if dist=True
    :param n_components: number of components for projection
    :param n_neighbors: number of neighbors for distance matrix computation
    :return: Projected output of shape (n_components, n)
    """
    # Compute distance matrix
    print("distance_mat")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    data, _ = nbrs.kneighbors(data)
    print("shortest_path")
    print(data.shape)
    # Compute shortest paths from distance matrix
    graph = graph_shortest_path(data, directed=True)
    graph = -0.5 * (graph ** 2)
    print("mds")
    # Return the MDS projection on the shortest paths graph
    return mds(graph, n_components)
