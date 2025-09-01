import numpy as np
from scipy.stats import skew
import xgi
import networkx as nx
import itertools 
from collections import Counter

from sknetwork.data import from_edge_list
from sknetwork.clustering import Louvain, get_modularity



# Degree stats
def avg_degree(H):
    degrees = list(H.degree().values())
    return np.mean(degrees)

def std_degree(H):
    degrees = list(H.degree().values())
    return np.std(degrees)

def skewness(H):
    degrees = list(H.degree().values())
    return skew(degrees)
def median(H):
    degrees = list(H.degree().values())
    return np.median(degrees)
    
# Edge size stats
def avg_size(H):
    sizes = list(H.edges.size.values())
    return np.mean(sizes)

def std_size(H):
    size = list(H.edges.size.values())
    return np.std(size)

# Community struture
def xgi_2_sknetwork(X: xgi.Hypergraph):
    edge_list = xgi.to_bipartite_edgelist(X)
    network = from_edge_list(edge_list, bipartite=True, sum_duplicates =True)
    return network

def clustering(X: xgi.Hypergraph, res =1, random_state = None):
    network = xgi_2_sknetwork(X)
    biadjacency_matrix = network.biadjacency
    louvain = Louvain(resolution = res, modularity ='Newman' , shuffle_nodes = True, random_state = random_state)
    louvain.fit(biadjacency_matrix, force_bipartite = True)
    node_clusters = louvain.labels_row_
    edge_clusters = louvain.labels_col_
    node_labels = network.names
    return biadjacency_matrix, node_labels, node_clusters, edge_clusters

def clustering_metrics(H: xgi.Hypergraph, **kwargs):
    biadjacency_matrix, node_labels, node_clusters, edge_clusters = clustering(H, **kwargs)
    mod = get_modularity(biadjacency_matrix, node_clusters, edge_clusters)
    nb_clusters = max(node_clusters)
    return {'modularity': mod, 'nb_clusters': nb_clusters}


def modularity(H):   
    biadjacency_matrix, node_labels, node_clusters, edge_clusters = clustering(H)
    network = xgi_2_sknetwork(H)
    biadjacency_matrix = network.biadjacency
    return get_modularity(biadjacency_matrix, node_clusters, edge_clusters)

def nb_clusters(H, **kwargs):
    biadjacency_matrix, node_labels, node_clusters, edge_clusters = clustering(H, **kwargs)
    return max(node_clusters)

def local_clustering_coefficient(X):
    """Compute the local clustering coefficient.

    This clustering coefficient is based on the
    overlap of the edges connected to a given node,
    normalized by the size of the node's neighborhood.

    Parameters
    ----------
    H : Hypergraph
        Hypergraph

    Returns
    -------
    dict
        keys are node IDs and values are the
        clustering coefficients.
    """
    result = {}
    H = xgi.Hypergraph(X.edges.members())

    memberships = H.nodes.memberships()
    members = H.edges.members()

    for n in H.nodes:
        ev = list(memberships[n])
        dv = len(ev)
        if dv <= 1:
            result[n] = 0
        else:
            total_eo = 0
            # go over all pairs of edges pairwise
            for e1, e2 in itertools.combinations(ev, 2):
                edge1 = members[e1]
                edge2 = members[e2]
                # set differences for the hyperedges
                D1 = set(edge1) - set(edge2)
                D2 = set(edge2) - set(edge1)
                # if edges are the same by definition the extra overlap is zero
                if len(D1.union(D2)) == 0:
                    eo = 0
                else:
                    # otherwise we have to look at their neighbours
                    # the neighbours of D1 and D2, respectively.
                    neighD1 = {i for d in D1 for i in H.nodes.neighbors(d)}
                    neighD2 = {i for d in D2 for i in H.nodes.neighbors(d)}
                    # compute extra overlap [len() is used for cardinality of edges]
                    eo = (
                        len(neighD1.intersection(D2)) + len(neighD2.intersection(D1))
                    ) / len(
                        D1.union(D2)
                    )  # add it up
                # add it up
                total_eo = total_eo + eo

            # include normalisation by degree k*(k-1)/2
            result[n] = 2 * total_eo / (dv * (dv - 1))
    return np.mean(list(result.values()))


def assortativity(H):
    return  xgi.degree_assortativity(H, kind = 'top-2')


def degrees_t(hyperedges, freq):
    degrees = []
    for i in range(1,len(hyperedges) , freq):
        degree_t = Counter(itertools.chain.from_iterable(hyperedges[:i]))
        degrees.append(np.array(list(degree_t.values())))
    return degrees
        
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def evolution_nb_nodes(hyperedges: list, freq:int):
    n_t = [] # number of nodes at time t
    nodes_t = set() # set of nodes at time t
    for i in range(0, len(hyperedges), freq):
        batch_ids = set(np.concatenate(hyperedges[i:i+freq]))
        nodes_t |= batch_ids  
        n_t.append(len(nodes_t))
    return(n_t)

def rich_club_normalized(G, n_surrogates=10, Q=10, seed=None):

    # empirical
    rc_emp = nx.rich_club_coefficient(G, normalized=False)

    ks = sorted(rc_emp.keys())
    rc_surr_list = []
    rng = np.random.default_rng(seed)

    for s in range(n_surrogates):
        R = G.copy()
        E = R.number_of_edges()
        nswap =Q * E
        nx.double_edge_swap(R, nswap= nswap, max_tries=10 * nswap, seed=rng)
        rc_rand = nx.rich_club_coefficient(R, normalized=False)
        # align keys; missing ks are treated as 0
        rc_surr_list.append([rc_rand.get(k, 0.0) for k in ks])

    rc_surr_mean = np.mean(np.array(rc_surr_list), axis=0)

    # Normalize
    rc_norm = {}
    for k, emp, base in zip(ks, (rc_emp[k] for k in ks), rc_surr_mean):
        rc_norm[k] = emp / base   # or 1.0, or skip entirely

    return rc_norm, rc_emp, dict(zip(ks, rc_surr_mean))