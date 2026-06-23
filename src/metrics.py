import numpy as np
from scipy.stats import skew
import xgi
import networkx as nx
import itertools 
from collections import Counter
from itertools import combinations
from tqdm import tqdm


from sknetwork.data import from_edge_list
from sknetwork.clustering import Louvain, get_modularity
from bisect import bisect_right


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

def overlaps(H, nb_edges: int = 5_000):
    edges = [list(e) for e in H.edges.members()]  
    np.random.shuffle(edges)
    intersect = np.fromiter((len( set(e) & set(f)) for e, f in combinations(edges[:nb_edges], 2)) , dtype=int)
    return Counter(intersect)

def degree_count(H):
    return Counter(H.degree().values())

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

def rich_club(edge_list, degree_dict=None):
    """

    """
    degree_dict = _compute_degrees(edge_list)

    # Precompute min degrees for edges
    min_degrees = [min(degree_dict[node] for node in edge) for edge in edge_list]
    max_k = max(min_degrees)
    sorted_min_degrees = sorted(min_degrees)

    rc = {}

    sorted_min_degrees = sorted(min_degrees)
    for k in range(1, max_k):
        # Binary search to find how many min_degrees are > k
        
        num_edges = len(min_degrees) - bisect_right(sorted_min_degrees, k)
        sum_deg = sum(deg for deg in degree_dict.values() if deg > k) 
        rc[k] = num_edges 
        rc[k] = num_edges / sum_deg if sum_deg > 0 else 0.0

    return rc


def _restrict_m_hypergraph(X : xgi.Hypergraph, m: int, multiedges ):
    ''' m : minimum size of hyperedge'''
    if not multiedges:
        X.merge_duplicate_edges()
    e_remove = [edge for edge in X.edges if X.edges.size[edge] < m]
    X.remove_edges_from(e_remove)
    still_2_remove = len(e_remove) > 0
    return(X, still_2_remove)

def _restrict_k_hypergraph(X : xgi.Hypergraph, k: int, multiedges):
    ''' k : minimum degree of nodes'''
    if not multiedges:
        X.merge_duplicate_edges()
    n_remove = [node for node in X.degree().keys() if X.degree()[node]< k]
    X.remove_nodes_from(n_remove)
    still_2_remove = len(n_remove) > 0
    return(X , still_2_remove)


def core_decomposition(H : xgi.Hypergraph, multiedges) :
    X = H.copy()
    M = range(3 , max(X.edges.size.asnumpy()) + 1 ) # m = edge size
    core = { m : {} for m in M }
    core_e = { m : {} for m in M}

    for m in tqdm(M): # loop for the m shells
        # For each m, start with the initial hypergraph restricted to the edges of size >= m
        k = 1
        X = H.copy()
        while X.num_nodes > 0 : # loop for the k,m shell
            
            X , still_edges_2_remove = _restrict_m_hypergraph(X, m, multiedges)
            X , still_nodes_2_remove = _restrict_k_hypergraph(X, k, multiedges)
            # Store previous shell to compute the k,m shell at the end of the loop

            while  still_nodes_2_remove or still_edges_2_remove : # redo untill there are neither nodes nore edges that can be removed
                if X.num_nodes > 0: # restrict to largest connect component 
                    X = xgi.largest_connected_hypergraph(X)
                X , still_nodes_2_remove = _restrict_m_hypergraph(X, m, multiedges)
                X , still_edges_2_remove = _restrict_k_hypergraph(X, k, multiedges)

            if X.num_nodes > 0 :
                core[m][k] = [max(xgi.connected_components(X), key=len)]
                # # if keep non connected parts
                # core[m][k] = [list(component) for component in xgi.connected_components(X)]
                core_e[m][k] = [list(X.edges)]
                k += 1
    return core, core_e

def hypercoreness(core, g_m):
    c_m = { node : {} for node in core[list(core.keys())[0]][1][0]}
    for m in core.keys():
        k_max = max(core[m].keys())
        for k in core[m].keys():
            if k != k_max :
                k_m_shell = set.union(*core[m][k]) - set.union(*core[m][k+1])
            else :
                k_m_shell = set.union(*core[m][k])

            for node in k_m_shell:
                c_m[node][m] = k/k_max * g_m[m]


    R_i = {node : sum( [ c_m[node][m] for m in c_m[node].keys() ] ) for node in c_m.keys()}

    return(R_i)