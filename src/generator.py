import itertools
from collections import Counter
import numpy as np
import networkx as nx

import xgi
from tqdm import tqdm



def random_hypergraph(D:list,
    P_new : list,
    k0:int = 1,
    n0:int | None = None,
    seed: int |None = None):
    ''' 
    Generate a random hypergraph with a given edge size distribution
        
    Parameters
    ----------
    P_new : list
        Sequence of probability of adding a new node at time t 
    D : list
        sequence of edge size
    p : float
        Probability of a "friend encounter" (vs random encounter).
    k0 : int 
        average degree of intial hypergraph
    n0 : int or None, default=None
        Initial number of nodes in the seed hypergraph.
    rng_seed : int or None, default=None
        Random seed used for reproducibility.

    Returns
    -------
    H : xgi.Hypergraph
    '''
        
    # Relabel nodes and return the maximal connected component
    if any((d < 2) for d in D):
        raise ValueError("All edge sizes d in M must satisfy d >= 2.")

    rng = np.random.default_rng(seed)
    
    # Initialize an Erdos and Renyi connected hypergraph
    if not n0:
        n0=max(D)

    p0 = k0/n0
    H = _connected_er_like_hypergraph(n0 = n0, p0=p0, seed = seed)
    i=n0 # label of the new node
    
    # Add edges
    for d, p_new in zip(D, P_new):
        
        # Draw the number of new nodes 
        p_new_tild = min(1, d/(d-1)*p_new)
        n_new = rng.binomial(d-1, p_new_tild)
                
        # Add n_new nodes in e
        e = set(range(i, i + n_new))
        
        # Choose d- n_new nodes in the previous nodes list
        e |= set(rng.choice(i, size = d-n_new, replace = False))
        
        # Add e to the set of edges
        H.add_edge(e, idx = f'e_{H.num_edges}')
        # Increment node label
        i+=n_new
    return H 

def preferential_attachment(
    D:list,
    P_new : list,
    k0:int = 1,
    n0:int | None = None,
    seed: int |None = None):
    ''' 
    Generate a random hypergraph with a given edge size distribution
        
    Parameters
    ----------
    P_new : list
        Sequence of probability of adding a new node at time t 
    D : list
        sequence of edge size
    p : float
        Probability of a "friend encounter" (vs random encounter).
    k0 : int 
        average degree of intial hypergraph
    n0 : int or None, default=None
        Initial number of nodes in the seed hypergraph.
    rng_seed : int or None, default=None
        Random seed used for reproducibility.

    Returns
    -------
    H : xgi.Hypergraph
    '''
    # Relabel nodes and return the maximal connected component
    if any((d < 2) for d in D):
        raise ValueError("All edge sizes d in M must satisfy d >= 2.")

    rng = np.random.default_rng(seed)
    
    # Initialize an Erdos and Renyi like hypergraph
    if not n0:
        n0=max(D)

    p0 = k0/n0
    H = _connected_er_like_hypergraph(n0 = n0, p0=p0, seed = seed)
    i=n0 # label of the new node
    e_0 = H.num_edges

    # Pool of nodes. Each node is duplicated k_i times 
    duplicated_nodes = list(itertools.chain.from_iterable(H.edges.members()))
    degrees = H.degree()
    # Add edges
    for d, p_new in tqdm(zip(D, P_new), total= len(D)):
        
        # Draw the number of new nodes
        p_new_tild = min(1, d/(d-1)*p_new)
        n_new = rng.binomial(d-1, p_new_tild)
        
        # Add n_new nodes in e
        e = set(range(i, i + n_new))
        
        # Choose d- n_new nodes in the previous nodes list
        
        e |= _random_subset(duplicated_nodes, d - n_new, rng)

        # Add e to the set of hyperedges
        H.add_edge(e, idx = f'e_{H.num_edges}')

        # Add nodes in e to duplicated_nodes
        duplicated_nodes += e

        # Increment node label
        i+=n_new
    return H 

def _dependent_node_sampling_pref_overlaps(e_0, H, alpha , beta_neigh = 1, beta_remaining = 1):
    # Get degrees of nodes
    degree = H.degree()
    # Get sets N_e and rem
    all_nodes = set(degree)
    supersets = [e for e in H.edges.members() if e_0 < e]
    N_e = set().union(*supersets) - e_0
    rem  = all_nodes - e_0 - N_e
    # Compute un‑normalized weights
    weights = {}
    for i in N_e:
        weights[i] = alpha * (degree[i]**beta_neigh)
    for i in rem:
        weights[i] = (1 - alpha) * (degree[i]**beta_remaining)
    # Normalize wieghts
    total_w = sum(weights.values())
    if total_w == 0:
        return np.random.choice(list(weights.keys()))
        
    probs = [w/total_w for w in weights.values()]
    nodes = list(weights.keys())
    node = np.random.choice(nodes, p=probs)
    return node

def random_overlaps_pref(M, alpha, beta_neigh, beta_remaining,
    D:list,
    P_new : list,
    k0:int = 1,
    n0:int | None = None,
    seed: int |None = None):

    # Initialize
        # Relabel nodes and return the maximal connected component
    if any((d < 2) for d in D):
        raise ValueError("All edge sizes d in M must satisfy d >= 2.")

    rng = np.random.default_rng(seed)
    
    # Initialize an Erdos and Renyi like connected hypergraph
    if not n0:
        n0=max(D)

    p0 = k0/n0
    H = _connected_er_like_hypergraph(n0 = n0, p0=p0, seed = seed)
    i=n0 # label of the new node
    
    for d, p_new in tqdm(zip(D, P_new), total= len(D)):
        e = set()
        # Draw the number of new nodes 
        n_new = np.random.binomial(d, p_new) # Number of new nodes that will belong to e
        # Choose d- n_new nodes in the previous nodes list
        for i in range(d-n_new):
            node = _dependent_node_sampling_pref_overlaps(e, H, alpha, beta_neigh, beta_remaining)
            e.add(node)
        if n_new:
            # Add n_new nodes in e
            label_new_node = max(H.nodes) + 1 #label of the first new node  
            e |= (set(range(label_new_node, label_new_node + n_new)))
        # Add e to the set of hyperedges
        H.add_edge(e, idx = f'e_{H.num_edges}')
    return H



def temporal_triadic_closure(N, M, eta, l, xi):
    # Initialize with empty network 
    H_t = xgi.Hypergraph() # temporal network
    H_t.add_nodes_from(list(range(N)))
    H = H_t.copy() # Aggregated network

    for _ in tqdm(range(M)):
        e_remove = []
        e_add = []
        # Delete edge
        for e in H_t.edges:
            p = np.random.random()
            if p < l:
                e_remove.append(e)
        for i in H_t.nodes:
            p =np.random.random()
            # Random encounter
            if p < eta:
                j = np.random.choice(H_t.nodes)
                i = np.random.choice(list(H_t.nodes - set([i])))
                e_add.append({i,j})
            # Network based encounter
            if p < xi:
                N_i = H_t.nodes.neighbors(i)
                if N_i:
                    j = np.random.choice(list(N_i))
                    N_j = H.nodes.neighbors(j)
                    if N_j - N_i - set([i]):
                        k = np.random.choice(list(N_j - N_i - set([i])))
                        e_add.append({i,k})
        # Add and remove nodes
        H_t.remove_edges_from(e_remove)
        H_t.add_edges_from({ f'e_{H.num_edges}':e for e in e_add})
        H.add_edges_from({ f'e_{H.num_edges}': e for e in e_add})
    return H



def temporal_polyadic_closure(N, M, mu, p_r, p_n, d ):
    def _neighborhood(H: xgi.Hypergraph, e: set):
        #Get 1rst degree neighbors of nodes in e
        neighborhoods_1 = [H.nodes.neighbors(i) for i in e] 
        # N_e = union of N_i
        N_e_1 = set().union(*neighborhoods_1) 
        # Get 2nd degree neighbors of nodes
        neighborhoods_2 = [H.nodes.neighbors(i) for i in N_e_1] 
        # N_e = union of N_i
        N_e_2 = set().union(*neighborhoods_2) 
        # N_e 
        N_e = N_e_1 | N_e_2
        return N_e - e
        
    seed = 10
    np.random.seed(seed)
    # Initialize with empty network
    H_t = xgi.Hypergraph() # temporal network
    H_t.add_nodes_from(list(range(N)))
    H = H_t.copy() # Aggregated network
    t = 0 # time index
    desc_str = f"N={N}, M={M}, μ={mu:.3f}, pr={p_r:.3f}, pn={p_n:.3f}, d={d}"
    pbar = tqdm(total=M, desc=desc_str, dynamic_ncols=True)
    while H.num_edges < M:
        # Choose a node randomly
        e = set([np.random.choice(H_t.nodes)]) 
        for _ in range(d-1):
            N_e = _neighborhood(H_t, e)
            p = np.random.random()
            if N_e:
                # Random encounter
                if p < p_r:
                    node = np.random.choice(list(set(H_t.nodes ) - e ))
                    e.add(node)            
                # Meet a friend of friend
                else:
                    if p < p_n :
                        node = np.random.choice(list(N_e))
                        e.add(node)    
            else:
                # Random encounter
                if p < p_r:
                    node = np.random.choice(list(set(H_t.nodes ) -e )) 
                    e.add(node)    
        # Add e to the set of edges 
        if len(e)>1 and e not in H_t.edges.members():
            H_t.add_edge(e, idx = f'e_{H.num_edges}')
            H.add_edge(e, idx = f'e_{H.num_edges}')
            t +=1
            pbar.update(1)
        # Remove a random edge
        if H_t.num_edges >= mu:
            e_remove = np.random.choice(H_t.edges)
            H_t.remove_edge(e_remove)
    pbar.close()
    return H 

def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets

def triadic_closure(n, m, p, seed=None ):
    ''' Triadic closure alrgoithm for growing graphs
    
    Parameters
    ----------
    n : int
        the number of nodes
    m : int
        the number of random edges to add for each new node
    p : float
        Probablity of adding a triangle after adding a random edge
    seed :  integer, random_state, or None (default)
        Indictor of random number generation state
    
    Returns:
    -------
    H : xgi.Hypergraph()
        The generated Hypergraph with edges of size 2.

    Raises:
    -------
        error of 'm' does not satisfy "1<=m<=n" or "p" does not satisfy "0<=p<=1"
    
    '''
    assert  n > m
    assert p > 0 and p < 1

    # set random state
    rgn = np.random.default_rng(seed)
    # create an empty graph with n0 nodes
    n0 = m
    H = xgi.Hypergraph()
    H.add_nodes_from(list(range(n0)))
    source = len(H.nodes) # next node is n0

    while source < n:
        nodes = list(H.nodes)
        if len(nodes) < m:
            raise ValueError("Not enough nodes to sample targets from.")

        possible_targets = _random_subset(nodes, m, rgn) # to choose from if random encoutner
        # Choose a target node to connect with
        target = possible_targets.pop()
        e = {source, target}
        H.add_edge(e, idx = f'e_{H.num_edges}')
        count = 1
        while count < m:
            # meet friend of friend
            if rgn.random() < p:
                neighborhood = [
                    nbr 
                    for nbr in H.nodes.neighbors(target) 
                    if nbr not in H.nodes.neighbors(source) and  nbr != source
                    ]
                if neighborhood: 
                    nbr = rgn.choice(neighborhood)
                    e = {source, nbr}
                    H.add_edge(e, idx = f'e_{H.num_edges}')
                    count +=1
                    continue
            # else random encounter
            target = possible_targets.pop()
            e = {source, target}
            H.add_edge(e, idx = f'e_{H.num_edges}')
            count += 1
        source += 1
    return H


def m_polyadic_closure(n, m, p, seed=None ):
    
    assert m > 1 and n > m
    assert p > 0 and p < 1

    # set random state
    rgn = np.random.default_rng(seed)
    # create an empty graph with n0 nodes
    n0 = m
    H = xgi.Hypergraph()
    H.add_nodes_from(list(range(n0)))

    source = len(H.nodes) # next node
    nodes = list(H.nodes)

    while source < n:
        # Choose a target node to connect with
        target = rgn.choice(nodes)
        e = {source, target}
        # fille edge e 
        while len(e) < m :
            # meet friend of friend
            if rgn.random() < p:
                neighborhood = [
                    nbr 
                    for nbr in H.nodes.neighbors(target) 
                    if nbr not in e
                    ]
                if neighborhood: 
                    nbr = rgn.choice(neighborhood)
                    e.add(nbr)
                    continue
             
            # else random encounter
            remaining = list(set(nodes) - e)
            target = rgn.choice(remaining)
            e.add(target)
        # Add edge of size m
        H.add_edge(e, idx = f'e_{H.num_edges}')
        nodes.append(source)
        source += 1
    
    return H

def _sample_without_replacement(pool, k, rng):
    """pool: iterable; returns a set of size <= k (fallback to all if insufficient)."""
    pool = list(pool)
    if k <= len(pool):
        return set(rng.choice(pool, k, replace = False))
    return set(pool)


def _sample_fill(primary_pool, backup_pool, k, rng):
    """Try to sample k from primary; if short, fill from backup (without overlaps)."""
    S = _sample_without_replacement(primary_pool, k, rng)
    if len(S) < k:
        need = k - len(S)
        backup = list(set(backup_pool) - S)
        S.update(rng.choice(backup, need, replace = False))
    return S


def _neighbors_1hop(H: xgi.Hypergraph, S):
    """Return S union its 1-hop neighbors in H."""
    U = set(S)
    for node in S :
        U.update(H.nodes.neighbors(node))
    return U


def _neighbors_hops(H: xgi.Hypergraph, S, hops: int):
    """Return union n-hop neighbors in H of S union S."""
    U = set(S)
    for _ in range(hops):
        U.update(_neighbors_1hop(H, U))
    return U 


def _connected_er_like_hypergraph(n0, p0, seed=None):
    rng = np.random.default_rng(seed)
    G = nx.random_labeled_tree(n0, seed= seed)
    for u, v in itertools.combinations(range(n0), 2):
        if not G.has_edge(u, v) and rng.random() < p0:
            G.add_edge(u, v)
    edges = {f"e_{i}": {u, v} for i, (u, v) in enumerate(G.edges())}
    return xgi.Hypergraph(edges)

def polyadic_closure_1(
    n: int,
    M: dict,
    p: float,
    hops: int = 1,
    dynamic_target: bool = False,
    multiedges: bool = True,
    n0: int | None = None,
    seed: int | None = None, ):
    """
    Grow an XGI hypergraph according to your triadic-closure-on-hypergraphs model.

    Parameters
    ----------
    total_nodes : int
        Final number of nodes |V| in the returned hypergraph.
    M : dict
        Edge-size multiset per new node, e.g. {2: m2, 3: m3, ..., dmax: mdmax}.
        Interpreted as a multiset Di that resets for each newly added node.
    p : float
        Probability of a "friend encounter" (vs random encounter).
    hops : int, default=1
        1 or 2: depth of the friend neighborhood N^{(h)} used for closure.
    dynamic_target : bool, default=False
        If True, the target set S* grows by including nodes from each friend edge.
        If False, it stays fixed (either T or e1 depending on seed_target_is_edge).
    multiedges : bool, default=False
        If True, multiplicities are accumulated in an edge attribute `weight`.
        If False, repeated identical hyperedges are skipped (simple hypergraph).
    n0 : int or None, default=None
        Initial number of nodes in the seed hypergraph. If None, uses the *minimal*
        value n0 = max(d in M) - 1 so that the first sampling is feasible.
    rng_seed : int or None, default=None
        Random seed used for reproducibility.

    Returns
    -------
    H : xgi.Hypergraph
        The grown hypergraph.

    """
    if any((d < 2) for d in M.keys()):
        raise ValueError("All edge sizes d in M must satisfy d >= 2.")


    rng = np.random.default_rng(seed)

    # Build the per-node multiset as a list (respects counts m_d).
    base_multiset = []
    for d, m in M.items():
        base_multiset.extend([d] * m)

    if n0 is None:
        n0 = int(0.1*n)
        

    # Initialize an empty hypergraph with n0 nodes and no edges.
    H = xgi.Hypergraph()
    H.add_nodes_from(range(n0))

    # Main growth loop: add nodes i = n0 ... n
    for i in range(n0 , n ):

        # Per-node pool Di (reset)
        Di = base_multiset.copy()

        # ---------- Phase 1: seed edge ----------
        rng.shuffle(Di)
        #d = Di.pop()
        V_prev = set(H.nodes) - {i}
        # Target set T of size (d-1)
        #targetset = _sample_without_replacement(V_prev, d - 1, rng)
        targetset = _sample_without_replacement(V_prev, 1, rng)
        # # Seed edge e1
        # e1 = {i} | targetset
        # # Add edge e1
        # H.add_edge(e1, idx = f'e_{H.num_edges}')


        # ---------- Phase 2: friend vs random encounters ----------
        while Di:
            d = Di[-1]
            V_prev = set(H.nodes) - {i}
            is_friend = rng.random() < p

            if is_friend:  # friend encounter
                U = _neighbors_hops(H, targetset, hops=hops) - {i}
                S = _sample_fill(U, V_prev, d - 1, rng)
            else:                 # random encounter
                S = _sample_without_replacement(V_prev, d - 1, rng)

            e = {i} | S
            
            if not multiedges and e in H.edges.members():
                    continue
            
            H.add_edge(e, idx = f'e_{H.num_edges}')
            Di.pop()

            if dynamic_target:
                targetset |= S
    #H.cleanup(relabel= False, connected= True)
    return H


def node_addition_ho_fof(
    n: int,
    M: dict,
    p: float,
    k0 : int = 1,
    multiedges: bool = True,
    n0: int | None = None,
    seed: int | None = None,
    ):
    """
    Parameters
    ----------
    total_nodes : int
        Final number of nodes |V| in the returned hypergraph.
    M : dict
        Edge-size multiset per new node, e.g. {2: m2, 3: m3, ..., dmax: mdmax}.
        Interpreted as a multiset Di that resets for each newly added node.
    p : float
        Probability of a "friend encounter" (vs random encounter).
    multiedges : bool, default=False
        If True, multiplicities are accumulated in an edge attribute `weight`.
        If False, repeated identical hyperedges are skipped (simple hypergraph).
    n0 : int or None, default=None
        Initial number of nodes in the seed hypergraph. If None, n0 = 10 %  of the total number of nodes
    rng_seed : int or None, default=None
        Random seed used for reproducibility.

    Returns
    -------
    H : xgi.Hypergraph
        The grown hypergraph.

    """
    if any((d < 2) for d in M.keys()):
        raise ValueError("All edge sizes d in M must satisfy d >= 2.")


    rng = np.random.default_rng(seed)

    # Build the per-node multiset as a list (respects counts m_d).
    base_multiset = []
    for d, m in M.items():
        base_multiset.extend([d] * m)


    # Initialize an empty hypergraph with n0 nodes and no edges.
    p0 = k0/n0
    H = _connected_er_like_hypergraph(n0 = n0, p0=p0, seed = seed)
    n0 = H.num_nodes

    # Main growth loop: add nodes i = n0 ... n
    for i in range(n0 , n ):
        V_prev = set(H.nodes)


        # ---------- Phase 1: target set ----------
        # Target node
        j = rng.choice(i)
        targetset = H.nodes.neighbors(j) | {j}

        # ---------- Phase 2: friend vs random encounters ----------
        # Per-node pool Di (reset)
        Di = base_multiset.copy()
        rng.shuffle(Di)

        while Di:
            d = Di[-1]
            is_friend = rng.random() < p

            if is_friend:  # friend encounter
                S = _sample_without_replacement(targetset, d-1, rng)
            else:                 # random encounter
                S = set(rng.choice(i, size=d-1, replace=False))
            
            e = {i} | S
            
            if not multiedges: 
                if e in H.edges.members():
                    continue

            H.add_edge(e, idx = f'e_{H.num_edges}')

            Di.pop()

    return H



def edge_addition_ho_fof(
    D: list,
    p: float,
    P_new: list,
    redirection = False,
    k0: int = 1,
    n0: int | None = None,
    seed: int | None = None,
):
    """

    Parameters
    ----------
    P_new : list
        Sequence of probability of adding a new node at time t 
    D : list
        sequence of edge size
    p : float
        Probability of a "friend encounter" (vs random encounter).
    k0 : int 
        average degree of intial hypergraph
    n0 : int or None, default=None
        Initial number of nodes in the seed hypergraph.
    rng_seed : int or None, default=None
        Random seed used for reproducibility.

    Returns
    -------
    H : xgi.Hypergraph

    """
    if any((d < 2) for d in D):
        raise ValueError("All edge sizes d in M must satisfy d >= 2.")


    rng = np.random.default_rng(seed)
    

    # Initialize a random Erdos and Renyi like connected hypergraph .
    if not n0:
        n0=max(D)

    p0 = k0/n0
    H = _connected_er_like_hypergraph(n0 = n0, p0=p0, seed = seed)
    i=n0
    e_0 = H.num_edges
       
    for d, p_new in zip(D, P_new):
        e = {}
        while len(e) !=d:
            p_new_tild = min(1, d/(d-1)*p_new)
            n_new = rng.binomial(d-1, p_new_tild)
            # -----------Phase 1: add new nodes-----------
            # Add n_new nodes in e
            e = set(range(i, i + n_new))

            # Choose d- n_new nodes in the previous nodes list
            
            # ---------- Phase 2: target set----------
            # Target node
            j = rng.choice(i) # Choose j uar among old nodes
            if redirection:
                targetset = H.nodes.neighbors(j)
            else:
                targetset = H.nodes.neighbors(j) |{j}

            # ---------- Phase 2: friend vs random encounters ----------
            is_friend = rng.random() < p

            if is_friend: # friend encounter
                S = _sample_without_replacement(targetset, d-n_new, rng)
                type = 'f'
            else:
                S = set(rng.choice(i, size = d-n_new, replace = False))
                type = 'r'
            
            e |= S
        

        # Add edge e
        assert len(e) == d
        H.add_edge(e, idx = f'e_{H.num_edges}')
        
        # Increment node label
        i += n_new
    return(H)

