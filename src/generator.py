import itertools
from collections import Counter
import numpy as np
import networkx as nx

import xgi
from tqdm import tqdm

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

def _sample_without_replacement(pool, k, rng):
    """pool: iterable; returns a set of size <= k (fallback to all if insufficient)."""
    pool = list(pool)
    if k <= len(pool):
        return set(rng.choice(pool, k, replace = False))
    return set(pool)

def _sample_with_replacement(pool, k, rng):
    """pool: iterable; returns a set of size <= k (fallback to all if insufficient)."""
    pool = list(pool)
    if k <= len(pool):
        return list(rng.choice(pool, k, replace = True))
    return pool


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
    # Add edges
    for d, p_new in zip(D, P_new):

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

def node_addition_ho_fof(
    n: int,
    M: dict,
    p: float,
    k0 : int = 1,
    multiedges: bool = True,
    n0: int =20,
    seed: int | None = None,
    ):
    """
    Parameters
    ----------
    n : int
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
            if len(e) != d:
                print(f'edge size = {len(e)}, requested size  ={d}, time = {i}')
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


    # rng = np.random.default_rng(seed)

    # Initialize a random Erdos and Renyi like connected hypergraph .
    if not n0:
        n0=max(D)

    p0 = k0/n0
    H = _connected_er_like_hypergraph(n0 = n0, p0=p0, seed = seed)
    i = n0
    e_0 = H.num_edges

    for d, p_new in zip(D, P_new):
        r = np.random.random()
        is_friend = r < p

        count_while_loop = 0

        e = set()
        while len(e) !=d: # try to create edge e with edge size d
            count_while_loop +=1
            p_new_tild = min(1, d/(d-1)*p_new)
            n_new = np.random.binomial(d-1, p_new_tild)
            # -----------Phase 1: add new nodes-----------
            # Add n_new nodes in e
            e = set(range(i, i + n_new))

            # ---------- Phase 2: Choose d- n_new nodes in the previous nodes list----------
            # set encounter type (fof, or random)
            

            if is_friend: # friend encounter
                # set Target set
                j = np.random.choice(i) # Choose j uar among old nodes
                if redirection:
                    targetset = H.nodes.neighbors(j)
                else:
                    targetset = H.nodes.neighbors(j) |{j}
                # Sample in target set
                S = _sample_without_replacement(targetset, d-n_new, np.random)
                encounter_type = 'f'

            else: # random encounter
                S = set(np.random.choice(i, size = d-n_new, replace = False))
                encounter_type = 'r'

            e |= S

            # Test
            infinite_loop = count_while_loop > 10000
            if infinite_loop: # if the loop is infinite, move the edge at the end of the list 
                D.append(d)
                P_new.append(p_new)
                break
                # print(f'type of meeting = {encounter_type}, edge number = {itt}, r = {r:.2f}, p_new = {p_new:.2f}, n_new = {n_new},  d = {d}')
                # # for n_new_i, targetset_size in zip(list_n_new, list_target_set):
                # #     print(f'n_new = {n_new_i}, size of target set = {targetset_size}, sum = {n_new_i + targetset_size}, d = {d}'  )
                # return 'infinite loop'


        if not infinite_loop: # Add edge e
            assert len(e) == d
            H.add_edge(e, idx = f'e_{H.num_edges}')

            # Increment node label
            i += n_new
    return(H)


def node_addition_ho_fof_resampled_T(
    n: int,
    M: dict,
    p: float,
    k0 : int = 1,
    multiedges: bool = True,
    n0: int =20,
    seed: int | None = None,
    ):
    """
    In this version the model resamplesthe Targert set at each time 
    Parameters
    ----------
    n : int
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

        # ---------- Phase 2: friend vs random encounters ----------
        # Per-node pool Di (reset)
        Di = base_multiset.copy()
        rng.shuffle(Di)

        while Di:
            d = Di[-1]
            is_friend = rng.random() < p

            if is_friend:  # friend encounter
                # ---------- Phase 1: target set ----------
                # Target node
                j = rng.choice(i)
                targetset = (H.nodes.neighbors(j) | {j} )- {i}
                S = _sample_without_replacement(targetset, d-1, rng)
            else:                 # random encounter
                S = set(rng.choice(i, size=d-1, replace=False))

            e = {i} | S
            if len(S) == 0:
                print(targetset)
                break
            if len(e) == 1:
                print('edge of size 1 is added',j,i,S)
                break
            if not multiedges:
                if e in H.edges.members():
                    continue
            if len(e) != d:
                print(f'edge size = {len(e)}, requested size  ={d}, time = {i}')
            H.add_edge(e, idx = f'e_{H.num_edges}')

            Di.pop()

    return H
