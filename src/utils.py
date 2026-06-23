
import numpy as np
import math
import joblib
import xgi
import networkx as nx
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt




def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def XGI_2_nxBipartite (H : xgi.Hypergraph) :
    B = nx.Graph()

    B.add_nodes_from(H.nodes, bipartite='entity')
    for e in H.edges:
        B.add_node(e , bipartite='meeting',
             #enc_type = H.edges[e]['enc_type']
             )
    for node in H.nodes:
        for edge in H.nodes.memberships(node):
            B.add_edge(node, edge)
    return B


def plot_map_heatmap(
    df, feature, x, y, row=None, col=None, col_wrap=None,
    cmap="viridis", scale="quantile", q=(0.02, 0.98), k=2,
    annot=True, fmt=".1f", annot_kws=None, height=7, aspect=1.1, agg_func = 'mean'
):
    pivot = df.pivot_table(index=y, columns=x, values=feature, aggfunc = agg_func, observed = True)
    vals = pivot.values[np.isfinite(pivot.values)]
    if scale == "quantile":
        vmin, vmax = np.quantile(vals, q)
    elif scale == "std":
        mu, sigma = np.mean(vals), np.std(vals)
        vmin, vmax = mu - k * sigma, mu + k * sigma
    elif scale == "range":
        vmin, vmax = np.min(vals), np.max(vals)
    else:
        raise ValueError("scale must be 'quantile', 'std', or 'range'.")
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    master_x = sorted(df[x].round(3).unique())
    master_y = sorted(df[y].round(3).unique()) 

    if not col:
        col_wrap = None
        
    g = sns.FacetGrid(df, row=row, col=col, 
        col_wrap=col_wrap, 
        height = 7,  aspect=aspect)

    g.map_dataframe(
        _heatmap, feature=feature, x=x, y=y,
        master_x=master_x, master_y=master_y,
        vmin=vmin, vmax=vmax,agg_func=agg_func,
        cmap="viridis", cbar=True, annot=True, fmt=fmt,
    )
    # Set title
    g.fig.suptitle(feature)

    # g.fig.subplots_adjust(top=0.8)



def _heatmap(data, feature, x, y, master_x, master_y, 
    vmin, vmax, agg_func,
    **kwargs
    ):
    if data.empty:
        return

        # Round pr and pn to 3 decimals for consistent display
    df = data.copy()
    df[x] = df[x].round(3)
    df[y] = df[y].round(3)

    pivot = df.pivot_table(index=y, columns=x, values=feature, aggfunc = agg_func, observed = True)

    pivot = pivot.reindex(index=master_y, columns=master_x) 

    ax = sns.heatmap(pivot, 
        vmin = vmin, vmax = vmax, 
        **kwargs)
    ax.invert_yaxis()
    




def iter_Ms( pair_cap=20, min_events=2, m_max=None):
    # Only degrees that can possibly fit under the pair capacity
    Ds = [d for d in range(2, pair_cap + 1) if math.comb(d, 2) <= pair_cap]
    costs = {d: math.comb(d, 2) for d in Ds}

    def dfs(i, rem_pairs, events, curr):
        # Prune: even if we spent all remaining pairs on d=2 (cost 1 per edge),
        # can we still reach the minimum number of edges (events)?
        if events + rem_pairs < min_events:
            return

        if i == len(Ds) or rem_pairs == 0:
            if events >= min_events:
                # Yield only nonzero entries for cleanliness
                yield {d: md for d, md in curr.items() if md > 0}
            return

        d = Ds[i]
        c = costs[d]
        upper = rem_pairs // c                   # max md allowed by pair cap
        if m_max is not None:
            upper = min(upper, m_max)            # optional per-d cap

        for md in range(upper + 1):
            curr[d] = md
            yield from dfs(i + 1, rem_pairs - md * c, events + md, curr)
        curr.pop(d, None)

    yield from dfs(0, pair_cap, 0, {})


def clean_edges(edges):
    edges = [list(e) for e in edges ]     
    node_labels = {}     # keep order within edges
    next_id = 0
    for e in edges:
        for node in e:
            if node not in node_labels:        # first time we see it
                node_labels[node] = next_id
                next_id += 1

    edges_relabeled = [{node_labels[node] for node in e} for e in edges]
    return edges_relabeled


def get_param_from_emp(
    edges,
    t_i : int = 0, 
    shuffle: bool = False, 
    seed = None):
    
    _edges = edges.copy()
    # Shuffle edges 
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(_edges)
        
    # Get edge size sequence
    D =  [len(e) for e in _edges]
    # p_new as a function of time.
    nodes= set()
    p_new =[]
    for e in _edges:
        new_nodes = set(e) - nodes
        p_new.append(len(new_nodes)/len(e))
        nodes|= new_nodes

    return D, p_new

def calibrate_from_emp(edges, n0, shuffle=True, seed=None):

    rng = np.random.default_rng(seed)
    _edges = edges.copy()
    if shuffle:
        rng.shuffle(_edges)

    D = [len(e) for e in _edges]

    # cumulative unique nodes curve
    seen = set()
    Ncum = []
    for e in _edges:
        seen |= set(e)
        Ncum.append(len(seen))

    # find t0 s.t. N_emp(t0) ~= n0
    t0 = int(np.argmin(np.abs(np.array(Ncum) - n0)))

    # recompute P_new starting at t0 (and with a "known old set" = nodes seen up to t0)
    old = set()
    P_new = []
    for e in _edges: #[t0:]:
        new_nodes = set(e) - old
        P_new.append(len(new_nodes) / len(e))
        old |= new_nodes
    return D[t0:], P_new[t0:], t0
    
def rescale_pnew(D, P_new, n0, N_emp_final):
    target_add = N_emp_final - n0
    denom = sum((d-1) * p for d, p in zip(D, P_new))
    s = target_add / denom if denom > 0 else 1.0
    P_new_scaled = [min(1.0, s * p) for p in P_new]
    return P_new_scaled, s

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two discrete probability distributions p and q.

    Parameters
    ----------
    p : array_like
        The first discrete probability distribution.
    q : array_like
        The second discrete probability distribution.

    Returns
    -------
    kl : float
        The Kullback-Leibler divergence between p and q.
    """
    length = max(len(p), len(q))
    p = np.pad(p, (0, length - len(p)))
    q = np.pad(q, (0, length - len(q)))

    # Normalization
    p = np.array(p) / np.sum(p) if np.sum(p) > 0 else np.zeros_like(p)
    q = np.array(q) / np.sum(q) if np.sum(q) > 0 else np.zeros_like(q)

    # Avoid log(0)
    mask = (p > 0) & (q > 0)
    p_masked = p[mask]
    q_masked = q[mask]

    return np.sum(p_masked * np.log(p_masked / q_masked))