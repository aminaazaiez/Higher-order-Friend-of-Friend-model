
import os
import numpy as np
import networkx as nx
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd

import xgi
from metrics import rich_club_normalized, rich_club, degree_count
from generator import node_addition_ho_fof
from configuration_model import vertex_labeled_MH


def graph_rc(task):
    M, seed_M, seed_swap= task
    H = node_addition_ho_fof(n=n, M=M, p=p, k0=k0, n0=n0, seed = seed_M)
    # Create a projected graph
    G = xgi.to_graph(H)
    rng = np.random.default_rng(seed_swap)
    # compute rich club
    R = G.copy()
    E = R.number_of_edges()
    nswap =Q * E
    nx.double_edge_swap(R, nswap= nswap, max_tries=10 * nswap, seed=rng)
    rc = nx.rich_club_coefficient(R, normalized=False )
    rc_emp = nx.rich_club_coefficient(G, normalized=False)
    return {"M": M, "seed_M": seed_M, "seed_swap":seed_swap, "rc": rc, 'rc_emp': rc_emp }

def ho_rc(task):
    M, seed_M, seed_swap= task
    H = node_addition_ho_fof(n=n, M=M, p=p, k0=k0, n0=n0, seed = seed_M)
    edges_input = H.edges.members()
    # mcmc configuration model
    rng = np.random.default_rng(seed_swap)
    E = H.num_edges
    nswap = Q * E

    MCMC_kwargs = {'edges': edges_input,
                'n_steps': nswap,
                'burnin_steps': 2 * nswap,
                'multiedges': True,
                'n_clash': 0,
                'seed': seed_swap,
                }

    H_cm = xgi.Hypergraph(vertex_labeled_MH(**MCMC_kwargs))
    # compute rich club
    rc = rich_club(degree_count(H) )
    rc_emp = rich_club(degree_count(H_cm) )
    return {"M": M, "seed_M": seed_M, "seed_swap":seed_swap, "rc": rc, 'rc_emp': rc_emp }



def run_parallel(Ms, nb_it, n, p, k0, n0, already_done,processes=None):
    tasks = []
    for M in Ms:
        M_key = canonicalize_M(M)
        for seed_M in range( nb_it):
            pending_swaps = [
                seed_swap for seed_swap in range(N_surrogates)
                if (M_key, seed_M, seed_swap) not in already_done
            ]
            if not pending_swaps:
                continue
            
            tasks.extend([(M, seed_M, seed_swap) for seed_swap in pending_swaps])

    procs = processes or mp.cpu_count()

    with mp.Pool(processes=procs) as pool:
        rows = list(tqdm(pool.imap_unordered(graph_rc, tasks, ), total=len(tasks)))

    return pd.DataFrame(rows)

def canonicalize_M(M):
    """Turn dict M into a stable, hashable representation."""
    return tuple(sorted((int(k), int(v)) for k, v in M.items()))


n = 10_000

n0 =  20
p = 0.97
k0 = 1
nb_it = 400

Q = 30
N_surrogates = 10 


Ms = [
    {2:7},
    {2:4,3:1},
    {2:1,3:2},
    {2:1,4:1}
]

metrics_path = f'./out/metrics/ho_rc_n_{n}_p_{p}_n0_{n0}_k0_{k0}_Q_{Q}.json'

if __name__ == "__main__":

    if os.path.exists(metrics_path):
        df_metrics = pd.read_json(metrics_path)
        df_metrics['M'] = df_metrics['M'].apply(lambda d: {int(k): v for k, v in d.items()})
        df_metrics['rc'] = df_metrics['rc'].apply(lambda d: {int(k): v for k, v in d.items()})

        already_done = {
            (canonicalize_M(M), int(seed_M), int(seed_swap))
            for M, seed_M, seed_swap in zip(df_metrics['M'], df_metrics['seed_M'], df_metrics['seed_swap'])
        }

    else:
        df_metrics = pd.DataFrame()
        already_done = set()

  

    result = run_parallel(Ms, nb_it, n, p, k0, n0, already_done=already_done)

    df_metrics = pd.concat([df_metrics, result], ignore_index= True)


    df_metrics.to_json(metrics_path)


