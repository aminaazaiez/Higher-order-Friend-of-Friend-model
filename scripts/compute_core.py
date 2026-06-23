import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from generator import node_addition_ho_fof
from metrics import core_decomposition
from collections import Counter


def canonicalize_M(M):
    """Turn dict M into a stable, hashable representation."""
    return tuple(sorted((int(k), int(v)) for k, v in M.items()))


def worker(task):
    M, p, seed_M = task 
    H = node_addition_ho_fof(n=n, M=M, p=p, k0=k0, n0=n0)
    core, core_e = core_decomposition(H,multiedges)
    print(f'M={M}, p={p}, seed = {seed_M}')
    return {
        'M': canonicalize_M(M),
        'p':p,
        'k0': k0,
        'n0':n0,
        'core':core,
        'edge_size_counter': Counter(H.edges.size.asdict().values()),
        'seed' : seed_M
    }


n = 10_000
ps = [0,0.5,0.97]
k0=1
n0=20
nb_it =15

multiedges= True
Ms = [
    {2:7},
    {2:4,3:1},
    {2:1,3:2},
    {2:1,4:1}
]

tasks = [
    (M,p,seed_M) for M in Ms
    for p in ps
    for seed_M in range(nb_it)

]

if __name__ == "__main__":

    procs = mp.cpu_count()
    with mp.Pool(processes = procs) as pool:
        rows = list(pool.imap_unordered(worker,tasks))
    
    pd.DataFrame(rows).to_json(f'./out/metrics/core_n_{n}_multiedges_{multiedges}.json')