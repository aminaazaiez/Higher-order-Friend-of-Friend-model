import os
import random
import argparse
import multiprocessing as mp
import joblib
import numpy as np
import xgi
from tqdm.auto import tqdm

from configuration_model import vertex_labeled_MH

def run_and_save_H(param):
    edges, multiedges, n_steps, seed, nb_sim, out_dir = param

    # Prepare kwargs
    cleanup_kwargs = {    
        'isolates' : False,
        'singletons': False,
        'multiedges': multiedges,
        'connected': False, # To test
        'relabel': False,
        'in_place': True}


    # Empirical Hypergraph
    
    H = xgi.Hypergraph(list(edges))
    H.cleanup(**cleanup_kwargs)
    edges_input = H.edges.members()

    MCMC_kwargs = {'edges': edges_input,
                    'n_steps': n_steps,
                    'burnin_steps': 2 * n_steps,
                    'multiedges': multiedges,
                    'n_clash': 0,
                    'seed': seed,
                    'sim': nb_sim
                    }

    H = xgi.Hypergraph(vertex_labeled_MH(**MCMC_kwargs))

    # Save
    fname = f"multi_{multiedges}_n_{n_steps}_sim_{nb_sim}.joblib"
    filepath = os.path.join(out_dir, fname)

    param = {'H': H, 'multiedges': multiedges, 'nsteps': n_steps, 'sim': nb_sim}
    joblib.dump(param, filepath)



if __name__ == "__main__":
    dataset = 'village'

    edges = joblib.load(f'./data/edges_{dataset}.joblib') 
    directory = f'./out/simulations/mcmc_{dataset}'
    
    n_steps = 15000
    nb_it = 100


    param_list = [
        (edges, multiedges, n_steps, random.randint(0, 10**6), i,directory)
        for multiedges in [True]
        for i in range(nb_it)
    ]

    with mp.Pool() as pool:
        pool = mp.Pool()
        pool.map(run_and_save_H, param_list)
        pool.close()