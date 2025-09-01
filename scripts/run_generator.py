import numpy as np
import multiprocessing as mp
import joblib
import os

from utils import iter_Ms

from generator import (preferential_attachment,
                    random_overlaps_pref,
                    triadic_closure,
                    polyadic_closure_2,
                    temporal_triadic_closure,
                    temporal_polyadic_closure)

# Save simulation
def param_to_filename(directory, param_dict):
    def fmt(val):
        return f"{val:.3f}" if isinstance(val, float)  else str(val)
    filename = "_".join(f"{key}-{fmt(val)}" for key, val in param_dict.items())
    return os.path.join(directory, f'{filename}.joblib')

def save_to_joblib(H, param, directory):
    filename = param_to_filename(directory, param)
    # Save in joblib file
    joblib.dump({ "hypergraph": H, "params": param}, filename)

def run_and_save(model, param_model, directory):
    filename = param_to_filename(directory, param_model)
    if os.path.exists(filename):
        return filename
    H = model(**param_model)
    save_to_joblib(H, param_model, directory)
    return filename

def wrapper(args):
    return run_and_save(*args)


MODEL = 'polyadic_closure'

# if MODEL ==  'preferential_attachement':
#     ps = np.linspace(0,1, 5)
#     k_0 = 1
#     n_0 = 100
#     params = [{'M': 100, 'p_new': p_new, 'k_0': k_0, 'n_0': n_0}
#                 for p_new in ps 
#                 ]
#     directory = './out/simulations/preferential'
#     arg = [(preferential_attachment, param, directory) for param in params ]

# elif MODEL == 'random_overlaps_pref':
#     M = 500 # Number of edges 
#     p_new = 0.7
#     alphas = np.linspace(0,1, 5)
#     betas_neigh = np.linspace(0,1, 3)
#     betas_remaining = np.linspace(0,1, 3)
#     params = [{'M': M, 'alpha': alpha, 'beta_neigh': beta_neigh, 'beta_remaining': beta_remaining, 'p_new': p_new}
#                 for alpha in alphas 
#                 for beta_neigh in betas_neigh 
#                 for beta_remaining in betas_remaining 
#                 ]
#     directory = './out/simulations/pref_overlaps/'
#     arg = [(random_overlaps_pref, param, directory) for param in params ]

if MODEL == 'triadic_closure':
    ns = [ 10_000]   # number of nodes/steps
    ms = np.arange(2,10)
    ps = np.linspace(0.1,0.9,5)
    params = [{'n' : n,
        'm': m, 
        'p': p, 
        'seed':seed  }
                for n in ns
                for p in ps
                for m in ms
                for seed in range(20)
            ]
    directory = './out/simulations/triadic_closure/'
    args = [(triadic_closure, param, directory) for param in params ]

if MODEL == 'polyadic_closure':
    ns = [30_000]   # number of nodes/steps

    pair_cap = 15
    min_events =2
    Ms = list(iter_Ms(pair_cap=pair_cap, min_events=min_events))

    nps = 5
    x = np.linspace(0, 1.3, nps, endpoint = False)
    ps = 1 - 10**(-1.5* x) 
    
    
    multiedges = [True]
    #n0s = np.linspace(20,1000,15, dtype = int)
    n0s = [20]
    H0s = ['random']
    params = [{'n' : n,
        'M': M, 
        'p': p,
        'n0': n0,
        'H0': H0,
        'multiedges': value,
        'seed':seed  }
                for n in ns
                for p in ps
                for M in Ms
                for n0 in n0s
                for H0 in H0s
                for value in multiedges
                for seed in range(50)
            ]

    directory = './out/simulations/polyadic_closure/'
    args = [(polyadic_closure_2, param, directory) for param in params ]

# elif MODEL == 'temporal_triadic_closure':
#     M = 200
#     N = 100
#     etas = np.linspace(0,1,3)
#     ls = np.linspace(0,1,3)
#     xis = np.linspace(0,1,3)
#     params = [{'N': N, 'M': M, 'eta': eta, 'l': l, 'xi': xi }
#                 for eta in etas
#                 for l in ls
#                 for xi in xis 
#                 ]

#     directory = './out/simulations/temporal_triadic_closure/'
#     arg = [(temporal_triadic_closure, param, directory) for param in params ]

# elif MODEL == 'temporal_polyadic_closure':
#     M = 100
#     N = 100
#     mus = np.linspace(100,2000,3).astype(int)
#     prs = np.logspace(-3,0,4)
#     pns = np.logspace(-3,0,4)
#     ds = np.arange(2,5)
#     params = [{'N': N, 'M': M, 'mu': mu, 'p_r': pr, 'p_n': pn, 'd': d }
#                 for mu in mus
#                 for pr in prs
#                 for pn in pns
#                 for d in ds 
#                 ]
#     directory = './out/simulations/temporal_polyadic_closure/'
#     arg = [(temporal_polyadic_closure, param, directory) for param in params ]


if __name__ == "__main__":
    with mp.Pool() as pool:
        for i, result in enumerate(pool.imap_unordered(wrapper, args), 1):
            print(f"[{i}/{len(args)}] Saved: {os.path.basename(result)}")
