import numpy as np
import multiprocessing as mp
import joblib
import os

from utils import iter_Ms

from generator import (preferential_attachment,
                    node_addition_ho_fof,
                    node_addition_ho_fof_resampled_T,
                    edge_addition_ho_fof)

from utils import get_param_from_emp, calibrate_from_emp, rescale_pnew, clean_edges

# Save simulation
def param_to_filename(directory, param_dict):
    def fmt(val):
        return f"{val:.3f}" if isinstance(val, float)  else str(val)
    filename = "_".join(f"{key}-{fmt(val)}" for key, val in param_dict.items() if type(val) is not list)
    return os.path.join(directory, f'{filename}.joblib')

def save_to_joblib(H, param, directory):
    filename = param_to_filename(directory, param)
    # Save in joblib file
    joblib.dump({ "hypergraph": H, "params": param}, filename)

def run_and_save(model, param_model, directory):
    filename = param_to_filename(directory, param_model)
    if os.path.exists(filename):
        return ' '
    # print (f'{filename} doesn t exisit')
    H = model(**param_model)
    save_to_joblib(H, param_model, directory)
    return filename

def wrapper(args):
    return run_and_save(*args)


MODEL = 'ho_fof_resampled_T'
dataset = 'village'

if MODEL ==  'preferential_emp':
    # Load empirical hypegraph
    edges_emp = joblib.load(f'./data/edges_{dataset}.joblib')
    params = []
    nb_itt = 100
    for seed in range(nb_itt):
        np.random.seed(seed)
        np.random.shuffle(edges_emp)
        edges_emp = clean_edges(edges_emp)
        # Get paramaters from emprical hypergraph
        n0 = max(len(edge) for edge in edges_emp)
        
        D, P_new, t0 =  calibrate_from_emp(edges_emp, n0, shuffle = False, seed = seed)

        params.append({'D':D, 'P_new':P_new, 'n0':n0, 'seed': seed})

    directory = f'./out/simulations/{MODEL}_{dataset}'
    args = [(preferential_attachment, param, directory) for param in params ]
    

if MODEL == 'ho_fof_emp':
    # Load empirical hypegraph
    edges_emp = joblib.load(f'./data/edges_{dataset}.joblib')
    params = []
    ps =[0,0.1,0.3,0.5,0.7, 0.9,1]
    nb_itt = 100
    
    for seed in range(nb_itt):
        #np.random.seed(seed)
        np.random.shuffle(edges_emp)
        edges_emp = clean_edges(edges_emp)

        # Get paramaters from emprical hypergraph
        n0 = max(len(edge) for edge in edges_emp)
        
        D, P_new, t0 =  calibrate_from_emp(edges_emp, n0, shuffle = False, seed = seed)

        params.extend([{'p':p, 'D':D, 'P_new':P_new, 'n0':n0, 'seed': seed} for p in ps])


    directory = f'./out/simulations/{MODEL}_{dataset}'
    args = [(edge_addition_ho_fof, param, directory) for param in params ]



if MODEL == 'ho_fof':
    ns = [100_000]   # number of nodes/steps


    # pair_cap = 15
    # min_events =2
    # Ms = list(iter_Ms(pair_cap=pair_cap, min_events=min_events))

    # nps = 5
    # x = np.linspace(0, 1.3, nps, endpoint = False)
    # ps = 1 - 10**(-1.5* x) 
    
    ps =[0,0.1,0.3,0.5,0.7,0.9,]
    
    Ms = [{2:1},{3:1},{4:1}]
    multiedges = [True]
    #n0s = np.linspace(20,1000,15, dtype = int)
    n0s = [20]
    H0s = ['random']
    params = [{'n' : n,
        'M': M, 
        'p': p,
        'n0': n0,
        'multiedges': value,
        'seed':seed  }
                for n in ns
                for p in ps
                for M in Ms
                for n0 in n0s
                for H0 in H0s
                for value in multiedges
                for seed in range(100)
            ]

    directory = f'./out/simulations/{MODEL}'
    args = [(node_addition_ho_fof, param, directory) for param in params ]


if MODEL == 'ho_fof_resampled_T':
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
        'multiedges': value,
        'seed':seed  }
                for n in ns
                for p in ps
                for M in Ms
                for n0 in n0s
                for H0 in H0s
                for value in multiedges
                for seed in range(30)
            ]

    directory = f'./out/simulations/{MODEL}'
    args = [(node_addition_ho_fof_resampled_T, param, directory) for param in params ]


if __name__ == "__main__":
    procs = mp.cpu_count()
    with mp.Pool(processes= procs) as pool:
        for i, result in enumerate(pool.imap_unordered(wrapper, args), 1):
            print(f"[{i}/{len(args)}] Saved: {os.path.basename(result)}")

