import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from functools import partial
from collections import Counter


from metrics import (avg_degree,
    std_degree,
    modularity,
    nb_clusters,
    local_clustering_coefficient,
    assortativity,
    clustering_metrics,
    skewness,
    median,
    overlaps,
    degree_count
    )


def _run_feature_funcs(H, feature_funcs, only_keep=None, run_funcs=None):
    """
    Run only the functions listed in run_funcs (subset of feature_funcs).
    If only_keep is provided, filter outputs to those columns.
    """
    out = {}
    items = feature_funcs.items() if run_funcs is None else ((n, feature_funcs[n]) for n in run_funcs)
    for name, func in items:
        res = func(H)

        if isinstance(res, Counter):
            if (only_keep is None) or (name in only_keep):
                out[name] = dict(res)   # or out[name] = res (Counter) if you prefer
            continue
        if isinstance(res, dict):
            if only_keep is None:
                out.update(res)
            else:
                for k, v in res.items():
                    if k in only_keep:
                        out[k] = v
        else:
            if (only_keep is None) or (name in only_keep):
                out[name] = res
    return out


def _load_h(directory, filename, multiedges, clean=True):
    path = os.path.join(directory, filename)
    data = joblib.load(path)
    H = data["hypergraph"]
    if clean:
        H.cleanup(relabel=False, connected=True, multiedges=multiedges)
    params = data.get("params", {})
    return H, params


def _discover_schema(directory, feature_funcs, multiedges, clean=True):
    """
    Returns:
      expected_cols: set of all column names produced by feature_funcs
      func_to_keys: dict {func_name -> set_of_columns_it_produces}
    """
    files = [f for f in os.listdir(directory) if f.endswith(".joblib")]
    if not files:
        return set(), {name: set() for name in feature_funcs}

    H, _ = _load_h(directory, files[0], multiedges, clean=clean)

    expected_cols = set()
    func_to_keys = {}

    for name, func in feature_funcs.items():
        try:
            res = func(H)
        except Exception:
            # If discovery fails for this func, assume scalar under its own name
            func_to_keys[name] = {name}
            expected_cols.add(name)
            continue
        
        if isinstance(res, Counter):
            func_to_keys[name] = {name}
            expected_cols.add(name)

        elif isinstance(res, dict):
            keys = set(res.keys())
            func_to_keys[name] = keys
            expected_cols |= keys
        else:
            func_to_keys[name] = {name}
            expected_cols.add(name)

    return expected_cols, func_to_keys


def _process_missing_features(row, directory, feature_funcs, expected_cols, func_to_keys, multiedges, clean=True):
    # columns that are NaN for this row
    need_cols = {col for col in expected_cols if pd.isna(row.get(col))}
    if not need_cols:
        return {}

    # minimal set of functions whose outputs intersect need_cols
    run_funcs = {fname for fname, keys in func_to_keys.items() if keys & need_cols}
    if not run_funcs:
        return {col: np.nan for col in need_cols}

    try:
        H, _ = _load_h(directory, row['filename'], multiedges, clean)
        return _run_feature_funcs(H, feature_funcs, only_keep=need_cols, run_funcs=run_funcs)
    except Exception as e:
        print(f"Error for {row['filename']}: {e}")
        return {col: np.nan for col in need_cols}


def _process_new_file(filename, directory, feature_funcs, multiedges, clean = True):
    try:
        H, params = _load_h(directory, filename, multiedges, clean)
        feats = _run_feature_funcs(H, feature_funcs)
        feats["filename"] = filename
        feats.update(params)
        return feats
    except Exception as e:
        print(f"Error for {filename}: {e}")
        return {feature: np.nan for feature in feature_funcs} | {"filename": filename}
    

def compute_features_parallel(simulation_directory, feature_funcs, metrics_path, multiedges, processes=None, clean = True):
    ''' 
    Compute multiple features from each joblib file, loading each file only once.

    Parameters
    ----------
    simulation_directory : str
        Path to simulations
    feature_funcs : dict
        Dictionary of {feature_name: function} to apply on the hypergraph
    metrics_path : str
        Path to metrics.json file
        Returns
    -------
    df : pandas.DataFrame
        Updated with new columns for each feature and new rows for new simulations
    '''
    # Load or initialize metrics DataFrame
    if os.path.exists(metrics_path):
        df_metrics = pd.read_json(metrics_path)
    else:
        df_metrics = pd.DataFrame()
    


    # Determine files and feature gaps
    all_files = [f for f in os.listdir(simulation_directory) if f.endswith(".joblib") 
    and 'n-30000' in f 
    #and any( e in f  for e in [f'seed-{i}.' for i in range(10)]) 
    ]
    
   
    already_present = set(df_metrics.get("filename", []))
    missing_files = [f for f in all_files if f not in already_present]

    # First: Process entirely new files
    if missing_files:
        
        with mp.Pool(processes=processes or mp.cpu_count()) as pool:
            worker = partial(_process_new_file, directory=simulation_directory, feature_funcs=feature_funcs, multiedges = multiedges)
            new_results = list(tqdm(pool.imap(worker, missing_files), total=len(missing_files)))

        df_new = pd.DataFrame(new_results)
        df_metrics = pd.concat([df_metrics, df_new], ignore_index=True)

    # Second: Process files already present but missing new features

    expected_cols, func_to_keys = _discover_schema(simulation_directory, feature_funcs, multiedges, clean=clean)

    for col in expected_cols:
        if col not in df_metrics.columns:
            df_metrics[col] = np.nan
            

    files_with_gaps = df_metrics[df_metrics["filename"].isin(all_files)]
    # Identify rows (files) with missing features or nans
    needs_update_mask = pd.isna(files_with_gaps).apply(lambda row: any(
	feature not in df_metrics.columns or row.get(feature)
	for feature in expected_cols
	),
	axis = 1
)
    #needs_update_mask = pd.isna(files_with_gaps)
    needs_update = files_with_gaps[needs_update_mask]

    if not needs_update.empty:
        rows = needs_update.to_dict("records")
        with mp.Pool(processes=processes or mp.cpu_count()) as pool:
            worker = partial(_process_missing_features, directory=simulation_directory, feature_funcs=feature_funcs, expected_cols=expected_cols, func_to_keys=func_to_keys,multiedges=multiedges, clean = clean)
            
            updates = list(tqdm(pool.imap(worker, rows), total=len(rows)))

        for update, row in zip(updates,rows):
            for k, v in update.items():
                df_metrics.loc[df_metrics["filename"] == row["filename"], k] = v

    # Save and return
    df_metrics.to_json(metrics_path, index=False)
    return df_metrics


model = 'ho_fof'
# model = 'ho_fof_emp_village'


multiedges = True
simulation_directory = f"./out/simulations/{model}"

metrics_path = f"./out/metrics/{model}_metrics_multiedges_{multiedges}.json"

feature_funcs = {
    # "avg_degree": avg_degree,
    # "std_degree": std_degree,
    "clustering": clustering_metrics,
    #"median": median,
    #"skewness": skewness,
    #"assortativity": assortativity
    #"clutering_coef": local_clustering_coefficient
    "overlaps": overlaps,
    "degree_count": degree_count
}



if __name__ == "__main__":
    compute_features_parallel(simulation_directory, feature_funcs, metrics_path, multiedges)    
