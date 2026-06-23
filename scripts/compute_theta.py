import os
from collections import Counter
from functools import partial
from multiprocessing import get_context
from tqdm import tqdm

import numpy as np
import pandas as pd
import networkx as nx
import xgi 
import joblib

# --- your project helpers (assumed available) ---
# from your_module import load_h, fit_powerlaw_log_ols


def load_h(directory, filename):
    path = os.path.join(directory, filename)
    data = joblib.load(path)
    H = data['hypergraph']
    params = data.get('params', {})
    return H, params

def fit_powerlaw_log_ols(h, Theta):
    x = np.log(h)
    y = np.log(Theta)
    theta, logc = np.polyfit(x, y, 1)
    #R2
    yhat = theta * x + logc
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    c = float(np.exp(logc))
    return c, float(theta), float(r2)

# ---------- per-file worker ----------
def process_one_file(directory: str, filename: str):
    try:
        # build hypergraph
        H, params = load_h(directory, filename)
        p = params['p']
        M = params['M']
        s = list(M.keys())[0] - 1
        seed = params['seed']

        # H-degrees
        nodes = list(H.nodes)
        deg = np.fromiter(H.degree().values(), dtype=int)

        # counts of each hyper-degree
        h_emp, counts_emp = np.unique(deg.astype(int), return_counts=True)
        counts_per_h = dict(zip(h_emp, counts_emp))

        # --- Measure Theta
        G = xgi.to_graph(H)
        g_deg = np.fromiter(dict(G.degree()).values(), dtype=float)

        A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format='csr')

        invdeg = np.zeros_like(g_deg, dtype=float)
        # avoid div by zero; +1 as in your code
        np.divide(1.0, g_deg + 1, out=invdeg)

        theta_i = (invdeg + A @ invdeg) * p  # node-level theta

        # --- Theta vs h (group by hyper-degree h_i)
        df = pd.DataFrame(
            {"h_i": deg.astype(int), "theta_i": theta_i, "k_i": g_deg}
        )
        gb = df.groupby("h_i", sort=True)["theta_i"].mean()

        h = np.asarray(gb.index, dtype=int)
        Theta = np.asarray(gb.values, dtype=float)

        c, theta, r2 = fit_powerlaw_log_ols(h, Theta)
        theory = c * h ** theta

        rows = []
        for h_i, Theta_h, theory_h in zip(h, Theta, theory):
            rows.append(
                {
                    "s": s,
                    "p": p,
                    "seed": seed,
                    "theta": theta,     # exponent from fit
                    "c": c,             # prefactor from fit
                    "h_i": h_i,
                    "degree_counts": counts_per_h.get(int(h_i), 0),
                    "Theta": float(Theta_h),
                    "Theta_theory": float(theory_h),
                    "file": filename,
                    "r2": r2,
                }
            )
        return rows

    except Exception as e:
        # Return a single error row so failures don't kill the whole run
        return [{
            "file": filename,
            "error": repr(e)
        }]

def main():
    metrics_out_csv = "./out/metrics/degree_dist_metrics.csv"
    directory = "./out/simulations/node_addition_ho_fof"

    all_files = [
        f for f in os.listdir(directory)
        if f.endswith(".joblib") and "n-100000" in f and "p-0.97" not in f
    ]

    # Use 'spawn' for safety across platforms (esp. with heavy libs)
    ctx = get_context("spawn")
    results = []

    worker = partial(process_one_file, directory)

    # Tune processes/chunksize if desired
    processes = max(1, (os.cpu_count() or 2) - 1)
    chunksize = max(1, len(all_files) // (processes * 4) or 1)

    with ctx.Pool(processes=processes) as pool:
        for rows in tqdm(pool.imap_unordered(worker, all_files, chunksize=chunksize), total=len(all_files)):
            # rows is a list of dicts (or a single error row)
            results.extend(rows)

    metrics_df = pd.DataFrame(results)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(metrics_out_csv), exist_ok=True)

    metrics_df.to_csv(metrics_out_csv, index=False)
    print(f"Saved {len(metrics_df)} rows to {metrics_out_csv}")

if __name__ == "__main__":
    main()