import numpy as np
from collections import Counter 
import random
from tqdm.auto import tqdm

def vertex_labeled_MH(
    edges,
    multiedges=True,
    n_steps=10000,
    burnin_steps=2000,
    n_clash=0,
    seed=None,
    sim=None,
    message = True,         
    **kwargs
):
    if seed is not None:
        np.random.seed(seed)

    # build pbar description string
    desc = f"multi={multiedges} │ sim={sim}"
    pbar = tqdm(
        total=n_steps + burnin_steps,
        desc=desc,
        dynamic_ncols=True,
        bar_format="{desc} |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    rand = np.random.rand
    randint = np.random.randint

    edge_multiset = [tuple(sorted(e)) for e in edges]

    return _metropolis_rewire(
        edges=edge_multiset,
        n_steps=n_steps,
        rand=rand,
        randint=randint,
        burnin_steps=burnin_steps,
        pbar=pbar,
        multiedges=multiedges,
        n_clash=n_clash,
        rewire_fun=lambda e1, e2: pairwise_reshuffle(e1, e2, preserve_dimensions=True),
        intersect_fun=lambda e1, e2: 2 ** (-len(set(e1) & set(e2))),
        seed=seed,
        message=message
    )



def _metropolis_rewire(
    edges, n_steps, rand, randint, multiedges, n_clash,
    rewire_fun, intersect_fun, burnin_steps, pbar = None, seed=None, message=True
):
    """
    Metropolis-Hastings rewiring algorithm that preserves vertex labels and edge sizes.

    Parameters
    ----------

    Returns
    -------

    """

    num_samples = 0
    num_rejected = 0
    epoch_count = 0
    num_burnin = 0
    done = False

    edge_counter = Counter(edges)
    total_edges = sum(edge_counter.values())
    a = 0
    while not done:
        epoch_count += 1
        edge_list = list(edge_counter.elements())
        additions, removals = [], []
        num_clashes = 0

        # Pre-generate random samples
        buffer_size = 20000
        rand_indices = randint(0, total_edges, buffer_size)
        rand_floats = rand(buffer_size)
        ptr = 0

        while True:
            # Refill buffer if half-used
            if ptr >= buffer_size // 2:
                rand_indices = randint(0, total_edges, buffer_size)
                rand_floats = rand(buffer_size)
                ptr = 0
            
            # Pick two distinct edges
            i, j = rand_indices[ptr], rand_indices[ptr + 1]
            ptr += 2
            e1, e2 = edge_list[i], edge_list[j]
            while e1 == e2:
                i, j = rand_indices[ptr], rand_indices[ptr + 1]
                ptr += 2
                e1, e2 = edge_list[i], edge_list[j]

        
            use_acceptance = num_burnin >= burnin_steps

            if use_acceptance:
                # Metropolis-Hastings acceptance criterion
                intersection_penalty = intersect_fun(e1, e2)
                acceptance_prob = intersection_penalty / (edge_counter[e1] * edge_counter[e2])
                # Reject proposal
                if rand_floats[ptr] > acceptance_prob:
                    num_rejected += 1
                    num_samples += 1
                    continue

           
            # Generate proposal for rewiring
            g1, g2 = rewire_fun(e1, e2)

            if not multiedges:
                if (edge_counter[g1] > removals.count(g1)) or (edge_counter[g2] > removals.count(g2)):
                    num_rejected += int(use_acceptance)
                    num_samples += int(use_acceptance)
                    continue

            num_clashes += removals.count(e1) + removals.count(e2)
            if n_clash > 0 and num_clashes >= n_clash:
                break
            
            # Add proposal
            removals.extend([e1, e2])
            additions.extend([g1, g2])
            num_burnin += int(not(use_acceptance))
            num_samples += int(use_acceptance)

            if pbar is not None:
                pbar.update(1)

            if n_clash == 0:
                break

        delta = Counter(additions)
        delta.subtract(Counter(removals))
        edge_counter.update(delta)
        

            
        if num_samples - num_rejected >= n_steps:
            done = True

    if message:
        print(f"{epoch_count} epochs completed, "
              f"{num_samples - num_rejected} steps taken, "
              f"{num_rejected} steps rejected.")
    
    if pbar is not None:
        pbar.close()
    
    if not multiedges:
        assert max(edge_counter.values())<2
    
    return [tuple(sorted(e)) for e in edge_counter.elements()]

def pairwise_reshuffle(f1, f2, preserve_dimensions = True):
    '''
    Randomly reshuffle the nodes of two edges while preserving their sizes.
    '''
    
    f = list(f1) + list(f2)
    s = set(f)
    
    intersection = set(f1).intersection(set(f2))
    ix = list(intersection)
    
    g1 = ix.copy()
    g2 = ix.copy()
    
    for v in ix:
        f.remove(v)
        f.remove(v)
    
    for v in f:
        if (len(g1) < len(f1)) & (len(g2) < len(f2)):
            if np.random.rand() < .5:
                g1.append(v)
            else:
                g2.append(v)
        elif len(g1) < len(f1):
            g1.append(v)
        elif len(g2) < len(f2):
            g2.append(v)
            
    assert len(g1) == len(f1)

    return (tuple(sorted(g1)), tuple(sorted(g2)))

