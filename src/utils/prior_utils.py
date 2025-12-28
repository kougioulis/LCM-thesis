import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def find_paths_iterative(Y: torch.Tensor, max_length: int = 2) -> list[dict]:
    """
    Enumerates all directed causal paths of length ≤ max_length in a lagged adjacency tensor Y.
    Each path is valid only if all constituent edges are active (Y[j,i,k] > 0).

    Args:
        Y: lagged adjacency tensor of shape (n_vars, n_vars, max_lag)
        max_length: maximum path length

    Returns:
        paths: a list of dictionaries with path info:
        {"nodes": [...], "lags": [...], "last_edge": (i, j, lag)}
    """
    n_vars, _, max_lag = Y.shape
    if max_length is None:
        max_length = random.randint(1, max_lag)
    edges = {}  # edges[i] = [(j, lag)] where i → j with lag 'lag'

    for j in range(n_vars):
        for i in range(n_vars):
            for k in range(max_lag):
                if Y[j, i, k] > 0:
                    lag = max_lag - k
                    edges.setdefault(i, []).append((j, lag))

    paths = []
    for start_node in range(n_vars):
        stack = [([start_node], [], 0, 1)] # stack elements: (nodes, lags, depth, strength)

        while stack:
            nodes, lags, depth, strength = stack.pop()
            last_node = nodes[-1]
            if last_node not in edges:
                continue

            for next_node, lag in edges[last_node]:
                if next_node in nodes:
                    continue  # avoiding cycles

                edge_active = int(Y[next_node, last_node, max_lag - lag] > 0) # "Logical multiplication": 1 if edge exists, else 0

                new_strength = strength * edge_active

                if new_strength > 0:  # only continue if all edges so far are active
                    new_nodes = nodes + [next_node]
                    new_lags = lags + [lag]
                    if len(new_nodes) >= 2:
                        paths.append({
                            "nodes": new_nodes,
                            "lags": new_lags,
                            "last_edge": (nodes[-1], next_node, lag)
                        })
                    if depth + 1 < max_length:
                        stack.append((new_nodes, new_lags, depth + 1, new_strength))

    return paths


def get_random_prior_knowledge(
    Y: torch.Tensor,
    max_length: int = 1,
    pct_true: float = 0.1,
    pct_false: float = 0.1,
    pct_exclude: float = 0.1,
    pct_true_belief_range: tuple[float, float] = (0.7, 1.0),
    pct_false_belief_range: tuple[float, float] = (0.1, 0.4),
    pct_excl_belief_range: tuple[float, float] = (0.7, 1.0),
    max_allowed: int = None,
    find_paths_iterative=find_paths_iterative
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Sample priors (1=must exist, 2=must not exist, 0=no information) 
    over both direct edges (lag<=1) and multi-step paths (lag<=max_length).

    Returns:
        prior  : LongTensor (V, V, L) with {0,1,2}
        belief : FloatTensor (V, V, L) with confidences in [0,1]
    """
    V, _, L = Y.shape
    prior = torch.zeros_like(Y, dtype=torch.long)
    belief = torch.zeros_like(Y, dtype=torch.float)

    # === direct edges only ===
    if max_length == 1:
        true_edges = (Y > 0).nonzero(as_tuple=False)
        false_edges = (Y <= 0).nonzero(as_tuple=False)

        # how many to pick
        n_true = max(1, int(pct_true * len(true_edges)))
        n_false = max(1, int(pct_false * len(false_edges)))
        n_excl = max(1, int(pct_exclude * len(false_edges)))
        if max_allowed:
            n_true = min(n_true, max_allowed)
            n_false = min(n_false, max_allowed)
            n_excl = min(n_excl, max_allowed)

        # pick them
        sel_true = true_edges[torch.randperm(len(true_edges))[:n_true]]
        sel_false = false_edges[torch.randperm(len(false_edges))[:n_false+n_excl]]

        # assign true priors
        for (j,i,k) in sel_true:
            prior[j,i,k] = 1
            belief[j,i,k] = random.uniform(*pct_true_belief_range)

        # split into false priors vs exclusions
        sel_fp, sel_ex = sel_false[:n_false], sel_false[n_false:n_false+n_excl]
        for (j,i,k) in sel_fp:
            if prior[j,i,k]==0:
                prior[j,i,k] = 1
                belief[j,i,k] = random.uniform(*pct_false_belief_range)
        for (j,i,k) in sel_ex:
            if prior[j,i,k]==0:
                prior[j,i,k] = 2
                belief[j,i,k] = random.uniform(*pct_excl_belief_range)

        return prior, belief

    # === multi-step paths ===
    assert find_paths_iterative is not None, "find_paths_iterative must be provided for multi-step mode"
    all_paths = find_paths_iterative(Y, max_length)
    all_paths = [p for p in all_paths if p["nodes"][0] != p["nodes"][-1]]  # remove self-loops
    if len(all_paths) == 0:
        return prior, belief

    # how many to pick
    n_true_paths = max(1, int(pct_true * len(all_paths)))
    n_false_paths = max(1, int(pct_false * len(all_paths)))
    n_excl_paths = max(1, int(pct_exclude * len(all_paths)))
    if max_allowed:
        n_true_paths = min(n_true_paths, max_allowed)
        n_false_paths = min(n_false_paths, max_allowed)
        n_excl_paths = min(n_excl_paths, max_allowed)

    # pick true paths
    sel_true_paths = random.sample(all_paths, n_true_paths)

    # generate fake paths (not in all_paths)
    all_possible = [(j,i,k) for j in range(V) for i in range(V) for k in range(L) if j!=i]
    true_edges = {(p["last_edge"][1], p["last_edge"][0], L - p["last_edge"][2]) for p in all_paths}
    fake_edges = [e for e in all_possible if e not in true_edges]

    #sel_fake = random.sample(fake_edges, n_false_paths + n_excl_paths)
    #sel_fp, sel_ex = sel_fake[:n_false_paths], sel_fake[n_false_paths:]

    # number of fake edges to sample
    total_needed = n_false_paths + n_excl_paths
    n_to_sample = min(len(fake_edges), total_needed)

    if n_to_sample > 0:
        sel_fake = random.sample(fake_edges, n_to_sample)
        # split proportionally between false and exclude
        if total_needed > 0:
            n_false_adj = round(n_to_sample * n_false_paths / total_needed)
        else:
            n_false_adj = 0
        n_excl_adj = n_to_sample - n_false_adj

        sel_fp, sel_ex = sel_fake[:n_false_adj], sel_fake[n_false_adj:]
    else:
        sel_fp, sel_ex = [], []

    # assign true path priors
    for p in sel_true_paths:
        i,j,lag = p["last_edge"]  # i→j at lag
        k = L - lag
        prior[j,i,k] = 1
        belief[j,i,k] = random.uniform(*pct_true_belief_range)

    # assign false priors (hallucinated paths)
    for (j,i,k) in sel_fp:
        if prior[j,i,k] == 0:
            prior[j,i,k] = 1
            belief[j,i,k] = random.uniform(*pct_false_belief_range)

    # assign exclusions
    for (j,i,k) in sel_ex:
        if prior[j,i,k] == 0:
            prior[j,i,k] = 2
            belief[j,i,k] = random.uniform(*pct_excl_belief_range)

    return prior, belief


def prior_knowledge_loss(predictions: torch.Tensor, prior: torch.Tensor, belief_tensor: torch.Tensor, data: torch.Tensor=None, lab_class: torch.Tensor=None):
    """
    Computes prior knowlege loss using BCEWithLogits, for numerical stability.
    Overall, gently guides low-confidence priors, ignores unknown edges (`prior=0`) and strongly
    penalaizes contradictions to high-confidence priors.
    NOTE: Assumes predictions are transformed logits in `[0,1]`. E.g. passed predictions tensor is `torch.sigmoid(predictions)`.

    Args:
        predictions: Raw prediction logits of shape `(n_vars, n_vars, max_lags)`
        prior: Prior tensor (`0`=unknown, `1`=exists, `2`=not exists)
        belief_tensor: Confidence weights `[0,1]`
        data: Optional data tensor (unused in this version)
        lab_class: Optional class labels (unused)

    Returns:
        loss (torch.tensor): The prior knowledge loss
    """
    prior = prior.to(predictions.device)
    belief_tensor = belief_tensor.to(predictions.device)

    # Convert logits → probs for soft target construction
    preds = torch.sigmoid(predictions)

    # Positive priors (should be 1)
    pos_mask = (prior == 1)
    # Negative priors (should be 2)
    neg_mask = (prior == 2) 

    # soft targets
    soft_targets = preds.detach().clone()

    # For positive priors → target between preds and 1
    soft_targets[pos_mask] = (
        belief_tensor[pos_mask] * 1.0 +
        (1.0 - belief_tensor[pos_mask]) * preds.detach()[pos_mask]
    )

    # For negative priors → target between preds and 0
    soft_targets[neg_mask] = (
        belief_tensor[neg_mask] * 0.0 +
        (1.0 - belief_tensor[neg_mask]) * preds.detach()[neg_mask]
    )

    # Compute BCE against these soft targets
    loss = F.binary_cross_entropy(
        preds,          # model probabilities (sigmoid(logits))
        soft_targets,   # interpolated targets
        reduction="none"
    )

    # Only apply to positions with priors
    mask = (pos_mask | neg_mask).float()
    n_priors = (pos_mask | neg_mask).sum() # bitwise OR

    #print(f'Number of priors: {n_priors}')

    # Only average over positions with active priors
    if n_priors == 0:
        return torch.tensor(0.0, device=prior.device)

    return (loss * mask).sum() / n_priors



# We extend get_random_prior_knowledge for needs of the inference experiments
def get_random_prior_knowledge_for_inference(
    Y: torch.Tensor,
    max_length: int = 1,
    pct_true: float = 0.1,
    pct_false: float = 0.1,
    pct_exclude: float = 0.1,
    #belief_range: tuple[float, float] = (0.1, 0.9),
    pct_true_belief_range: tuple[float, float] = (0.1, 0.9),
    pct_false_belief_range: tuple[float, float] = (0.1, 0.9),
    pct_excl_belief_range: tuple[float, float] = (0.1, 0.9),
    max_allowed = None
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Sample priors (1=must exist, 2=must not exist, 0=no information) over both direct edges (lag<=1)
    and multi-step paths (lag<=max_length), balanced and reproducible.
    """
    V, _, L = Y.shape
    prior = torch.zeros_like(Y, dtype=torch.long)
    belief = torch.zeros_like(Y, dtype=torch.float)

    # === direct edges only ===
    if max_length == 1:
        true_edges = (Y > 0).nonzero(as_tuple=False)
        false_edges = (Y <= 0).nonzero(as_tuple=False)

        # how many to pick
        n_true = max(1, int(pct_true * len(true_edges)))
        n_false = max(1, int(pct_false * len(false_edges)))
        n_excl = max(1, int(pct_exclude * len(false_edges)))
        if max_allowed:
            n_true = min(n_true, max_allowed)
            n_false = min(n_false, max_allowed)
            n_excl = min(n_excl, max_allowed)

        # pick them
        sel_true = true_edges[torch.randperm(len(true_edges))[:n_true]]
        sel_false = false_edges[torch.randperm(len(false_edges))[:n_false+n_excl]]

        # assign true priors
        for (j,i,k) in sel_true:
            prior[j,i,k] = 1
            belief[j,i,k] = random.uniform(*pct_true_belief_range)

        # split into false priors vs exclusions
        sel_fp, sel_ex = sel_false[:n_false], sel_false[n_false:n_false+n_excl]
        for (j,i,k) in sel_fp:
            if prior[j,i,k]==0:
                prior[j,i,k] = 1
                belief[j,i,k] = random.uniform(*pct_false_belief_range)
        for (j,i,k) in sel_ex:
            if prior[j,i,k]==0:
                prior[j,i,k] = 2
                belief[j,i,k] = random.uniform(*pct_excl_belief_range)

        return prior, belief

    # === multi-step paths ===
    assert find_paths_iterative is not None, "find_paths_iterative must be provided for multi-step mode"
    all_paths = find_paths_iterative(Y, max_length)
    all_paths = [p for p in all_paths if p["nodes"][0] != p["nodes"][-1]]  # remove self-loops
    if len(all_paths) == 0:
        return prior, belief

    # how many to pick
    n_true_paths = max(1, int(pct_true * len(all_paths)))
    n_false_paths = max(1, int(pct_false * len(all_paths)))
    n_excl_paths = max(1, int(pct_exclude * len(all_paths)))
    if max_allowed:
        n_true_paths = min(n_true_paths, max_allowed)
        n_false_paths = min(n_false_paths, max_allowed)
        n_excl_paths = min(n_excl_paths, max_allowed)

    # pick true paths
    sel_true_paths = random.sample(all_paths, n_true_paths)

    # generate fake paths (not in all_paths)
    all_possible = [(j,i,k) for j in range(V) for i in range(V) for k in range(L) if j!=i]
    true_edges = {(p["last_edge"][1], p["last_edge"][0], L - p["last_edge"][2]) for p in all_paths}
    fake_edges = [e for e in all_possible if e not in true_edges]

    #sel_fake = random.sample(fake_edges, n_false_paths + n_excl_paths)
    #sel_fp, sel_ex = sel_fake[:n_false_paths], sel_fake[n_false_paths:]

    # number of fake edges to sample
    total_needed = n_false_paths + n_excl_paths
    n_to_sample = min(len(fake_edges), total_needed)

    if n_to_sample > 0:
        sel_fake = random.sample(fake_edges, n_to_sample)
        # split proportionally between false and exclude
        if total_needed > 0:
            n_false_adj = round(n_to_sample * n_false_paths / total_needed)
        else:
            n_false_adj = 0
        n_excl_adj = n_to_sample - n_false_adj

        sel_fp, sel_ex = sel_fake[:n_false_adj], sel_fake[n_false_adj:]
    else:
        sel_fp, sel_ex = [], []

    # assign true path priors
    for p in sel_true_paths:
        i,j,lag = p["last_edge"]  # i→j at lag
        k = L - lag
        prior[j,i,k] = 1
        belief[j,i,k] = random.uniform(*pct_true_belief_range)

    # assign false priors (hallucinated paths)
    for (j,i,k) in sel_fp:
        if prior[j,i,k] == 0:
            prior[j,i,k] = 1
            belief[j,i,k] = random.uniform(*pct_false_belief_range)

    # assign exclusions
    for (j,i,k) in sel_ex:
        if prior[j,i,k] == 0:
            prior[j,i,k] = 2
            belief[j,i,k] = random.uniform(*pct_excl_belief_range)

    return prior, belief