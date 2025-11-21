import sys
sys.path.append(".")

import numpy as np
import pandas as pd
import torch
from tigramite import data_processing as pp
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr_wls import ParCorr
from tigramite.pcmci import PCMCI

from src.utils.transformation_utils import to_lagged_adj_ready


def tensor_to_pcmci_res_modified(sample: torch.tensor, c_test: str, max_tau: int) -> np.array:
    """
    Converts a time-series sample tensor (time-series dataset of shape `(n_vars, max_tau)`)
    to an appropriate dataframe for PCMCI and then runs the PCMCI algorithm.

    Args:
        sample (torch.tensor): the sample tensor described before
        c_test (str): cond ind test to be used ("ParCorr" or "GPDC").
        max_tau (int): maximum lag
    
    Returns:
        the PCMCI q-matrix 
    """

    if c_test == "ParCorr":
        c_test = ParCorr()
    elif c_test == "GPDC":
        c_test = GPDC()
    else:
        raise Exception("c_test must be either ParCorr or GPDC")

    dataframe = pp.DataFrame(
        sample.detach().numpy().astype(float),
        datatime=np.arange(len(sample)),  # time-axis for PCMCI
        var_names=np.arange(sample.shape[1])  # should be (0,1,..., num_vars-1)
    )
    
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=0)
    results = pcmci.run_pcmci(tau_max=max_tau) # p-values output of shape `(num_vars, num_vars, max_tau + 1)`

    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"], fdr_method="fdr_bh"
    ) # returns the corrected p-values using the Benjamini-Hochberg method, including contemporaneous edges with shape `(num_vars, num_vars, max_tau +1)`

    q_matrix = np.swapaxes(np.flip(q_matrix[:, :, 1:], 2), 0, 1)

    return q_matrix


def run_inv_pcmci(sample: pd.DataFrame, c_test=None, max_tau: int=1, fdr_method: str="fdr_bh", invert: bool=True, rnd: int=3, threshold: float=0.05) -> np.array:
    """
    Converts an fMRI datasample to appropriate dataframe for PCMCI and then runs the PCMCI algorithm.

    Args:
        sample (pd.DataFrame) : the time-series data as a Pandas DataFrame 
        c_test (tigramite.independence_tests) : conditional independence test to be used (ParCorr() or GPDC())
        max_tau (int) : (optional) the maximum lag to use; default is `1`.
        fdr_method (str) : (optional) the FDR method that PCMCI will use internally; for more info, 
                           please refer to the official PCMCI documentation
        invert (bool) : (optional) if true, it inverts the time-slices of the returning adjacency matrix, 
                           in order to match the effect-case order of CP
        rnd (str): (optional) the rounding range for the output 
        threshold (float) : (optional) the threshold on which the corrected p-values of the p-matrix are adjusted; default is `0.05`.
    
    Returns:
        the PCMCI q-matrix (numpy.array) without contemporaneous edges, of shape `(num_vars, num_vars, max_tau)`
    """
    if isinstance(sample, pd.DataFrame):
        sample = sample.values
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().numpy()

    if c_test is None:
        c_test = ParCorr()

    dataframe = pp.DataFrame(
        sample,
        datatime=np.arange(sample.shape[0]),
        var_names=np.arange(sample.shape[1]),
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=1)
    pcmci.verbosity = 0
    results = pcmci.run_pcmci(tau_min=0, tau_max=max_tau, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues( # returns corrected p-values
        p_matrix=results["p_matrix"], fdr_method=fdr_method
    )
    out = q_matrix[:, :, 1:] # exclude contamporaneous edges

    # Select the edges with low p-values, zero-out the edges with high-p-values
    out[out < threshold] = 1
    out[out < 1] = 0

    if invert:
        out = to_lagged_adj_ready(out.round(rnd))
    else:
        out = torch.tensor(out)
    
    return out

def run_inv_pcmciplus(sample: pd.DataFrame, c_test=None, max_tau: int=1, fdr_method: str="fdr_bh", invert: bool=True, rnd: int=3, threshold: float=0.05) -> np.array:
    """
    Converts a datasample to an appropriate dataframe for PCMCI+ and then runs the PCMCI+ algorithm.

    Args:
        sample (pd.DataFrame) : the time-series data as a Pandas DataFrame 
        c_test (tigramite.independence_tests) : conditional independence test to be used (ParCorr() or GPDC())
        max_tau (int) : (optional) the maximum lag to use; defaults to 1
        fdr_method (str) : (optional) the FDR method that PCMCI will use internally; for more info, 
                           please refer to the official PCMCI documentation
        invert (bool) : (optional) if true, it inverts the time-slices of the returning adjacency matrix, 
                           in order to match the effect-case order of CP
        rnd (str): (optional) the rounding range for the output 
        threshold (float) : (optional) the threshold on which the corrected p-values of the p-matrix are adjusted; default is 0.05
    
    Returns:
        the PCMCI+ q-matrix (numpy.array) excluding contemporaneous edges, of shape `(num_vars, num_vars, max_tau)`
    """
    if isinstance(sample, pd.DataFrame):
        sample = sample.values
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().numpy()

    if c_test is None:
        c_test = ParCorr()

    dataframe = pp.DataFrame(
        sample,
        datatime=np.arange(sample.shape[0]),
        var_names=np.arange(sample.shape[1]),
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=1)
    pcmci.verbosity = 0
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=max_tau, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"], fdr_method="fdr_bh", exclude_contemporaneous=True # excluding cont. edges as PCMCI+ accouts for that
    )         
    out = q_matrix[:, :, 1:] # exclude contamporaneous edges

    # Select the edges with low p-values, zero-out the edges with high-p-values
    out[out < threshold] = 1
    out[out < 1] = 0

    if invert:
        out = to_lagged_adj_ready(out.round(rnd))
    else:
        out = torch.tensor(out)
    
    return out