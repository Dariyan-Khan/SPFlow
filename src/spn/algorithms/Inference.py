"""
Created on March 21, 2018

@author: Alejandro Molina
"""
import logging
import numpy as np
from scipy.special import logsumexp

from spn.structure.Base import Product, Sum, eval_spn_bottom_up, Max, Out_Latent

from spn.structure.leaves.parametric.Parametric import In_Latent

logger = logging.getLogger(__name__)

EPSILON = np.finfo(float).eps

def add_eps(x, eps=1e-6):
    return np.log(np.exp(x) + np.array(eps))


def leaf_marginalized_likelihood(node, data=None, dtype=np.float64):
    # print(f"==>> node.scope: {node.scope}")
    assert len(node.scope) == 1, node.scope 
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    assert data.shape[1] >= 1
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]
    assert len(observations.shape) == 1, observations.shape
    return probs, marg_ids, observations


def prod_log_likelihood(node, children, data=None, dtype=np.float64):
    eps=np.array(1e-6)
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype
    pll = np.sum(llchildren, axis=1).reshape(-1, 1) #+ np.log(eps)
    pll[np.isinf(pll)] = np.finfo(pll.dtype).min
    return pll

def out_latent_log_likelihood(node, children, data=None, dtype=np.float64):
    eps = np.array(1e-6)
    llchildren = np.concatenate(children, axis=1) ##+ np.log(eps)
    assert llchildren.dtype == dtype
    #print("node and llchildren", (node, llchildren))
    mll = np.max(llchildren, axis=1).reshape(-1, 1)
    return mll

def prod_likelihood(node, children, data=None, dtype=np.float64):
    eps = np.array(1e-6)
    llchildren = np.concatenate(children, axis=1) #+ np.log(eps)
    assert llchildren.dtype == dtype
    return np.prod(llchildren, axis=1).reshape(-1, 1)

def max_log_likelihood(node, children, data=None, dtype=np.float64):
    eps = np.array(1e-6)
    llchildren = add_eps(np.concatenate(children, axis=1)) #+ np.log(eps)

    assert llchildren.dtype == dtype

    if llchildren.shape[1] == 1:    # if only one child, then it is max.
        return llchildren

    assert data is not None, "data must be passed through to max nodes for proper evaluation."
    decision_value_given = data[:, node.dec_idx]
    max_value = np.argmax(llchildren, axis=1)
    d_given = np.full(decision_value_given.shape[0], np.nan)
    mapd = {node.dec_values[i]:i for i in range(len(node.dec_values))}
    for k, v in mapd.items(): d_given[decision_value_given==k] = v
    # if data contains a decision value use that otherwise use max
    child_idx = np.select([np.isnan(d_given), True],
                          [max_value, d_given]).astype(int)

    mll = llchildren[np.arange(llchildren.shape[0]), child_idx].reshape(-1, 1)

    # if decision value given is not in children, assign 0 probability
    missing_dec_branch = np.logical_and(np.logical_not(np.isnan(decision_value_given)),np.isnan(d_given))
    mll[missing_dec_branch] = np.finfo(mll.dtype).min

    return mll

def max_likelihood(node, children, data=None, dtype=np.float64):
    eps = np.array(1e-6)
    llchildren = np.concatenate(children, axis=1) # + np.log(eps)
    assert llchildren.dtype == dtype
    # print("node and llchildren", (node,llchildren))
    assert data is not None, "data must be passed through to max nodes for proper evaluation."
    decision_value_given = data[:,node.dec_idx]
    max_value = np.argmax(llchildren, axis=1)
    # if data contains a decision value use that otherwise use max
    child_idx = np.select([np.isnan(decision_value_given), True],
                          [max_value, decision_value_given]).astype(int)
    return llchildren[np.arange(llchildren.shape[0]), child_idx].reshape(-1, 1)


def sum_log_likelihood(node, children, data=None, dtype=np.float64):
    eps=1e-6
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)
    b += eps

    sll = logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)

    return sll


def sum_likelihood(node, children, data=None, dtype=np.float64):
    eps=1e-6
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)
    b+= eps

    return np.dot(llchildren, b).reshape(-1, 1)

  
_node_log_likelihood = {Sum: sum_log_likelihood, Product: prod_log_likelihood, Max: max_log_likelihood, Out_Latent: out_latent_log_likelihood}
_node_likelihood = {Sum: sum_likelihood, Product: prod_likelihood, Max: max_likelihood}


def log_node_likelihood(node, *args, **kwargs):
    tnode = _node_likelihood[type(node)]
    probs = _node_likelihood[type(node)](node, *args, **kwargs)
    
    with np.errstate(divide="ignore"):
        eps = 1e-6
        if type(node) == In_Latent:
            nll = probs
        else:
            nll = np.log(probs + eps)
        nll[np.isinf(nll)] = np.finfo(nll.dtype).min
        assert not np.any(np.isnan(nll))
        return nll


def add_node_likelihood(node_type, lambda_func, log_lambda_func=None):
    _node_likelihood[node_type] = lambda_func
    if log_lambda_func is None:
        log_lambda_func = log_node_likelihood
    _node_log_likelihood[node_type] = log_lambda_func


def likelihood(node, data, dtype=np.float64, node_likelihood=_node_likelihood, lls_matrix=None, debug=False):
    all_results = {}

    if debug:
        assert len(data.shape) == 2, "data must be 2D, found: {}".format(data.shape)
        original_node_likelihood = node_likelihood

        def exec_funct(node, *args, **kwargs):
            assert node is not None, "node is nan "
            funct = original_node_likelihood[type(node)]
            ll = funct(node, *args, **kwargs)
            assert ll.shape == (data.shape[0], 1), "node %s result has to match dimensions (N,1)" % node.id
            assert not np.any(np.isnan(ll)), "ll is nan %s " % node.id
            return ll

        node_likelihood = {k: exec_funct for k in node_likelihood.keys()}
        # print(f"==>> node_likelihood: {node_likelihood}")



    result = eval_spn_bottom_up(node, node_likelihood, all_results=all_results, debug=debug, dtype=dtype, data=data)
    # print(f"==>> result: {result}")

    if lls_matrix is not None:
        for n, ll in all_results.items():
            lls_matrix[:, n.id] = ll[:, 0]

    return result


def log_likelihood(
    node, data, dtype=np.float64, node_log_likelihood=_node_log_likelihood, lls_matrix=None, debug=False
):
    return likelihood(node, data, dtype=dtype, node_likelihood=node_log_likelihood, lls_matrix=lls_matrix, debug=debug)


def conditional_log_likelihood(node_joint, node_marginal, data, log_space=True, dtype=np.float64):
    result = log_likelihood(node_joint, data, dtype) - log_likelihood(node_marginal, data, dtype)
    if log_space:
        return result

    return np.exp(result)
