"""
Created on July 02, 2018

@author: Alejandro Molina
"""
from spn.algorithms.Inference import log_likelihood, sum_log_likelihood, prod_log_likelihood, max_log_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, Max, get_nodes_by_type, eval_spn_top_down
import numpy as np
import logging

logger = logging.getLogger(__name__)


def merge_input_vals(l):
    return np.concatenate(l)


def mpe_prod(node, parent_result, data=None, lls_per_node=None, rand_gen=None):
    if parent_result is None:
        return None

    parent_result = merge_input_vals(parent_result)

    children_row_ids = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = parent_result

    return children_row_ids


def mpe_sum(node, parent_result, data=None, lls_per_node=None, rand_gen=None):
    if parent_result is None:
        return None

    parent_result = merge_input_vals(parent_result)

    w_children_log_probs = np.zeros((len(parent_result), len(node.weights)))
    eps = 1e-6
    for i, c in enumerate(node.children):
        # print(f"==>> node.weights[i]: {node.weights[i]}, mpe.py line 41")
        w_children_log_probs[:, i] = lls_per_node[parent_result, c.id] + np.log(node.weights[i] + eps)

    max_child_branches = np.argmax(w_children_log_probs, axis=1)

    children_row_ids = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = parent_result[max_child_branches == i]

    return children_row_ids

def mpe_max(node, parent_result, data=None, lls_per_node=None, rand_gen=None):
    if parent_result is None:
        return None

    parent_result = merge_input_vals(parent_result)

    children_row_ids = {}

    children_log_probs = np.zeros((len(parent_result), len(node.children)))
    for i, c in enumerate(node.children):
        children_log_probs[:, i] = lls_per_node[parent_result, c.id]

    max_child_branches = np.argmax(children_log_probs, axis=1)

    assert data is not None, "data must be passed through to max nodes for proper evaluation."
    given_decision = data[parent_result, node.dec_idx]

    children_row_ids = {}

    for i, c in enumerate(node.children):
        max_and_no_selection = np.concatenate(
                (
                    np.isnan(given_decision).reshape(-1,1),
                    (max_child_branches == i).reshape(-1,1)
                ),
                axis=1
            ).all(axis=1).reshape(-1,1)
        max_or_selected = np.concatenate(
                (
                    max_and_no_selection.reshape(-1,1),
                    (given_decision==i).reshape(-1,1)
                ),
                axis=1
            ).any(axis=1).reshape(-1)
        children_row_ids[c] = parent_result[max_or_selected]

    return children_row_ids


def get_mpe_top_down_leaf(node, input_vals, data=None, mode=0):
    if input_vals is None:
        return None

    input_vals = merge_input_vals(input_vals)

    # we need to find the cells where we need to replace nans with mpes
    data_nans = np.isnan(data[input_vals, node.scope])

    n_mpe = np.sum(data_nans)

    if n_mpe == 0:
        return None

    data[input_vals[data_nans], node.scope] = mode


_node_top_down_mpe = {Product: mpe_prod, Sum: mpe_sum, Max: mpe_max}
_node_bottom_up_mpe = {}
_node_bottom_up_mpe_log = {Sum: sum_log_likelihood, Product: prod_log_likelihood, Max: max_log_likelihood}

def get_node_funtions():
    return (_node_top_down_mpe, _node_bottom_up_mpe)

def log_node_bottom_up_mpe(node, *args, **kwargs):
    probs = _node_bottom_up_mpe[type(node)](node, *args, **kwargs)
    with np.errstate(divide="ignore"):
        return np.log(probs)


def add_node_mpe(node_type, bottom_up_lambda, top_down_lambda, bottom_up_lambda_is_log=False):
    _node_top_down_mpe[node_type] = top_down_lambda

    if bottom_up_lambda_is_log:
        _node_bottom_up_mpe_log[node_type] = bottom_up_lambda
    else:
        _node_bottom_up_mpe[node_type] = bottom_up_lambda
        _node_bottom_up_mpe_log[node_type] = log_node_bottom_up_mpe


def mpe(
    node,
    input_data,
    node_top_down_mpe=_node_top_down_mpe,
    node_bottom_up_mpe_log=_node_bottom_up_mpe_log,
    in_place=False,
):
    valid, err = is_valid(node)
    assert valid, err

    assert np.all(
        np.any(np.isnan(input_data), axis=1)
    ), "each row must have at least a nan value where the samples will be substituted"

    if in_place:
        data = input_data
    else:
        data = np.array(input_data)

    nodes = get_nodes_by_type(node)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    # one pass bottom up evaluating the likelihoods
    log_likelihood(node, data, dtype=data.dtype, node_log_likelihood=node_bottom_up_mpe_log, lls_matrix=lls_per_node)

    instance_ids = np.arange(data.shape[0])

    # one pass top down to decide on the max branch until it reaches a leaf, then it fills the nan slot with the mode
    eval_spn_top_down(node, node_top_down_mpe, parent_result=instance_ids, data=data, lls_per_node=lls_per_node)

    return data
