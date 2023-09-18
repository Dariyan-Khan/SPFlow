import numpy as np
from spn.structure.Base import Product, Leaf
from spn.structure.leaves.parametric.Parametric import Gaussian, Multivariate_Gaussian, Bernoulli, Categorical, In_Latent


def update_mean_and_covariance(node, parent_result, params, data, lls_per_node=None):
    scope = node.scope.copy()
    tot_vars = data.shape[1]
    for scpe in node.scope:
        if scpe >= tot_vars:
            scope.remove(scpe)
    
    # print(f"data: {data}")
    # print(f"==>> np.ix_(parent_result, scope): {np.ix_(parent_result, scope)}")
    x = data[np.ix_(parent_result, scope)]
    # assert False

    m = x.shape[0]
    n = node.count

    print(f"type of node: {type(node)}")
    print(f"isinstance(node, Leaf): {isinstance(node, Leaf)}")

    assert not np.isnan(x).any(), "data cointains NaN values"
    if isinstance(node, Product):

        if node.cov is None:
            # print(node.scope)
            node.cov = np.identity(len(scope))
        if node.mean is None:
            node.mean = np.zeros(len(scope))

        mean = node.mean
        cov = node.cov

        # update mean
        curr_sample_sum = x.sum(axis=0)
        new_mean = ((n) * (mean) + curr_sample_sum) / (n + m)

        # update covariance
        dx = x - mean
        dm = new_mean - mean

        new_cov = (n * cov + dx.T.dot(dx)) / (n + m) - np.outer(dm, dm)

        # update node values
        node.mean = new_mean
        node.cov = new_cov

    # for gaussian leaves
    if isinstance(node, Leaf):

        if type(node) in [Gaussian, Multivariate_Gaussian]:

            if type(node) == Gaussian:
                if node.stdev is None:
                    # print("in leaf update", node.scope)
                    node.stdev = 1
                if node.mean is None:
                    node.mean = 0

                mean = node.mean
                stdev = node.stdev
                cov = np.array(np.square(stdev))

            elif type(node) == Multivariate_Gaussian:
                if node.cov is None:
                    # print("in leaf update", node.scope)
                    node.cov = np.identity(len(scope))
                if node.mean is None:
                    node.mean = np.zeros(len(scope))

                mean = node.mean
                cov = node.cov
            

            curr_sample_sum = x.sum(axis=0)
            new_mean = ((n) * (mean) + curr_sample_sum) / (n + m)

            # update covariance
            dx = x - mean
            dm = new_mean - mean
            new_cov = (n * cov + dx.T.dot(dx)) / (n + m) - np.outer(dm, dm)

            assert not np.isnan(new_mean).any(), "new mean is NaN at %s" % (node)
            assert not np.isnan(new_cov).any(), "new covariance is NaN at %s" % (node)
            # print("cov",new_cov) 
            # print("mean", new_mean)
            # update node values
            if type(node) == Gaussian:
                new_stdev = np.sqrt(np.abs(new_cov))
                node.mean = new_mean[0]
                node.stdev = new_stdev[0][0]
            else:
                node.mean = new_mean
                node.cov = new_cov
        
        else:
            if type(node) == Bernoulli:
                curr_sample_sum = x.sum(axis=0)
                p = node.p
                new_p = ((p*n) + curr_sample_sum) / (n + m)

                node.p = new_p
            
            elif type(node) == Categorical:

                # print(f"Categorical p: {node.p}")
                # print(f"node.name: {node.name}")
                print(f"x: {x}")
                #print(f"x.shape: {x.shape}")
                x_flat = x.flatten().tolist()
                value_counts = {int(value): x_flat.count(value) for value in set(x_flat)}
                val_min = np.nanmin(list(value_counts.keys()))
                val_max = np.nanmax(list(value_counts.keys()))
                probs = node.p

                all_val_max = max(val_max, len(probs)-1)
                new_probs = [value_counts[i] / (m+n) if i in value_counts.keys() else 0 for i in range(all_val_max+1)]
                
                
                for i, p in enumerate(probs):
                    new_probs[i] = (new_probs[i] * (m+n) + (p * n)) / (m+n)
                
                node.p = new_probs


            
            else:
                raise NotImplementedError("This node type hasn't been implemented")

        return


def iterate_corrs(node, corrthresh):
    v = np.diag(node.cov).copy()
    v[v < 1e-4] = 1e-4
    corrs = np.abs(node.cov) / np.sqrt(np.outer(v, v))
    rows, cols = np.unravel_index(np.argsort(corrs.flatten()), corrs.shape)

    for i, j in zip(reversed(rows), reversed(cols)):
        if corrs[i, j] < corrthresh:
            break
        yield i, j


def update_curr_mean_and_covariance(node, parent_result, params, data, lls_per_node=None):
    scope = node.scope.copy()
    tot_vars = data.shape[1]
    for scpe in node.scope:
        if scpe >= tot_vars:
            scope.remove(scpe)

    x = data[np.ix_(parent_result, scope)]
    m = x.shape[0]
    n = node.count

    if isinstance(node, Product):

        if node.cov is None:
            # print(node.scope)
            node.curr_cov = np.identity(len(scope))
        if node.mean is None:
            node.curr_mean = np.zeros(len(scope))

        # update mean
        mean = np.mean(x, axis=0)
        if m == 1:
            # x1 = np.repeat(x, 2, axis =0)
            cov = np.identity(len(scope))
        else:
            cov = np.cov(x, rowvar=False)

        node.curr_mean = mean
        node.curr_cov = cov
