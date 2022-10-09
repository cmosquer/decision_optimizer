import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt

def plot_costs_vs_theta(saving_dir, cost_ratios=np.array([0.01, 0.1, 1, 10, 100]),
                        expected_positive_priors=[0.1, 0.3, 0.5]):

    fig, ax = plt.subplots(2, 1)

    for expected_positive_prior in expected_positive_priors:
        thetas = np.log(cost_ratios*((1-expected_positive_prior)/expected_positive_prior))
        ax[0].plot(cost_ratios, thetas, label="Expected prior: "+str(expected_positive_prior))
        ax[1].plot(np.log(cost_ratios), thetas, label="Expected prior: "+str(expected_positive_prior))
    ax[0].set_xlabel('costFP/costFN')
    ax[1].set_xlabel('Log(costFP/costFN)')
    ax[0].set_ylabel('Optimal LLR threshold (theta)')
    ax[0].legend()
    fig.savefig(saving_dir)


def cross_entropy(pos, neg, pos_prior=0.5, deriv=False):
    baseline = -pos_prior * np.log(pos_prior) - (1 - pos_prior) * np.log(1 - pos_prior)
    logitprior = logit(pos_prior)
    if not deriv:
        pos = np.mean(softplus(-pos - logitprior))
        neg = np.mean(softplus(neg + logitprior))
        return (pos_prior * pos + (1 - pos_prior) * neg) / baseline

    pos, back1 = softplus(- pos - logitprior, deriv=True)
    neg, back2 = softplus(neg + logitprior, deriv=True)
    k1 = pos_prior / (len(pos) * baseline)
    k2 = (1 - pos_prior) / (len(neg) * baseline)
    y = k1 * pos.sum() + k2 * neg.sum()

    def back(dy):
        dtar = back1(-dy * k1)
        dnon = back2(dy * k2)
        return dtar, dnon

    return y, back


def cs_softplus(x):
    """numerically stable and complex-step-friendly version of:

       softplus = log( 1 + exp(x) )
    """
    if not np.iscomplexobj(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    # return np.log( 1 + np.exp(x) )
    rx = np.real(x)
    y = cs_softplus(rx)
    return y + 1.0j * expit(rx) * np.imag(x)


def cs_sigmoid(x):
    """numerically stable and complex-step-friendly version of sigmoid"""
    if not np.iscomplexobj(x): return expit(x)
    rx = np.real(x)
    p, q = expit(rx), expit(-rx)
    return p + 1.0j * p * q * np.imag(x)


def softplus(x, deriv=False):
    y = cs_softplus(x)
    if not deriv: return y

    dydx = cs_sigmoid(x)

    def back(dy): return dy * dydx

    return y, back


def optobjective(f, trans=None, sign=1.0, **kwargs):
    """Wrapper for f to turn it into a minimization objective.

    The objective is:
        obj(x) = sign*f(*trans(x),**kwargs)
    where f can take multiple inputs and has a scalar output.
    The input x is a vector.

    Both f(...,deriv=True) and trans(...) return backpropagation
    handles, but obj(x) immediately returns value, gradient.


    """
    if not trans is None:
        def obj(x):
            *args, back1 = trans(x, deriv=True)
            y, back2 = f(*args, deriv=True, **kwargs)
            g = back1(*back2(sign))
            return sign * y, g
    else:
        def obj(x):
            y, back2 = f(x, deriv=True, **kwargs)
            g = back2(sign)
            return sign * y, g

    return obj


def trans(a, b, *, x, deriv=False):
    if not deriv:
        return a * x + b

    def back(dy):
        da = dy @ x
        db = dy.sum()
        return da, db

    return a * x + b, back


def obj(params, *, pos_scores, neg_scores, positive_prior=0.5, deriv=False):
    a, b = params
    if not deriv:
        return cross_entropy(trans(a, b, x=pos_scores),
                             trans(a, b, x=neg_scores),
                             pos_prior=positive_prior)
    t, back1 = trans(a, b, x=pos_scores, deriv=True)
    n, back2 = trans(a, b, x=neg_scores, deriv=True)
    y, back3 = cross_entropy(t, n, pos_prior=positive_prior, deriv=True)

    def back(dy):
        dt, dn = back3(dy)
        da, db = back2(dn)
        da2, db2 = back1(dt)
        return np.array([da + da2, db + db2])

    return y, back
