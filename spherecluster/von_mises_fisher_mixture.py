import numpy as np
from scipy.special import jv
#import scipy.sparse as sp

#from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
#from sklearn.cluster.k_means_ import _init_centroids, _labels_inertia
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils import (
    #check_array,
    check_random_state,
    #as_float_array,
)
from sklearn.utils.extmath import squared_norm



def _vmf_distribution(X, kappa, mu):
    n_examples, n_features = X.shape
    return _vmf_normalize(kappa, n_features) * np.exp(kappa * np.dot(mu, X))


def _vmf_normalize(kappa, dim):
    num = np.pow(kappa, dim/2. - 1.)
    denom = np.pow(2. * np.pi, dim/2.) * jv(kappa, dim/2. - 1.)
    if denom == 0:
        print 'TODO: raise error here'
    return num/denom


def _update_params(X, posterior):
    n_examples, n_features = X.shape
    n_clusters, n_examples = posterior.shape
    weights = np.zeros((n_clusters,))
    centers = np.zeros((n_clusters, n_examples))
    concentrations = np.zeros((n_clusters,))
    for cc in range(n_clusters):
        # update weights (alpha)
        weights[cc] = np.sum(posterior[cc, :]) / n_examples

        # update centers (mu)
        for ee in range(n_examples):
            centers[cc, ee] = X[ee, :] * posterior[cc, ee]

        # precomputes
        center_norm = np.linalg.norm(centers[cc, :])
        rbar = center_norm / (n_examples * weights[cc])

        # normalize centers
        centers[cc, :] = centers[cc, :] / center_norm

        # update concentration (kappa)
        concentrations[cc] = rbar * n_features - np.pow(rbar, 3)
        concentrations[cc] /= 1 - np.pow(rbar, 2)

    return centers, weights, concentrations


def _soft_moVMF(X, n_clusters, max_iter=300, verbose=False,
               init='k-means++', random_state=None, tol=1e-4):
    random_state = check_random_state(random_state)

    # init centers (mus)
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=np.ones(X.shape[1]))

    # init weights (alphas)
    weights = np.ones((n_clusters,))
    weights = weights / np.sum(weights)

    # init concentrations (kappas)
    concentrations = np.ones((n_clusters,))
    concentrations = concentrations / np.sum(concentrations)

    n_examples, n_features = np.shape(X)

    if verbose:
        print("Initialization complete")

    for iter in range(max_iter):
        centers_prev = centers.copy()

        # (expectation)

        # estimate posterior

        f = np.zeros((n_clusters, n_examples))
        for cc in range(n_clusters):
            f[cc, :] = _vmf_distribution(X, concentrations[cc], centers[cc, :])

        posterior = np.zeros((n_clusters, n_examples))
        for cc in range(n_clusters):
            posterior[cc, :] = weights[cc] * f[cc, :]

        for ee in range(n_examples):
            posterior[:, ee] /= np.sum(weights * f[:, ee])

        # (maximization)
        centers, weights, concentrations = _update_params(X, posterior)

        # check convergence
        tolcheck = squared_norm(centers_prev - centers)
        if tolcheck <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (iter, tolcheck, tol))
            break



#class VonMisesFisher(BaseEstimator, ClusterMixin, TransformerMixin):
#    def __init__(self):

