import warnings

import numpy as np
from scipy.special import iv # modified bessel function of first kind
from numpy import i0 # Modified Bessel function of the first kind, order 0, I_0

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster.k_means_ import (
    _init_centroids,
    _tolerance,
    _validate_center_shape,
)
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import (
    check_array,
    check_random_state,
    as_float_array,
)
from sklearn.utils.extmath import squared_norm
from sklearn.metrics.pairwise import cosine_distances
from sklearn.externals.joblib import Parallel, delayed


def _inertia_from_labels(X, centers, labels):
    """Compute interia with cosine distance using known labels.
    """
    n_examples, n_features = X.shape
    intertia = np.zeros((n_examples, ))
    for ee in range(n_examples):
        intertia[ee] = np.dot(X[ee, :], centers[int(labels[ee]), :].T)

    return 1 - np.sum(intertia)

def _labels_inertia(X, centers):
    """Compute labels and interia with cosine distance.
    """
    n_examples, n_features = X.shape
    n_clusters, n_features = centers.shape

    labels = np.zeros((n_examples, ))
    intertia = np.zeros((n_examples, ))

    for ee in range(n_examples):
        dists = np.zeros((n_clusters, ))
        for cc in range(n_clusters):
            dists[cc] = np.dot(X[ee, :], centers[cc, :].T)

        labels[ee] = np.argmin(dists)
        intertia[ee] = dists(labels[ee])

    return labels, 1 - np.sum(intertia)

def _vmf(X, kappa, mu):
    n_examples, n_features = X.shape
    return _vmf_normalize(kappa, n_features) * np.exp(kappa * np.dot(mu, X.T))


def _vmf_normalize(kappa, dim):
    num = np.power(kappa, dim/2. - 1.)

    denom = np.power(2. * np.pi, dim/2.)
    if dim/2. - 1. < 1e-15:
        denom *= i0(kappa)
    else:
        denom *= iv(kappa, dim/2. - 1.)

    if denom == 0:
        raise ValueError("VMF scaling denominator was 0.")

    return num/denom


def _update_params(X, posterior):
    n_examples, n_features = X.shape
    n_clusters, n_examples = posterior.shape
    weights = np.zeros((n_clusters,))
    centers = np.zeros((n_clusters, n_features))
    concentrations = np.zeros((n_clusters,))
    for cc in range(n_clusters):
        # update weights (alpha)
        weights[cc] = np.mean(posterior[cc, :])

        # update centers (mu)
        for ee in range(n_examples):
            centers[cc, :] += 1. * X[ee, :] * posterior[cc, ee]

        # precomputes
        center_norm = np.linalg.norm(centers[cc, :])
        rbar = center_norm / (n_examples * weights[cc])

        # normalize centers
        centers[cc, :] = 1. * centers[cc, :] / center_norm

        # update concentration (kappa)
        concentrations[cc] = rbar * n_features - np.power(rbar, 3.)
        concentrations[cc] /= 1. - np.power(rbar, 2.)

    return centers, weights, concentrations


def _moVMF(X, n_clusters, posterior_type='soft', max_iter=300, verbose=False,
               init='k-means++', random_state=None, tol=1e-6):
    """Mixture of von Mises Fisher clustering.

    Implements the algorithms (i) and (ii) from
    "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"
    """
    random_state = check_random_state(random_state)

    # init centers (mus)
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=np.ones(X.shape[0]))

    # init weights (alphas)
    weights = np.ones((n_clusters,))
    weights = weights / np.sum(weights)

    # init concentrations (kappas)
    concentrations = np.ones((n_clusters,))

    n_examples, n_features = np.shape(X)

    if verbose:
        print("Initialization complete")

    for iter in range(max_iter):
        centers_prev = centers.copy()

        # (expectation)

        f = np.zeros((n_clusters, n_examples))
        for cc in range(n_clusters):
            f[cc, :] = _vmf(X, concentrations[cc], centers[cc, :])

        posterior = np.zeros((n_clusters, n_examples))
        if posterior_type == 'soft':
            posterior = np.tile(weights.T, (n_examples, 1)).T * f
            for ee in range(n_examples):
                posterior[:, ee] /= np.sum(posterior[:, ee])

        elif posterior_type == 'hard':
            weighted_f = np.tile(weights.T, (n_examples, 1)).T * f
            for ee in range(n_examples):
                posterior[np.argmax(weighted_f[:, ee]), ee] = 1.0

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

    # labels come for free via posterior
    labels = np.zeros((n_examples, ))
    for ee in range(n_examples):
        labels[ee] = np.argmax(posterior[:, ee])

    inertia = _inertia_from_labels(X, centers, labels)

    return centers, weights, concentrations, posterior, labels, inertia


def moVMF(X, n_clusters, posterior_type='soft', n_init=10, n_jobs=1,
            max_iter=300, verbose=False, init='k-means++', random_state=None,
            tol=1e-6, copy_x=True):
    """
    """
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    best_inertia = np.infty
    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # cluster on the sphere
            centers, weights, concentrations, posterior, labels, inertia = _moVMF(
                    X,
                    n_clusters,
                    posterior_type=posterior_type,
                    max_iter=max_iter,
                    verbose=verbose,
                    init=init,
                    random_state=random_state,
                    tol=tol)

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_weights = weights.copy()
                best_concentrations = concentrations.copy()
                best_posterior = posterior.copy()
                best_inertia = inertia
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_moVMF)(X,
                    n_clusters,
                    posterior_type=posterior_type,
                    max_iter=max_iter,
                    verbose=verbose,
                    init=init,
                    random_state=random_state,
                    tol=tol)
            for seed in seeds)

        # Get results with the lowest inertia
        centers, weights, concentrations, posterior, labels, inertia = \
                zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]

    return (best_centers, best_labels, best_inertia, best_weights,
        best_concentrations, best_posterior)



class VonMisesFisherMixture(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    """
    def __init__(self, n_clusters, posterior_type='soft', n_init=10, n_jobs=1,
            max_iter=300, verbose=False, init='k-means++', random_state=None,
            tol=1e-6, copy_x=True):
        self.n_clusters = n_clusters
        self.posterior_type = posterior_type
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.tol = tol
        self.copy_x = copy_x

        # results from algorithm
        self.cluster_centers_ = None
        self.labels = None
        self.intertia_ = None
        self.weights_ = None
        self.concentrations_ = None
        self.posterior_ = None


    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        return X


    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES,
                        warn_on_dtype=True)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X


    def fit(self, X, y=None):
        """Compute mixture of von Mises Fisher clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        (self.cluster_centers_, self.labels_, self.inertia_, self.weights_,
                self.concentrations_, self.posterior_) = moVMF(
                X, self.n_clusters, posterior_type=self.posterior_type,
                n_init=self.n_init, n_jobs=self.n_jobs, max_iter=self.max_iter,
                verbose=self.verbose, init=self.init,
                random_state=random_state, tol=self.tol, copy_x=self.copy_x)

        return self


    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_


    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        X = self._check_fit_data(X)
        return self.fit(X)._transform(X)


    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the cosine distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return self._transform(X)


    def _transform(self, X):
        """guts of transform method; no input validation"""
        return cosine_distances(X, self.cluster_centers_)


    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Note:  Does not check that each point is on the sphere.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return _labels_inertia(X, self.cluster_centers_)[0]


    def score(self, X, y=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return -_labels_inertia(X, self.cluster_centers_)[1]

