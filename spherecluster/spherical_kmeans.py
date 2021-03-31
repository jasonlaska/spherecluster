import warnings

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from sklearn.cluster import KMeans

from sklearn.cluster import _k_means_fast as _k_means
from sklearn.cluster import _k_means_lloyd
from sklearn.cluster._kmeans import (
    _check_sample_weight,
    _labels_inertia,
    _tolerance,
)
from sklearn.preprocessing import normalize
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils.validation import _num_samples
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


def _spherical_kmeans_single_lloyd(
    X,
    n_clusters,
    centers_init,
    sample_weight=None,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
    n_threads=1,
):
    """
    Modified from sklearn.cluster.k_means_.k_means_single_lloyd.
    """
    random_state = check_random_state(random_state)

    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

    best_labels, best_inertia, best_centers = None, None, None

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = _k_means_lloyd.lloyd_iter_chunked_sparse
        _inertia = _k_means._inertia_sparse
    else:
        lloyd_iter = _k_means_lloyd.lloyd_iter_chunked_dense
        _inertia = _k_means._inertia_dense

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()

        # labels assignment
        # TODO: _labels_inertia should be done with cosine distance
        #       since ||a - b|| = 2(1 - cos(a,b)) when a,b are unit normalized
        #       this doesn't really matter.
        labels, inertia = _labels_inertia(
            X,
            sample_weight,
            x_squared_norms,
            centers,
        )
        lloyd_iter(X, sample_weight, x_squared_norms, centers, centers_new,
                   weight_in_clusters, labels, center_shift, n_threads)

        centers, centers_new = centers_new, centers
        # l2-normalize centers (this is the main contibution here)
        centers = normalize(centers)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print(
                    "Converged at iteration %d: "
                    "center shift %e within tolerance %e" % (i, center_shift_total, tol)
                )
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = _labels_inertia(
            X,
            sample_weight,
            x_squared_norms,
            best_centers,
        )

    return best_labels, best_inertia, best_centers, i + 1


def spherical_k_means(
    X,
    n_clusters,
    centers_init,
    sample_weight=None,
    init="k-means++",
    n_init=10,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    n_jobs=1,
    algorithm="auto",
    return_n_iter=False,
):
    """Modified from sklearn.cluster.k_means_.k_means.
    """
    if n_init <= 0:
        raise ValueError(
            "Invalid number of initializations."
            " n_init=%d must be bigger than zero." % n_init
        )
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError(
            "Number of iterations should be a positive number,"
            " got %d instead" % max_iter
        )

    best_inertia = np.infty
    # avoid forcing order when copy_x=False
    order = "C" if copy_x else None
    X = check_array(
        X, accept_sparse="csr", dtype=[np.float64, np.float32], order=order, copy=copy_x
    )
    # verify that the number of samples given is larger than k
    if _num_samples(X) < n_clusters:
        raise ValueError(
            "n_samples=%d should be >= n_clusters=%d" % (_num_samples(X), n_clusters)
        )
    tol = _tolerance(X, tol)

    if hasattr(init, "__array__"):
        if n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: "
                "performing only one init in k-means instead of n_init=%d" % n_init,
                RuntimeWarning,
                stacklevel=2,
            )
            n_init = 1

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = _spherical_kmeans_single_lloyd(
                X,
                n_clusters,
                centers_init=centers_init,
                sample_weight=sample_weight,
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                random_state=random_state,
            )

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_spherical_kmeans_single_lloyd)(
                X,
                n_clusters,
                sample_weight,
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                # Change seed to ensure variety
                random_state=seed,
            )
            for seed in seeds
        )

        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


class SphericalKMeans(KMeans):
    """Spherical K-Means clustering

    Modfication of sklearn.cluster.KMeans where cluster centers are normalized
    (projected onto the sphere) in each iteration.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    normalize : boolean, default True
        Normalize the input to have unnit norm.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        n_jobs=1,
        verbose=0,
        random_state=None,
        copy_x=True,
        normalize=True,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.normalize = normalize

    def _check_params(self, X):
        # n_jobs
        if self.n_jobs != 'deprecated':
            warnings.warn("'n_jobs' was deprecated in version 0.23 and will be"
                          " removed in 1.0 (renaming of 0.25).", FutureWarning)
            self._n_threads = self.n_jobs
        else:
            self._n_threads = None
        self._n_threads = _openmp_effective_n_threads(self._n_threads)

        # n_init
        if self.n_init <= 0:
            raise ValueError(
                f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"n_samples={X.shape[0]} should be >= "
                             f"n_clusters={self.n_clusters}.")

        # tol
        self._tol = _tolerance(X, self.tol)

        # init
        if not (hasattr(self.init, '__array__') or callable(self.init)
                or (isinstance(self.init, str)
                    and self.init in ["k-means++", "random"])):
            raise ValueError(
                f"init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{self.init}' instead.")

        if hasattr(self.init, '__array__') and self._n_init != 1:
            warnings.warn(
                f"Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self._n_init}.", RuntimeWarning, stacklevel=2)
            self._n_init = 1


    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------

        X : array-like or sparse matrix, shape=(n_samples, n_features)

        y : Ignored
            not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        """
        if self.normalize:
            X = normalize(X)
        self._check_params(X)

        random_state = check_random_state(self.random_state)

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype.type, order="C", copy=True)
            self._validate_center_shape(X, init)

        # TODO: add check that all data is unit-normalized
        x_squared_norms = row_norms(X, squared=True)
        centers_init = self._init_centroids(
            X,
            init=self.init,
            random_state=random_state,
            x_squared_norms=x_squared_norms
        )

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = spherical_k_means(
            X,
            n_clusters=self.n_clusters,
            centers_init=centers_init,
            sample_weight=sample_weight,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            verbose=self.verbose,
            tol=self.tol,
            random_state=random_state,
            copy_x=self.copy_x,
            n_jobs=self.n_jobs,
            return_n_iter=True,
        )

        return self
