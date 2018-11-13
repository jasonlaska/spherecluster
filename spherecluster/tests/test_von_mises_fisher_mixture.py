import pytest
import scipy as sp
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from spherecluster import VonMisesFisherMixture
from spherecluster import von_mises_fisher_mixture
from spherecluster import sample_vMF


def test_vmf_log_dense():
    """
    Test that approximation approaches whatever scipy has.
    """
    n_examples = 2
    n_features = 50

    kappas = np.linspace(2, 600, 20)

    mu = np.random.randn(n_features)
    mu /= np.linalg.norm(mu)

    X = np.random.randn(n_examples, n_features)
    for ee in range(n_examples):
        X[ee, :] /= np.linalg.norm(X[ee, :])

    diffs = []
    for kappa in kappas:
        v = von_mises_fisher_mixture._vmf_log(X, kappa, mu)

        v_approx = von_mises_fisher_mixture._vmf_log_asymptotic(X, kappa, mu)

        normalized_approx_diff = np.linalg.norm(v - v_approx) / np.linalg.norm(v)
        print(normalized_approx_diff)
        diffs.append(normalized_approx_diff)

    assert diffs[0] > 10 * diffs[-1]


def test_vmf_log_detect_breakage():
    """
    Find where scipy approximation breaks down.
    This doesn't really test anything but demonstrates where approximation
    should be applied instead.
    """
    n_examples = 3
    kappas = [5, 30, 100, 1000, 5000]
    n_features = range(2, 500)

    breakage_points = []
    for kappa in kappas:
        first_breakage = None
        for n_f in n_features:
            mu = np.random.randn(n_f)
            mu /= np.linalg.norm(mu)

            X = np.random.randn(n_examples, n_f)
            for ee in range(n_examples):
                X[ee, :] /= np.linalg.norm(X[ee, :])

            try:
                von_mises_fisher_mixture._vmf_log(X, kappa, mu)
            except:
                if first_breakage is None:
                    first_breakage = n_f

        breakage_points.append(first_breakage)
        print(
            "Scipy vmf_log breaks for kappa={} at n_features={}".format(
                kappa, first_breakage
            )
        )

    print(breakage_points)
    assert_array_equal(breakage_points, [141, 420, 311, 3, 3])


def test_maximization():
    num_points = 5000
    n_features = 500
    posterior = np.ones((1, num_points))

    kappas = [5000, 8000, 16400]
    for kappa in kappas:
        mu = np.random.randn(n_features)
        mu /= np.linalg.norm(mu)

        X = sample_vMF(mu, kappa, num_points)

        centers, weights, concentrations = von_mises_fisher_mixture._maximization(
            X, posterior
        )

        print("center estimate error", np.linalg.norm(centers[0, :] - mu))
        print(
            "kappa estimate",
            np.abs(kappa - concentrations[0]) / kappa,
            kappa,
            concentrations[0],
        )

        assert_almost_equal(1., weights[0])
        assert_almost_equal(0.0, np.abs(kappa - concentrations[0]) / kappa, decimal=2)
        assert_almost_equal(0.0, np.linalg.norm(centers[0, :] - mu), decimal=2)


@pytest.mark.parametrize(
    "params_in",
    [
        {"posterior_type": "soft"},
        {"posterior_type": "hard"},
        {"posterior_type": "soft", "n_jobs": 2},
        {"posterior_type": "hard", "n_jobs": 3},
        {"posterior_type": "hard", "force_weights": np.ones(5) / 5.},
        {"posterior_type": "soft", "n_jobs": -1},
    ],
)
def test_integration_dense(params_in):
    n_clusters = 5
    n_examples = 20
    n_features = 100
    X = np.random.randn(n_examples, n_features)
    for ee in range(n_examples):
        X[ee, :] /= np.linalg.norm(X[ee, :])

    params_in.update({"n_clusters": n_clusters})
    movmf = VonMisesFisherMixture(**params_in)
    movmf.fit(X)

    assert movmf.cluster_centers_.shape == (n_clusters, n_features)
    assert len(movmf.concentrations_) == n_clusters
    assert len(movmf.weights_) == n_clusters
    assert len(movmf.labels_) == n_examples
    assert len(movmf.posterior_) == n_clusters

    for center in movmf.cluster_centers_:
        assert_almost_equal(np.linalg.norm(center), 1.0)

    for concentration in movmf.concentrations_:
        assert concentration > 0

    for weight in movmf.weights_:
        assert not np.isnan(weight)

    plabels = movmf.predict(X)
    assert_array_equal(plabels, movmf.labels_)

    ll = movmf.log_likelihood(X)
    ll_labels = np.zeros(movmf.labels_.shape)
    for ee in range(n_examples):
        ll_labels[ee] = np.argmax(ll[:, ee])

    assert_array_equal(ll_labels, movmf.labels_)


@pytest.mark.parametrize(
    "params_in",
    [
        {"posterior_type": "soft"},
        {"posterior_type": "hard"},
        {"posterior_type": "soft", "n_jobs": 2},
        {"posterior_type": "hard", "n_jobs": 3},
        {"posterior_type": "hard", "force_weights": np.ones(5) / 5.},
        {"posterior_type": "soft", "n_jobs": -1},
    ],
)
def test_integration_sparse(params_in):
    n_clusters = 5
    n_examples = 20
    n_features = 100
    n_nonzero = 10
    X = sp.sparse.csr_matrix((n_examples, n_features))
    for ee in range(n_examples):
        ridx = np.random.randint(n_features, size=(n_nonzero))
        random_values = np.random.randn(n_nonzero)
        random_values = random_values / np.linalg.norm(random_values)
        X[ee, ridx] = random_values

    params_in.update({"n_clusters": n_clusters})
    movmf = VonMisesFisherMixture(**params_in)
    movmf.fit(X)

    assert movmf.cluster_centers_.shape == (n_clusters, n_features)
    assert len(movmf.concentrations_) == n_clusters
    assert len(movmf.weights_) == n_clusters
    assert len(movmf.labels_) == n_examples
    assert len(movmf.posterior_) == n_clusters

    for center in movmf.cluster_centers_:
        assert_almost_equal(np.linalg.norm(center), 1.0)

    for concentration in movmf.concentrations_:
        assert concentration > 0

    for weight in movmf.weights_:
        assert not np.isnan(weight)

    plabels = movmf.predict(X)
    assert_array_equal(plabels, movmf.labels_)

    ll = movmf.log_likelihood(X)
    ll_labels = np.zeros(movmf.labels_.shape)
    for ee in range(n_examples):
        ll_labels[ee] = np.argmax(ll[:, ee])

    assert_array_equal(ll_labels, movmf.labels_)
