import scipy as sp
import numpy as np
from numpy.testing import assert_almost_equal
from spherecluster import VonMisesFisherMixture


class TestVonMisesFisherMixture(object):
    def test_integration_multiple_jobs(self):
        n_clusters = 5
        n_examples = 20
        n_features = 100
        X = np.random.randn(n_examples, n_features)
        for ee in range(n_examples):
            X[ee, :] /= np.linalg.norm(X[ee, :])

        movmf = VonMisesFisherMixture(
                n_clusters=n_clusters,
                posterior_type='soft',
                n_jobs=2)
        movmf.fit(X)

        assert movmf.cluster_centers_.shape == (n_clusters, n_features)
        assert len(movmf.concentrations_) == n_clusters
        assert len(movmf.weights_) == n_clusters

        for center in movmf.cluster_centers_:
            assert_almost_equal(np.linalg.norm(center), 1.0)

        for concentration in movmf.concentrations_:
            assert concentration > 0

        for weight in movmf.weights_:
            assert not np.isnan(weight)

    def test_integration_dense_soft(self):
        n_clusters = 5
        n_examples = 20
        n_features = 100
        X = np.random.randn(n_examples, n_features)
        for ee in range(n_examples):
            X[ee, :] /= np.linalg.norm(X[ee, :])

        movmf = VonMisesFisherMixture(
                n_clusters=n_clusters,
                posterior_type='soft')
        movmf.fit(X)

        assert movmf.cluster_centers_.shape == (n_clusters, n_features)
        assert len(movmf.concentrations_) == n_clusters
        assert len(movmf.weights_) == n_clusters

        for center in movmf.cluster_centers_:
            assert_almost_equal(np.linalg.norm(center), 1.0)

        for concentration in movmf.concentrations_:
            assert concentration > 0

        for weight in movmf.weights_:
            assert not np.isnan(weight)


    def test_integration_dense_hard(self):
        n_clusters = 5
        n_examples = 20
        n_features = 100
        X = np.random.randn(n_examples, n_features)
        for ee in range(n_examples):
            X[ee, :] /= np.linalg.norm(X[ee, :])

        movmf = VonMisesFisherMixture(
                n_clusters=n_clusters,
                posterior_type='hard')
        movmf.fit(X)

        assert movmf.cluster_centers_.shape == (n_clusters, n_features)
        assert len(movmf.concentrations_) == n_clusters
        assert len(movmf.weights_) == n_clusters

        for center in movmf.cluster_centers_:
            assert_almost_equal(np.linalg.norm(center), 1.0)

        for concentration in movmf.concentrations_:
            assert concentration > 0

        for weight in movmf.weights_:
            assert not np.isnan(weight)


    def test_integration_sparse_soft(self):
        n_clusters = 5
        n_examples = 20
        n_features = 100
        n_nonzero = 10
        X = sp.sparse.csr_matrix((n_examples, n_features))
        for ee in range(n_examples):
            ridx = np.random.randint(n_features, size=(n_nonzero))
            X[ee, ridx] = np.random.randn(n_nonzero)
            X[ee, :] /= sp.sparse.linalg.norm(X[ee, :])

        movmf = VonMisesFisherMixture(
                n_clusters=n_clusters,
                posterior_type='soft')
        movmf.fit(X)

        assert movmf.cluster_centers_.shape == (n_clusters, n_features)
        assert len(movmf.concentrations_) == n_clusters
        assert len(movmf.weights_) == n_clusters

        for center in movmf.cluster_centers_:
            assert_almost_equal(np.linalg.norm(center), 1.0)

        for concentration in movmf.concentrations_:
            assert concentration > 0

        for weight in movmf.weights_:
            assert not np.isnan(weight)


    def test_integration_sparse_hard(self):
        n_clusters = 5
        n_examples = 20
        n_features = 100
        n_nonzero = 10
        X = sp.sparse.csr_matrix((n_examples, n_features))
        for ee in range(n_examples):
            ridx = np.random.randint(n_features, size=(n_nonzero))
            X[ee, ridx] = np.random.randn(n_nonzero)
            X[ee, :] /= sp.sparse.linalg.norm(X[ee, :])

        movmf = VonMisesFisherMixture(
                n_clusters=n_clusters,
                posterior_type='hard')
        movmf.fit(X)

        assert movmf.cluster_centers_.shape == (n_clusters, n_features)
        assert len(movmf.concentrations_) == n_clusters
        assert len(movmf.weights_) == n_clusters

        for center in movmf.cluster_centers_:
            assert_almost_equal(np.linalg.norm(center), 1.0)

        for concentration in movmf.concentrations_:
            assert concentration > 0

        for weight in movmf.weights_:
            assert not np.isnan(weight)