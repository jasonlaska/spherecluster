from sklearn.utils.estimator_checks import check_estimator
from ..spherical_kmeans import SphericalKMeans
from ..von_mises_fisher_mixture import VonMisesFisherMixture


def test_estimator_spherical_k_means():
    return check_estimator(SphericalKMeans)


def test_estimator_vmf():
    return check_estimator(VonMisesFisherMixture)
