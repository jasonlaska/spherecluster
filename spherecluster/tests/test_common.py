from sklearn.utils.estimator_checks import check_estimator
from spherecluster import SphericalKMeans, VonMisesFisherMixture

def test_estimator_spherical_k_means():
    return check_estimator(SphericalKMeans)


def test_estimator_von_mises_fisher():
    return check_estimator(VonMisesFisherMixture)
