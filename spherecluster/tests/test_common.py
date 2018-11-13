from sklearn.utils.estimator_checks import check_estimator
from spherecluster import SphericalKMeans


def test_estimator_spherical_k_means():
    return check_estimator(SphericalKMeans)
