from sklearn.utils.estimator_checks import check_estimator
from spherecluster import SphericalKMeans


class TestCommon(object):
    def test_estimator_spherical_k_means(self):
        return check_estimator(SphericalKMeans)
