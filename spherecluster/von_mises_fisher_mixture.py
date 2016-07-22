import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster.k_means_ import _init_centroids, _labels_inertia

#class VonMisesFisher(BaseEstimator, ClusterMixin, TransformerMixin):
