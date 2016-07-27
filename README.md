# Clustering on the unit hypersphere in scikit-learn
Clustering routines for the unit sphere

#Algorithms

- Spherical K-Means

- Mixture of von Mises Fisher
-- hard-moVMF
-- soft-moVMF

from
Clustering on the Unit Hypersphere using von Mises-Fisher Distributions
http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf

# Usage
The tools in this package are sklearn estimators and mirror the interface and parameter names for `sklearn.cluster.kmeans`.

    # Find K clusters from data matrix X (n_examples x n_features)

    # Spherical K-Means
    from spherecluster import SphericalKMeans
    skm = SphericalKMeans(n_clusters=K)
    skm.fit(X)

    print skm.cluster_centers_
    print skm.labels_

    # movMF-soft
    from spherecluster import VonMisesFisherMixture
    vmf_soft = VonMisesFisherMixture(n_clusters=true_k, posterior_type='soft')
    vmf_soft.fit(X)

    print vmf_soft.cluster_centers_
    print vmf_soft.labels_
    print vmf_soft.weights_
    print vmf_soft.concentrations_

    # movMF-hard
    from spherecluster import VonMisesFisherMixture
    vmf_hard = VonMisesFisherMixture(n_clusters=true_k, posterior_type='hard')
    vmf_hard.fit(X)

    print vmf_hard.cluster_centers_
    print vmf_hard.labels_
    print vmf_hard.weights_
    print vmf_hard.concentrations_

# Examples

## Small mix

<img src="images/small_mix_2d.png" alt="Small mix 2d" width="500">
<img src="images/small_mix_3d.png" alt="Small mix 3d" width="500">


## Document clustering

<img src="images/document_clustering.png" alt="Document clustering" width="800">


# Acknowledgments / Attribution


# Other implementations
http://nipy.sourceforge.net/nipy/devel/api/generated/nipy.algorithms.clustering.von_mises_fisher_mixture.html
https://github.com/nipy/nipy/blob/master/nipy/algorithms/clustering/von_mises_fisher_mixture.py