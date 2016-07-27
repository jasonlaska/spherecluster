# Clustering on the unit hypersphere in scikit-learn
Clustering routines for the unit sphere.

<p style="text-align:center;"><img src="images/sphere_w_clusters.png" alt="Mixture of von Mises Fisher" width="500"></p>

## Algorithms
This package implements the three algorithms outlined in ["Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"](http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf), Banerjee et al., JMLR 2005.

1. Spherical K-means
Spherical K-means differs from conventional K-means in that it projects the estimatoed cluster centroids on the the unit sphere at the end of each maximization step (i.e., normalizes the centroids).

2. Mixtures of von Mises Fisher distributions (movMF)

    The [von Mises Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) is commonly used in directional statistics.  Much like the Gaussian distribution has a mean and variance, the von Mises Fisher distribution has a mean direction $\mu$ and a concentration paramater $\kappa$, however, all points are on the unit hypersphere.

    If we model our data as a [mixture](https://en.wikipedia.org/wiki/Mixture_model) of von Mises Fisher Distributions, we have an additional `weight` parameter for each distribution in the mixture, and can cluster our data accordingly.

    - soft-movMF: Estimates the posterior on each example for each class.

    - hard-movMF: Sets the posterior on each example to be 1 for a single class and 0 for all others by selecting the location of the max value in the estimator soft posterior.


## Other goodies

- utility sampling from a multivariate von Mises Fisher distribution
- python implementation of approximation to the modified Bessel function of the first kind

## Usage
Both `SphericalKMeans` and `VonMisesFisherMixture` are standard sklearn estimators and mirror the parameter names for `sklearn.cluster.kmeans`.

    # Find K clusters from data matrix X (n_examples x n_features)

    # spherical k-means
    from spherecluster import SphericalKMeans
    skm = SphericalKMeans(n_clusters=K)
    skm.fit(X)

    # skm.cluster_centers_
    # skm.labels_
    # skm.intertia_

    # movMF-soft
    from spherecluster import VonMisesFisherMixture
    vmf_soft = VonMisesFisherMixture(n_clusters=K, posterior_type='soft')
    vmf_soft.fit(X)

    # vmf_soft.cluster_centers_
    # vmf_soft.labels_
    # vmf_soft.weights_
    # vmf_soft.concentrations_
    # vmf_soft.intertia_

    # movMF-hard
    from spherecluster import VonMisesFisherMixture
    vmf_hard = VonMisesFisherMixture(n_clusters=K, posterior_type='hard')
    vmf_hard.fit(X)

    # vmf_hard.cluster_centers_
    # vmf_hard.labels_
    # vmf_hard.weights_
    # vmf_hard.concentrations_
    # vmf_hard.intertia_

Other notes of interest:

- `SphericalKMeans` projects each centroid onto the sphere at the end of each EM iteration and is therefore a small modification to `sklearn.cluster.kmeans`
- X can be a dense `numpy.array` or a sparse `scipy.sparse.csr_matrix`
- This has been tested with sparse documents as large as `n_features = 43256` but may encounter numerical instability when `n_features` is very large
- `cluster_centers_` in `VonMisesFisherMixture` are dense vectors in current implementation

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