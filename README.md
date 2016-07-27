# spherecluster
Clustering routines for the unit sphere

Notes to clean up later:

#Implements:

- Spherical K-Means

- Mixture of von Mises-Fisher
-- hard-moVMF
-- soft-moVMF

from
Clustering on the Unit Hypersphere using von Mises-Fisher Distributions
http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf

# Examples:

## Small mix

<img src="images/small_mix_2d.png" alt="Small mix 2d" width="70">

![Small mix 3d](images/small_mix_3d.png =70x?raw=true "Small mix 3d")

## Document clustering

![Document clustering](images/document_clustering.png =70x?raw=true "Document clustering")

#Also see:
http://nipy.sourceforge.net/nipy/devel/api/generated/nipy.algorithms.clustering.von_mises_fisher_mixture.html
https://github.com/nipy/nipy/blob/master/nipy/algorithms/clustering/von_mises_fisher_mixture.py