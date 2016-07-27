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

![Small mix 2d](images/small_mix_2d.png?raw=true "Small mix 2d" =250x)
![Small mix 3d](images/small_mix_3d.png?raw=true "Small mix 3d" =250x)

## Document clustering

![Document clustering](images/document_clustering.png?raw=true "Document clustering" =250x)

#Also see:
http://nipy.sourceforge.net/nipy/devel/api/generated/nipy.algorithms.clustering.von_mises_fisher_mixture.html
https://github.com/nipy/nipy/blob/master/nipy/algorithms/clustering/von_mises_fisher_mixture.py