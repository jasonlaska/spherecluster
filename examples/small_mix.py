import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import sys
sys.path.append('../spherecluster')

import sample_vMF
from spherecluster import SphericalKMeans

plt.ion()

'''
Implements "small-mix" example from
"Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"

Provides a basic smell test that the algoriths are performing as intended.
'''

###############################################################################
# Generate small-mix dataset
mu_0 = np.array([-0.251, -0.968])
mu_1 = np.array([0.399, 0.917])
mus = [mu_0, mu_1]
kappa_0 = 3 # concentration parameter
kappa_1 = 3 # concentration parameter
kappas = [kappa_0, kappa_1]
num_points_per_class = 50

X_0 = sample_vMF.vMF(mu_0, kappa_0, num_points_per_class)
X_1 = sample_vMF.vMF(mu_1, kappa_1, num_points_per_class)
X = np.zeros((2 * num_points_per_class, 2))
X[:num_points_per_class, :] = X_0
X[num_points_per_class:, :] = X_1

###############################################################################
# K-Means clustering
km = KMeans(n_clusters=2, init='k-means++', n_init=10)
km.fit(X)

cdists = []
for center in km.cluster_centers_:
    cdists.append(np.linalg.norm(mus[0] - center))

km_mu_0_idx = np.argmin(cdists)
km_mu_1_idx = 1 - km_mu_0_idx

km_mu_0_error = np.linalg.norm(mus[0] - km.cluster_centers_[km_mu_0_idx])
km_mu_1_error = np.linalg.norm(mus[1] - km.cluster_centers_[km_mu_1_idx])
km_mu_0_error_norm = np.linalg.norm(mus[0] - km.cluster_centers_[km_mu_0_idx] / np.linalg.norm(km.cluster_centers_[km_mu_0_idx]))
km_mu_1_error_norm = np.linalg.norm(mus[1] - km.cluster_centers_[km_mu_1_idx] / np.linalg.norm(km.cluster_centers_[km_mu_1_idx]))

###############################################################################
# Spherical K-Means clustering
skm = SphericalKMeans(n_clusters=2, init='k-means++', n_init=10)
skm.fit(X)

cdists = []
for center in skm.cluster_centers_:
    cdists.append(np.linalg.norm(mus[0] - center))

skm_mu_0_idx = np.argmin(cdists)
skm_mu_1_idx = 1 - skm_mu_0_idx

skm_mu_0_error = np.linalg.norm(mus[0] - skm.cluster_centers_[skm_mu_0_idx])
skm_mu_1_error = np.linalg.norm(mus[1] - skm.cluster_centers_[skm_mu_1_idx])


###############################################################################
# Show results

# Original data
plt.figure()
for ex in X_0:
    plt.plot(ex[0], ex[1], 'r+')
    plt.hold(True)
for ex in X_1:
    plt.plot(ex[0], ex[1], 'b+')
    plt.hold(True)
plt.axis('equal')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.title('Original data')
plt.show()

# K-means labels
plt.figure()
for ex, label in zip(X, km.labels_):
    if label == km_mu_0_idx:
        plt.plot(ex[0], ex[1], 'r+')
    else:
        plt.plot(ex[0], ex[1], 'b+')
    plt.hold(True)

plt.axis('equal')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.title('K-means clustering')
plt.show()

# Spherical K-means labels
plt.figure()
for ex, label in zip(X, skm.labels_):
    if label == skm_mu_0_idx:
        plt.plot(ex[0], ex[1], 'r+')
    else:
        plt.plot(ex[0], ex[1], 'b+')
    plt.hold(True)

plt.axis('equal')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.title('Spherical K-means clustering')
plt.show()


print 'mu 0: {}'.format(mu_0)
print 'mu 0: {} (kmeans), error={} ({})'.format(km.cluster_centers_[km_mu_0_idx], km_mu_0_error, km_mu_0_error_norm)
print 'mu 0: {} (spherical kmeans), error={}'.format(skm.cluster_centers_[skm_mu_0_idx], skm_mu_0_error)
print '---'
print 'mu 1: {}'.format(mu_1)
print 'mu 1: {} (kmeans), error={} ({})'.format(km.cluster_centers_[km_mu_1_idx], km_mu_1_error, km_mu_1_error_norm)
print 'mu 1: {} (spherical kmeans), error={}'.format(skm.cluster_centers_[skm_mu_1_idx], skm_mu_1_error)

# sanity check, is spherical k-means just returning normalized result?
#print km.cluster_centers_[km_mu_0_idx] / np.linalg.norm(km.cluster_centers_[km_mu_0_idx])
#print km.cluster_centers_[km_mu_1_idx] / np.linalg.norm(km.cluster_centers_[km_mu_1_idx])


raw_input()
