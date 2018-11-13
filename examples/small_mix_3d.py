import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics

from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture
from spherecluster import sample_vMF

plt.ion()


def r_input(val=None):
    val = val or ''
    if sys.version_info[0] >= 3:
        return eval(input(val))

    return raw_input(val)


###############################################################################
# Generate small-mix dataset
mu_0 = np.array([-0.251, -0.968, -0.105])
mu_0 = mu_0 / np.linalg.norm(mu_0)
mu_1 = np.array([0.399, 0.917, 0.713])
mu_1 = mu_1 / np.linalg.norm(mu_1)
mus = [mu_0, mu_1]
kappa_0 = 8  # concentration parameter
kappa_1 = 2  # concentration parameter
kappas = [kappa_0, kappa_1]
num_points_per_class = 300

X_0 = sample_vMF(mu_0, kappa_0, num_points_per_class)
X_1 = sample_vMF(mu_1, kappa_1, num_points_per_class)
X = np.zeros((2 * num_points_per_class, 3))
X[:num_points_per_class, :] = X_0
X[num_points_per_class:, :] = X_1
labels = np.zeros((2 * num_points_per_class, ))
labels[num_points_per_class:] = 1


###############################################################################
# K-Means clustering
km = KMeans(n_clusters=2, init='k-means++', n_init=20)
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
skm = SphericalKMeans(n_clusters=2, init='k-means++', n_init=20)
skm.fit(X)

cdists = []
for center in skm.cluster_centers_:
    cdists.append(np.linalg.norm(mus[0] - center))

skm_mu_0_idx = np.argmin(cdists)
skm_mu_1_idx = 1 - skm_mu_0_idx

skm_mu_0_error = np.linalg.norm(mus[0] - skm.cluster_centers_[skm_mu_0_idx])
skm_mu_1_error = np.linalg.norm(mus[1] - skm.cluster_centers_[skm_mu_1_idx])


###############################################################################
# Mixture of von Mises Fisher clustering (soft)
vmf_soft = VonMisesFisherMixture(n_clusters=2, posterior_type='soft', n_init=20)
vmf_soft.fit(X)

cdists = []
for center in vmf_soft.cluster_centers_:
    cdists.append(np.linalg.norm(mus[0] - center))

vmf_soft_mu_0_idx = np.argmin(cdists)
vmf_soft_mu_1_idx = 1 - vmf_soft_mu_0_idx

vmf_soft_mu_0_error = np.linalg.norm(
        mus[0] - vmf_soft.cluster_centers_[vmf_soft_mu_0_idx])
vmf_soft_mu_1_error = np.linalg.norm(
        mus[1] - vmf_soft.cluster_centers_[vmf_soft_mu_1_idx])


###############################################################################
# Mixture of von Mises Fisher clustering (hard)
vmf_hard = VonMisesFisherMixture(n_clusters=2, posterior_type='hard', n_init=20)
vmf_hard.fit(X)

cdists = []
for center in vmf_hard.cluster_centers_:
    cdists.append(np.linalg.norm(mus[0] - center))

vmf_hard_mu_0_idx = np.argmin(cdists)
vmf_hard_mu_1_idx = 1 - vmf_hard_mu_0_idx

vmf_hard_mu_0_error = np.linalg.norm(
        mus[0] - vmf_hard.cluster_centers_[vmf_hard_mu_0_idx])
vmf_hard_mu_1_error = np.linalg.norm(
        mus[1] - vmf_hard.cluster_centers_[vmf_hard_mu_1_idx])


###############################################################################
# Show results

# Original data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(3, 2, 1, aspect='equal', projection='3d',
        adjustable='box-forced', xlim=[-1.1, 1.1], ylim=[-1.1, 1.1],
        zlim=[-1.1, 1.1])
ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], c='r')
ax.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], c='b')
ax.set_aspect('equal')
plt.title('Original data')
plt.show()

# K-means labels
ax = fig.add_subplot(3, 2, 3, aspect='equal', projection='3d',
        adjustable='box-forced', xlim=[-1.1, 1.1], ylim=[-1.1, 1.1],
        zlim=[-1.1, 1.1])
ax.scatter(X[km.labels_ == km_mu_0_idx, 0], X[km.labels_ == km_mu_0_idx, 1], X[km.labels_ == km_mu_0_idx, 2], c='r')
ax.scatter(X[km.labels_ == km_mu_1_idx, 0], X[km.labels_ == km_mu_1_idx, 1], X[km.labels_ == km_mu_1_idx, 2], c='b')
ax.set_aspect('equal')
plt.title('K-means clustering')
plt.show()

# Spherical K-means labels
ax = fig.add_subplot(3, 2, 4, aspect='equal', projection='3d',
        adjustable='box-forced', xlim=[-1.1, 1.1], ylim=[-1.1, 1.1],
        zlim=[-1.1, 1.1])
ax.scatter(X[skm.labels_ == skm_mu_0_idx, 0], X[skm.labels_ == skm_mu_0_idx, 1], X[skm.labels_ == skm_mu_0_idx, 2], c='r')
ax.scatter(X[skm.labels_ == skm_mu_1_idx, 0], X[skm.labels_ == skm_mu_1_idx, 1], X[skm.labels_ == skm_mu_1_idx, 2], c='b')
ax.set_aspect('equal')
plt.title('Spherical K-means clustering')
plt.show()

# von Mises Fisher soft labels
ax = fig.add_subplot(3, 2, 5, aspect='equal', projection='3d',
        adjustable='box-forced', xlim=[-1.1, 1.1], ylim=[-1.1, 1.1],
        zlim=[-1.1, 1.1])
ax.scatter(X[vmf_soft.labels_ == vmf_soft_mu_0_idx, 0], X[vmf_soft.labels_ == vmf_soft_mu_0_idx, 1], X[vmf_soft.labels_ == vmf_soft_mu_0_idx, 2], c='r')
ax.scatter(X[vmf_soft.labels_ == vmf_soft_mu_1_idx, 0], X[vmf_soft.labels_ == vmf_soft_mu_1_idx, 1], X[vmf_soft.labels_ == vmf_soft_mu_1_idx, 2], c='b')
ax.set_aspect('equal')
plt.title('soft-movMF clustering')
plt.show()

# von Mises Fisher hard labels
ax = fig.add_subplot(3, 2, 6, aspect='equal', projection='3d',
        adjustable='box-forced', xlim=[-1.1, 1.1], ylim=[-1.1, 1.1],
        zlim=[-1.1, 1.1])
ax.scatter(X[vmf_hard.labels_ == vmf_hard_mu_0_idx, 0], X[vmf_hard.labels_ == vmf_hard_mu_0_idx, 1], X[vmf_hard.labels_ == vmf_hard_mu_0_idx, 2], c='r')
ax.scatter(X[vmf_hard.labels_ == vmf_hard_mu_1_idx, 0], X[vmf_hard.labels_ == vmf_hard_mu_1_idx, 1], X[vmf_hard.labels_ == vmf_hard_mu_1_idx, 2], c='b')
ax.set_aspect('equal')
plt.title('hard-movMF clustering')
plt.show()


print('mu 0: {}'.format(mu_0))
print('mu 0: {} (kmeans), error={} ({})'.format(km.cluster_centers_[km_mu_0_idx], km_mu_0_error, km_mu_0_error_norm))
print('mu 0: {} (spherical kmeans), error={}'.format(skm.cluster_centers_[skm_mu_0_idx], skm_mu_0_error))
print('mu 0: {} (vmf-soft), error={}'.format(vmf_soft.cluster_centers_[vmf_soft_mu_0_idx], vmf_soft_mu_0_error))
print('mu 0: {} (vmf-hard), error={}'.format(vmf_hard.cluster_centers_[vmf_hard_mu_0_idx], vmf_hard_mu_0_error))

print('---')
print('mu 1: {}'.format(mu_1))
print('mu 1: {} (kmeans), error={} ({})'.format(km.cluster_centers_[km_mu_1_idx], km_mu_1_error, km_mu_1_error_norm))
print('mu 1: {} (spherical kmeans), error={}'.format(skm.cluster_centers_[skm_mu_1_idx], skm_mu_1_error))
print('mu 1: {} (vmf-soft), error={}'.format(vmf_soft.cluster_centers_[vmf_soft_mu_1_idx], vmf_soft_mu_1_error))
print('mu 1: {} (vmf-hard), error={}'.format(vmf_hard.cluster_centers_[vmf_hard_mu_1_idx], vmf_hard_mu_1_error))


print('---')
print('true kappas {}'.format(kappas))
print('vmf-soft kappas {}'.format(vmf_soft.concentrations_[[vmf_soft_mu_0_idx, vmf_soft_mu_1_idx]]))
print('vmf-hard kappas {}'.format(vmf_hard.concentrations_[[vmf_hard_mu_0_idx, vmf_hard_mu_1_idx]]))

print('---')
print('vmf-soft weights {}'.format(vmf_soft.weights_[[vmf_soft_mu_0_idx, vmf_soft_mu_1_idx]]))
print('vmf-hard weights {}'.format(vmf_hard.weights_[[vmf_hard_mu_0_idx, vmf_hard_mu_1_idx]]))

print('---')
print("Homogeneity: %0.3f (k-means)" % metrics.homogeneity_score(labels, km.labels_))
print("Homogeneity: %0.3f (spherical k-means)" % metrics.homogeneity_score(labels, skm.labels_))
print("Homogeneity: %0.3f (vmf-soft)" % metrics.homogeneity_score(labels, vmf_soft.labels_))
print("Homogeneity: %0.3f (vmf-hard)" % metrics.homogeneity_score(labels, vmf_hard.labels_))

print('---')
print("Completeness: %0.3f (k-means)" % metrics.completeness_score(labels, km.labels_))
print("Completeness: %0.3f (spherical k-means)" % metrics.completeness_score(labels, skm.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, vmf_soft.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, vmf_hard.labels_))

print('---')
print("V-measure: %0.3f (k-means)" % metrics.v_measure_score(labels, km.labels_))
print("V-measure: %0.3f (spherical k-means)" % metrics.v_measure_score(labels, skm.labels_))
print("V-measure: %0.3f (vmf-soft)" % metrics.v_measure_score(labels, vmf_soft.labels_))
print("V-measure: %0.3f (vmf-hard)" % metrics.v_measure_score(labels, vmf_hard.labels_))

r_input()
