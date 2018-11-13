import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import seaborn  # NOQA

from spherecluster import sample_vMF

plt.ion()

n_clusters = 3
mus = np.random.randn(3, n_clusters)
mus, r = np.linalg.qr(mus, mode='reduced')

kappas = [15, 15, 15]
num_points_per_class = 250

Xs = []
for nn in range(n_clusters):
    new_X = sample_vMF(mus[nn], kappas[nn], num_points_per_class)
    Xs.append(new_X.T)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(
    1, 1, 1, aspect='equal', projection='3d',
    adjustable='box-forced', xlim=[-1.1, 1.1], ylim=[-1.1, 1.1],
    zlim=[-1.1, 1.1]
)

colors = ['b', 'r', 'g']
for nn in range(n_clusters):
    ax.scatter(Xs[nn][0, :], Xs[nn][1, :], Xs[nn][2, :], c=colors[nn])

ax.set_aspect('equal')
plt.axis('off')
plt.show()

def r_input(val=None):
    val = val or ''
    if sys.version_info[0] >= 3:
        return eval(input(val))

    return raw_input(val)

r_input()
