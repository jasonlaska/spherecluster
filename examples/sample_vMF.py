'''
Generate multivariate von Mises Fisher samples via a
rejection sampling scheme from
    "Directional Statistics" (Mardia and Jupp, 1999)

This solution originally appears here:
http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python


References:

Sampling from vMF on S^2:
    https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    http://www.stat.pitt.edu/sungkyu/software/randvonMisesFisher3.pdf
'''
import numpy as np


def weight_rejection_sampling(kappa, dim, num_samples):
    dim = dim - 1 # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa**2 + dim**2) + 2 * kappa)
    x = (1 - b) / (1 + b)
    c = kappa * x + dim * np.log(1 - x**2)

    samples = []
    for nn in range(num_samples):
        while True:
            z = np.random.beta(dim / 2., dim / 2.)
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1 - x * w) - c >= np.log(u):
                samples.append(w)
                break

    return samples


def sample_tangent_unit(mu):
    mat = np.matrix(mu)

    if mat.shape[1] > mat.shape[0]:
        mat = mat.T

    U, _, _ = np.linalg.svd(mat)
    nu = np.matrix(np.random.randn(mat.shape[0])).T
    x = np.dot(U[:, 1:], nu[1:, :])
    return np.squeeze(np.array(x / np.linalg.norm(x)))


def vMF(mu, kappa, num_samples):
    dim = len(mu)

    # sample some weights
    ws = weight_rejection_sampling(kappa, dim, num_samples)

    result = np.zeros((num_samples, dim))
    for nn in range(num_samples):
        # draw sample v uniformly over the unit hyper sphere
        # and orthogonal to mu
        v = sample_tangent_unit(mu)

        # compute new point
        result[nn, :] = v * np.sqrt(1 - ws[nn]**2) + ws[nn] * mu

    return result
