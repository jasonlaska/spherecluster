from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

import numpy as np
from tabulate import tabulate

import logging
from sklearn.cluster import KMeans

from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture

# modified from
# http://scikit-learn.org/stable/auto_examples/text/document_clustering.html


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

###############################################################################
# Optional params
use_LSA = False
n_components = 500

###############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a sparse vectorizer")
vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
X = vectorizer.fit_transform(dataset.data)

print("n_samples: %d, n_features: %d" % X.shape)
print()

# table for results display
table = []


###############################################################################
# LSA for dimensionality reduction (and finding dense vectors)
if use_LSA:
  print("Performing dimensionality reduction using LSA")
  svd = TruncatedSVD(n_components)
  normalizer = Normalizer(copy=False)
  lsa = make_pipeline(svd, normalizer)
  X = lsa.fit_transform(X)

  explained_variance = svd.explained_variance_ratio_.sum()
  print("Explained variance of the SVD step: {}%".format(
      int(explained_variance * 100)))

  print()


###############################################################################
# K-Means clustering
km = KMeans(n_clusters=true_k, init='k-means++', n_init=20)

print("Clustering with %s" % km)
km.fit(X)
print()

table.append([
    'k-means',
    metrics.homogeneity_score(labels, km.labels_),
    metrics.completeness_score(labels, km.labels_),
    metrics.v_measure_score(labels, km.labels_),
    metrics.adjusted_rand_score(labels, km.labels_),
    metrics.adjusted_mutual_info_score(labels, km.labels_),
    metrics.silhouette_score(X, km.labels_, metric='cosine')])

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Adjusted Mututal Information: %.3f"
      % metrics.adjusted_mutual_info_score(labels, km.labels_))
print("Silhouette Coefficient (euclidean): %0.3f"
      % metrics.silhouette_score(X, km.labels_, metric='euclidean'))
print("Silhouette Coefficient (cosine): %0.3f"
      % metrics.silhouette_score(X, km.labels_, metric='cosine'))

print()



###############################################################################
# Spherical K-Means clustering
skm = SphericalKMeans(n_clusters=true_k, init='k-means++', n_init=20)

print("Clustering with %s" % skm)
skm.fit(X)
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, skm.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, skm.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, skm.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, skm.labels_))
print("Adjusted Mututal Information: %.3f"
      % metrics.adjusted_mutual_info_score(labels, skm.labels_))
print("Silhouette Coefficient (euclidean): %0.3f"
      % metrics.silhouette_score(X, skm.labels_, metric='euclidean'))
print("Silhouette Coefficient (cosine): %0.3f"
      % metrics.silhouette_score(X, skm.labels_, metric='cosine'))

print()

table.append([
    'spherical k-means',
    metrics.homogeneity_score(labels, skm.labels_),
    metrics.completeness_score(labels, skm.labels_),
    metrics.v_measure_score(labels, skm.labels_),
    metrics.adjusted_rand_score(labels, skm.labels_),
    metrics.adjusted_mutual_info_score(labels, skm.labels_),
    metrics.silhouette_score(X, skm.labels_, metric='cosine')])


###############################################################################
# Mixture of von Mises Fisher clustering (soft)
vmf_soft = VonMisesFisherMixture(n_clusters=true_k, posterior_type='soft',
    init='random-class', n_init=20, force_weights=np.ones((true_k,))/true_k)

print("Clustering with %s" % vmf_soft)
vmf_soft.fit(X)
print()
print('weights: {}'.format(vmf_soft.weights_))
print('concentrations: {}'.format(vmf_soft.concentrations_))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, vmf_soft.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, vmf_soft.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, vmf_soft.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, vmf_soft.labels_))
print("Adjusted Mututal Information: %.3f"
      % metrics.adjusted_mutual_info_score(labels, vmf_soft.labels_))
print("Silhouette Coefficient (euclidean): %0.3f"
      % metrics.silhouette_score(X, vmf_soft.labels_, metric='euclidean'))
print("Silhouette Coefficient (cosine): %0.3f"
      % metrics.silhouette_score(X, vmf_soft.labels_, metric='cosine'))

print()

table.append([
    'movMF-soft',
    metrics.homogeneity_score(labels, vmf_soft.labels_),
    metrics.completeness_score(labels, vmf_soft.labels_),
    metrics.v_measure_score(labels, vmf_soft.labels_),
    metrics.adjusted_rand_score(labels, vmf_soft.labels_),
    metrics.adjusted_mutual_info_score(labels, vmf_soft.labels_),
    metrics.silhouette_score(X, vmf_soft.labels_, metric='cosine')])


###############################################################################
# Mixture of von Mises Fisher clustering (hard)
vmf_hard = VonMisesFisherMixture(n_clusters=true_k, posterior_type='hard',
    init='spherical-k-means', n_init=20, force_weights=np.ones((true_k,))/true_k)

print("Clustering with %s" % vmf_hard)
vmf_hard.fit(X)
print()
print('weights: {}'.format(vmf_hard.weights_))
print('concentrations: {}'.format(vmf_hard.concentrations_))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, vmf_hard.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, vmf_hard.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, vmf_hard.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, vmf_hard.labels_))
print("Adjusted Mututal Information: %.3f"
      % metrics.adjusted_mutual_info_score(labels, vmf_hard.labels_))
print("Silhouette Coefficient (euclidean): %0.3f"
      % metrics.silhouette_score(X, vmf_hard.labels_, metric='euclidean'))
print("Silhouette Coefficient (cosine): %0.3f"
      % metrics.silhouette_score(X, vmf_hard.labels_, metric='cosine'))

print()

table.append([
    'movMF-hard',
    metrics.homogeneity_score(labels, vmf_hard.labels_),
    metrics.completeness_score(labels, vmf_hard.labels_),
    metrics.v_measure_score(labels, vmf_hard.labels_),
    metrics.adjusted_rand_score(labels, vmf_hard.labels_),
    metrics.adjusted_mutual_info_score(labels, vmf_hard.labels_),
    metrics.silhouette_score(X, vmf_hard.labels_, metric='cosine')])


###############################################################################
# Print all results in table
headers = [
    'Homogeneity',
    'Completeness',
    'V-Measure',
    'Adj Rand',
    'Adj MI',
    'Silhouette (cos)']
print(tabulate(table, headers, tablefmt="fancy_grid"))

