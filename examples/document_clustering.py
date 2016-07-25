from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

import numpy as np

import logging
from sklearn.cluster import KMeans

import sys
sys.path.append('../spherecluster')

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

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
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
print("Silhouette Coefficient (euclidean): %0.3f"
      % metrics.silhouette_score(X, skm.labels_, metric='euclidean'))
print("Silhouette Coefficient (cosine): %0.3f"
      % metrics.silhouette_score(X, skm.labels_, metric='cosine'))

print()

