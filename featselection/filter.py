#!/usr/bin/env python
# coding=utf-8

from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

__author__ = 'Rogério C. P. Fragoso'
__credits__ = ['Rogério C. P. Fragoso', 'Roberto H. W. Pinheiro', 'George D. C. Cavalavanti']
__version__ = '0.1.dev'
__maintainer__ = 'Rogério C. P. Fragoso'
__email__ = 'rcpf@cin.ufpe.br'


class Filter(ABC):

    def __init__(self, f, fef):
        """
        :param f: number of features to be selected per document
        :param fef: feature evaluation function
        """

        self.f = f
        self.fef = fef
        self.top_features = None

    @abstractmethod
    def fit(self, data, target):
        """
        :param data: {array-like, sparse matrix}, shape = (n_samples, n_features_in) Sample vectors.
        :param target: array-like, shape = (n_samples,) Target vector (class labels).
        """

        pass

    def fit_transform(self, data, target):
        """
        :param data:
        :param target:
        :return:
        """

        self.fit(data, target)
        return self.transform(data)

    def transform(self, data):
        return data[:, self.top_features]


class MFD(Filter):

    def fit(self, data, target):
        scores = self.fef(data, target)
        feat = pd.Series(list(scores[1]))
        sorted_features = feat.sort_values().index
        top = set()
        for x in data:
            if issparse(data):
                sorted_x = sorted_features[np.in1d(sorted_features, x.indices)]
            else:
                sorted_x = sorted_features[np.in1d(sorted_features, np.where(x > 0)[0])]
            top.update(sorted_x[0: self.f])
        self.top_features = list(top)


class MFDR(Filter):

    def fit(self, data, target):
        scores = self.fef(data, target)
        feat = pd.Series(list(scores[1]))
        sorted_features = feat.sort_values().index
        top = set()
        idx_docs = compute_dr(data)
        for doc in data[idx_docs, :]:
            if issparse(data):
                sorted_x = sorted_features[np.in1d(sorted_features, doc.indices)]
            else:
                sorted_x = sorted_features[np.in1d(sorted_features, np.where(doc > 0)[0])]
            top.update(sorted_x[0: self.f])
        self.top_features = list(top)


class CMFDR(Filter):

    def fit(self, data, target):
        labels = list(set(target))
        scores = self.fef(data, target)
        feat = pd.Series(list(scores[1]))
        sorted_features = feat.sort_values().index
        top = set()
        idx_docs = []
        for label in labels:
            idx_docs.extend(compute_dr(data[target == label]))
        for doc in data[idx_docs, :]:
            if issparse(data):
                sorted_x = sorted_features[np.in1d(sorted_features, doc.indices)]
            else:
                sorted_x = sorted_features[np.in1d(sorted_features, np.where(doc > 0)[0])]
            top.update(sorted_x[0: self.f])
        self.top_features = list(top)


class AFSA(Filter):
    def __init__(self, n, fef, base_filter, classifier, evaluation_metric):
        super().__init__(n, fef)
        self.base_filter_class = base_filter
        self.classifier = classifier
        self.evaluation_metric = evaluation_metric
        self.top_features = None

    def fit(self, train_data, train_target):
        performances = []
        base_filters = []

        _, val_data, _, val_target = train_test_split(train_data, train_target, test_size=0.1)
        for f in list(range(1, self.f + 1)):
            b_filter = self.base_filter_class(f, self.fef)
            data = b_filter.fit_transform(train_data, train_target)
            data_v = b_filter.transform(val_data)
            classifier = copy(self.classifier)
            classifier.fit(data, train_target)
            predicted = classifier.predict(data_v)

            base_filters.append(b_filter)
            performances.append(self.evaluation_metric(predicted, val_target))
        base_filter = base_filters[np.argmax(performances)]
        self.top_features = base_filter.top_features


def compute_dr(data):
    arr_dr = []
    for doc in data:
        arr_dr.append(sum(doc.data if issparse(data) else doc[doc > 0]))
    return np.where(arr_dr > np.mean(arr_dr))[0]