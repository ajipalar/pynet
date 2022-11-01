"""
Functionality to train and test probabilistic models.

Provides:
    - metrics (Accuracy, precision)
    - training data (real and synthetic)
   

Set up training data

Get an inference_procedure

get the results

report the inference_procedure and parameters


"""

import numpy as np
import scipy as sp
import sklearn.metrics
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Callable
from functools import partial

import pyext.src.core as core


class ExampleTrainingCase(core.KeyValueReport):
    def __init__(
        self,
        d=2,
        shape=None,
        bern_p=0.5,
        rseed=13,
        truth_seed=22,
        truth_p=0.1,
        nsamples=1000,
    ):

        core.KeyValueReport.__init__(self)

        if shape:
            self.shape = shape
        else:
            self.shape = (d, d)

        self.bern_p = bern_p
        self.rseed = rseed
        self.truth_seed = truth_seed
        self.truth_p = truth_p
        self.key = jax.random.PRNGKey(rseed)
        self.mutable_key = jax.random.PRNGKey(rseed)
        self.truth_key = jax.random.PRNGKey(truth_seed)

        self.ground_truth = jax.random.bernoulli(
            self.truth_key, truth_p, shape=self.shape
        )
        self.samples = []
        self.metrics = {}
        self.nsamples = nsamples

    def step(self, k):
        return np.array(jax.random.bernoulli(k, p=self.bern_p, shape=self.shape))

    # Getters

    def get_median_accuracy(self):
        return self.metrics["median_accuracy"]

    def get_median_precision(self):
        return self.metrics["median_precision"]

    def get_variance_accuracy(self):
        return self.metrics["variance_accuracy"]

    def get_variance_precision(self):
        return self.metrics["variance_precision"]

    # Updaters

    def update_sample(self):
        for i in range(self.nsamples):
            key, k1 = jax.random.split(self.mutable_key)
            self.mutable_key = key
            self.samples.append(self.step(k1))

    def update_metrics(self):
        assert len(self.samples) == self.nsamples
        accuracies = []
        precisions = []

        for i in range(self.nsamples):

            y_ref = np.ravel(self.ground_truth)
            y_pred = np.ravel(self.samples[i])
            accuracies.append(sklearn.metrics.accuracy_score(y_ref, y_pred))
            precisions.append(sklearn.metrics.precision_score(y_ref, y_pred))

            self.metrics["accuracies"] = accuracies
            self.metrics["precisions"] = precisions

            self.metrics["median_accuracy"] = np.median(accuracies)
            self.metrics["variance_accuracy"] = np.var(accuracies)

            self.metrics["median_precision"] = np.median(precisions)
            self.metrics["variance_precision"] = np.median(precisions)

    def update_report(self):
        attributes = [
            "shape",
            "bern_p",
            "rseed",
            "truth_seed",
            "truth_p",
            "nsamples",
            "key",
            "mutable_key",
            "truth_key",
        ]

        keys = [
            "median_accuracy",
            "variance_accuracy",
            "median_precision",
            "variance_precision",
        ]
        report = {attr: self.__dict__[attr] for attr in attributes} | {
            key: self.metrics[key] for key in keys
        }
        self.update_key_value_report_content(report)
        self.update_key_value_report_str()

    def show_report(self):
        self.show_key_value_report_str()


class CullinE3LigaseTrain:

    self.metadata = {
        "paper_url": "https://www.sciencedirect.com/science/article/pii/S1931312819302537?via%3Dihub",
        "paper_title": "ARIH2 Is a Vif-Dependent Regulator of CUL5-Mediated APOBEC3G Degradation in HIV Infection",
        "authors": [
            "Ruth Hüttenhain",
            "Jiewei Xu",
            "Lily A.Burton",
            "David E.Gordon",
            "Judd F.Hultquist",
            "Jeffrey R.Johnson",
            "Laura Satkamp",
            "Joseph Hiatt",
            "David Y.Rhee",
            "Kheewoong Baek",
            "David C.Crosby",
            "Alan D.Frankel",
            "Alexander Marson",
            "Wade Harper",
            "Arno F.Alpi",
            "Brenda A.Schulman",
            "John D.Gross",
            "Nevan J.Krogan",
        ],
    }
    self.training_data = {}
    self.protein_types = ["APOBEC3G", "CBFBeta", "ViF", "ELOC", "ELOB", "CUL5", "ARIH2", "RBX2", "NEDD8"]

    def __init__(self):
        ...
