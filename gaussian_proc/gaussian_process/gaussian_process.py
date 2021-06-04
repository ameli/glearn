# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

from .._likelihood import Likelihood


# ================
# gaussian process
# ================

class GaussianProcess(object):
    """
    Gaussian process for regression.

    :param X: Linear basis functions for the mean function. A 2D array of size
        ``(n, m)`` whre ``n`` is the size of the data and ``m`` is the number
        of the basis functions.
    :type X: numpy.ndarray

    :param K: Covariance matrix. A 2D array of size ``(n, n)`` where ``n`` is
        the size of the data.
    :type K: numpy.ndarray
    """

    # ====
    # init
    # ====

    def __init__(self, X, K, likelihood_method='direct'):
        """
        Constructor.
        """

        self.X = X
        self.K = K
        self.likelihood = Likelihood(X, K, likelihood_method=likelihood_method)

    # =====
    # train
    # =====

    def train(self, z, plot=False):
        """
        Finds the hyperparameters of the gaussian process model.
        """

        results = self.likelihood.maximize_log_likelihood(z, plot=plot)

        print(results)

    # ==========
    # likelihood
    # ==========

    # def log_likelihood(self, z, derivative=0):
    #     """
    #     Log likelihood function or its first and second derivative with
    #     respect to the hyperparameter.
    #     """
    #
    #     pass
