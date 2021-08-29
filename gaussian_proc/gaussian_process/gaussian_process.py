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

    def __init__(self, mean, cov):
        """
        Constructor.
        """

        self.mean = mean
        self.cov = cov
        self.likelihood = Likelihood(mean, cov)

    # =====
    # train
    # =====

    def train(
            self,
            z,
            likelihood_method='direct',
            optimization_method='Newton-CG',
            plot=False):
        """
        Finds the hyperparameters of the gaussian process model.
        """

        results = self.likelihood.maximize_log_likelihood(
                z, likelihood_method=likelihood_method,
                optimization_method=optimization_method, plot=plot)

        print(results)
