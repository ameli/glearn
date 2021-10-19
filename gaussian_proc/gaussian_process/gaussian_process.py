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

import numpy
from ._posterior import Posterior


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

    # =====
    # train
    # =====

    def train(
            self,
            z,
            hyperparam_guess=None,
            profile_hyperparam='var',
            log_hyperparam=True,
            optimization_method='Newton-CG',
            tol=1e-3,
            verbose=False,
            plot=False):
        """
        Finds the hyperparameters of the Gaussian process model.
        """

        # Find if scale is specifies or is None. If None, the training
        # process will find scale as an unknown hyperparameter. But, if scale
        # is given, it leaves it out of hyperparameters.
        scale = self.cov.get_scale()

        # Number of parameters of covariance function
        if profile_hyperparam == 'none':
            # hyperparameters are sigma and sigma0
            num_cov_hyperparam = 2
        elif profile_hyperparam == 'var':
            # hyperparameter is eta
            num_cov_hyperparam = 1
        elif profile_hyperparam == 'var_noise':
            num_cov_hyperparam = 0
        else:
            raise ValueError('"profile_hyperparam" can be one of "none", ' +
                             '"var", or "var_noise".')

        # Prepare hyperparameter guess
        if hyperparam_guess is None:

            # Set a default value for hyperparameter guess
            if scale is None:
                hyperparam_guess = [0.1, 0.1, 0.1, 0.1]
            else:
                hyperparam_guess = [0.1, 0.1]

        else:

            # Convert hyperparam to numpy array
            if isinstance(hyperparam_guess, list):
                hyperparam_guess = numpy.array(hyperparam_guess)

            # Number of hyperparameters
            if scale is None:
                # Finds sigma, sigma0, and all scale
                dimension = self.cov.mixed_cor.cor.points.shape[1]
                num_hyperparam = num_cov_hyperparam + dimension
            else:
                # Only find sigma and sigma0
                num_hyperparam = num_cov_hyperparam

            # check the size of input hyperparam_guess
            if hyperparam_guess.size != num_hyperparam:
                raise ValueError('The size of "hyperparam_guess" does not' +
                                 'match the number of hyperparameters.')

        # Create a posterior object
        posterior = Posterior(self.mean, self.cov, z,
                              profile_hyperparam=profile_hyperparam,
                              log_hyperparam=log_hyperparam)

        # Maximize posterior w.r.t hyperparameters
        result = posterior.maximize_posterior(
                hyperparam_guess=hyperparam_guess,
                optimization_method=optimization_method, tol=tol,
                verbose=verbose)

        if plot:
            posterior.plot(result)

        import pprint
        pprint.pprint(result)

    # =======
    # predict
    # =======

    def predict(self, z_star, dual=False):
        """
        Regression with Gaussian process on new data points.
        """

        pass
