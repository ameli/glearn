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
from ..priors.prior import Prior
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

    # ======================
    # check hyperparam guess
    # ======================

    def _check_hyperparam_guess(self, hyperparam_guess, profile_hyperparam):
        """
        Checks the input hyperparam, if not None.
        """

        # Find scale if not specifically given (as number, or array) the
        # training process will find scale as an unknown hyperparameter. But,
        # if scale is given, it leaves it out of hyperparameters.
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

        # Convert hyperparam to numpy array
        if isinstance(hyperparam_guess, list):
            hyperparam_guess = numpy.array(hyperparam_guess)

        # Check number of hyperparameters
        if not isinstance(scale, (int, float, numpy.ndarray, list)):
            # Finds sigma, sigma0 (or eta), and all scale
            dimension = self.cov.mixed_cor.cor.points.shape[1]
            num_hyperparam = num_cov_hyperparam + dimension
        else:
            # Only find sigma and sigma0 (or eta)
            num_hyperparam = num_cov_hyperparam

        # check the size of input hyperparam_guess
        if hyperparam_guess.size != num_hyperparam:
            raise ValueError(
                'The size of "hyperparam_guess" (which is %d'
                % hyperparam_guess.size + ') does not match the number ' +
                'of hyperparameters (which is %d).' % num_hyperparam)

    # ========================
    # suggest hyperparam guess
    # ========================

    def _suggest_hyperparam_guess(self, profile_hyperparam):
        """
        Suggests hyperparam_guess when it is None. ``hyperparam_guess`` may
        contain the following variables:

        * ``scale``: suggested from the mean, median, or peak of prior
          distributions for the scale hyperparam.
        * ``eta``: it uses the asymptotic relation that estimates eta before
          any computation is performed.
        * ``sigma`` and ``sigma0``: it assumes sigma is zero, and finds sigma0
          based on eta=infinity assumption.
        """

        # Find scale if not specifically given (as number, or array) the
        # training process will find scale as an unknown hyperparameter. But,
        # if scale is given, it leaves it out of hyperparameters.
        scale = self.cov.get_scale()

        # Set a default value for hyperparameter guess
        if isinstance(scale, (int, float, numpy.ndarray, list)):
            # Scale is given explicitly. No hyperparam is needed.
            scale_guess = []
        elif scale is None:

            # Get the prior of scale
            scale_prior = self.cov.cor.scale_prior

            if not isinstance(scale_prior, Prior):
                raise TypeError('"scale" should be given either explicitly ' +
                                'or as a prior distribution.')

            # Get the guess from the prior
            scale_guess = scale_prior.suggest_hyperparam_guess()

            # Check type of scale guess
            if numpy.isscalar(scale_guess):
                scale_guess = numpy.array([scale_guess], ftype=float)
            elif isinstance(scale_guess, list):
                scale_guess = numpy.array(scale_guess, ftype=float)
            elif not isinstance(scale_guess, numpy.ndarray):
                raise TypeError('"scale_guess" should be a numpy array.')

            # Check if the size of scale guess matches the dimension
            dimension = self.cov.mixed_cor.cor.points.shape[1]
            if scale_guess.size != dimension:
                if scale_guess.size == 1:
                    scale_guess = numpy.tile(scale_guess, dimension)
                else:
                    raise ValueError('Size of "scale_guess" and "dimension" ' +
                                     'does not match.')

        # Other hyperparameters of covariance (except scale)
        if profile_hyperparam == 'none':
            # hyperparameters are sigma and sigma0
            sigma_guess = 0.1  # TODO
            sigma0_guess = 0.1  # TODO
            hyperparam_guess = numpy.r_[sigma_guess, sigma0_guess, scale_guess]

        elif profile_hyperparam == 'var':
            # hyperparameter is eta
            eta_guess = 1.0  # TODO
            hyperparam_guess = numpy.r_[eta_guess, scale_guess]

        elif profile_hyperparam == 'var_noise':
            # No hyperparameter
            hyperparam_guess = scale_guess

        return hyperparam_guess

    # ===============
    # plot likelihood
    # ===============

    def plot_likelihood(self):
        """
        Plots likelihood in multiple figures. This function may take a long
        time, and is only used for testing purposes on small datasets.
        """



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
            use_rel_error=True,
            verbose=False,
            plot=False):
        """
        Finds the hyperparameters of the Gaussian process model.
        """

        # Prepare or suggest hyperparameter guess
        if hyperparam_guess is not None:
            self._check_hyperparam_guess(hyperparam_guess, profile_hyperparam)
        else:
            hyperparam_guess = self._suggest_hyperparam_guess(
                    profile_hyperparam)

        # Create a posterior object
        posterior = Posterior(self.mean, self.cov, z,
                              profile_hyperparam=profile_hyperparam,
                              log_hyperparam=log_hyperparam)

        # Maximize posterior w.r.t hyperparameters
        result = posterior.maximize_posterior(
                hyperparam_guess=hyperparam_guess,
                optimization_method=optimization_method, tol=tol,
                use_rel_error=use_rel_error, verbose=verbose)

        if plot:
            posterior.plot_convergence(result)

        if verbose:
            import pprint
            pprint.pprint(result)

        # Set optimal parameters (sigma, sigma0) to covariance object
        sigma = result['hyperparam']['sigma']
        sigma0 = result['hyperparam']['sigma0']
        self.cov.set_sigmas(sigma, sigma0)

        # Set optimal parameters (b and B) to mean object
        self.mean.update_hyperparam(self.cov, z)

    # =======
    # predict
    # =======

    def predict(self, z_star, dual=False):
        """
        Regression with Gaussian process on new data points.
        """

        pass
