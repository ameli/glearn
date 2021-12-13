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
from ._gaussian_process_utilities import plot_training_convergence, \
    print_training_summary, plot_prediction, print_prediction_summary
from .._utilities.memory import Memory
from .._utilities.timer import Timer


# ================
# gaussian process
# ================

class GaussianProcess(object):
    """
    Gaussian process for regression.

    :param X: Linear basis functions for the mean function. A 2D array of size
        ``(n, m)`` where ``n`` is the size of the data and ``m`` is the number
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

        # Store member data
        self.z = None
        self.posterior = None
        self.training_result = None
        self.prediction_result = None
        self.w = None
        self.Y = None
        self.Mz = None

        # Counting elapsed wall time and cpu proc time
        self.timer = Timer()

        # Record resident memory (rss) of this current process in bytes
        self.memory = Memory()

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
            # hyperparameters are sigma and sigma0. We assume all data is
            # noise, hence we set sigma to zero and solve sigma0 from
            # ordinary least square (OLS) solution.
            sigma_guess = 1e-2  # Small nonzero to avoid singularity
            sigma0_guess = self.posterior.likelihood.ols_solution()
            hyperparam_guess = numpy.r_[sigma_guess, sigma0_guess, scale_guess]

        elif profile_hyperparam == 'var':
            # Set scale before calling likelihood.asymptotic_maxima
            self.posterior.likelihood.cov.set_scale(scale_guess)

            # hyperparameter is eta. Use asymptotic relations to guess eta
            asym_degree = 2
            asym_maxima = self.posterior.likelihood.asymptotic_maxima(
                    degree=asym_degree)

            if asym_maxima != []:
                eta_guess = asym_maxima[0]
            else:
                # In case no asymptotic root was found (all negative, complex)
                eta_guess = 1.0

            hyperparam_guess = numpy.r_[eta_guess, scale_guess]

        elif profile_hyperparam == 'var_noise':
            # No hyperparameter
            hyperparam_guess = scale_guess

        return hyperparam_guess

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
            max_iter=1000,
            use_rel_error=True,
            verbose=False,
            plot=False):
        """
        Finds the hyperparameters of the Gaussian process model.

        Note: ``use_rel_error`` can be None, True, or False. When it is set to
        None, the callback function for minimize is not used.
        """

        # Create a posterior object. Note that self.posterior should be defined
        # before calling self._check_hyperparam_guess
        self.posterior = Posterior(self.mean, self.cov, z,
                                   profile_hyperparam=profile_hyperparam,
                                   log_hyperparam=log_hyperparam)

        # Reset function evaluation counters and timers
        self.posterior.reset()

        # Prepare or suggest hyperparameter guess
        if hyperparam_guess is not None:
            self._check_hyperparam_guess(hyperparam_guess, profile_hyperparam)
        else:
            hyperparam_guess = self._suggest_hyperparam_guess(
                    profile_hyperparam)

        # Maximize posterior w.r.t hyperparameters
        self.training_result = self.posterior.maximize_posterior(
                hyperparam_guess, optimization_method=optimization_method,
                tol=tol, max_iter=max_iter, use_rel_error=use_rel_error,
                verbose=verbose)

        if plot:
            plot_training_convergence(
                    self.posterior, self.training_result, verbose)

        if verbose:
            print_training_summary(self.training_result)

        # Set optimal parameters (sigma, sigma0) to covariance object
        sigma = self.training_result['hyperparam']['sigma']
        sigma0 = self.training_result['hyperparam']['sigma0']
        self.cov.set_sigmas(sigma, sigma0)

        # Set optimal parameters (b and B) to mean object
        self.mean.update_hyperparam(self.cov, z)

        # Store data for future reference
        self.z = z

        return self.training_result

    # ===============
    # plot likelihood
    # ===============

    def plot_likelihood(
            self,
            z=None,
            profile_hyperparam='var'):
        """
        Plots likelihood in multiple figures. This function may take a long
        time, and is only used for testing purposes on small datasets.
        """

        if z is None:
            if self.z is None:
                raise ValueError('Data "z" cannot be None.')
            z = self.z

        if self.training_result is None:

            # Train
            self.training_result = self.train(
                z, hyperparam_guess=None,
                profile_hyperparam=profile_hyperparam, log_hyperparam=True,
                optimization_method='Newton-CG', tol=1e-3, use_rel_error=True,
                verbose=False, plot=False)

            # Create a posterior object
            self.posterior = Posterior(self.mean, self.cov, z,
                                       profile_hyperparam=profile_hyperparam,
                                       log_hyperparam=True)

        # Plot likelihood
        self.posterior.likelihood.plot(self.training_result)

    # =======
    # predict
    # =======

    def predict(
            self,
            test_points,
            cov=False,
            plot=False,
            true_data=None,
            confidence_level=0.95,
            verbose=False):
        """
        Regression with Gaussian process on new data points.
        """

        if self.z is None:
            raise RuntimeError('Data should be trained first before calling ' +
                               'the predict function.')

        # If test points are 1d array, wrap them to a 2d array
        if test_points.ndim == 1:
            test_points = numpy.array([test_points], dtype=float).T

        if test_points.shape[1] != self.mean.points.shape[1]:
            raise ValueError('"test_points" should have the same dimension ' +
                             'as the training points.')

        # Record the used memory of the current process at this point in bytes
        if verbose:
            self.timer.reset()
            self.timer.tic()
            self.memory.reset()
            self.memory.start()

        # Design matrix on test points
        X_star = self.mean.generate_design_matrix(test_points)

        # Covariance on data points to test points
        cov_star = self.cov.cross_covariance(test_points)

        beta = self.mean.beta
        X = self.mean.X

        # w, Y, and Mz are computed once per data z and are independent of the
        # test points. On the future calls for the prediction on test points,
        # these will not be computed again.
        if (self.w is None) or (self.Y is None) or (self.Mz is None):

            # Solve Sinv * z and Sinv * X
            self.w = self.cov.solve(self.z)
            self.Y = self.cov.solve(X)

            # Compute Mz (Note: if b is zero, the following is actually Mz, but
            # if b is not zero, the following is Mz + C*Binv*b)
            self.Mz = self.w - numpy.matmul(self.Y, beta)

        # Posterior predictive mean. Note that the following uses the dual
        # formulation, that is, z_star at test point is just the dot product
        # of qualities (w, Mz) that are independent of the test point and they
        # were computed once.
        z_star_mean = cov_star.T.dot(self.Mz) + X_star.dot(beta)

        # Compute posterior predictive covariance
        z_star_cov = None
        if cov:

            # Compute R
            R = X_star.T - self.Y.T @ cov_star

            # Covariance on test points to test points
            cov_star_star = self.cov.auto_covariance(test_points)

            # Posterior covariance of beta
            C = self.mean.C

            if C is None:
                raise RuntimeError('Parameters of LinearModel are None. ' +
                                   'Call "train" function first.')

            # Covariance of data points to themselves
            Sinv_cov_star = self.cov.solve(cov_star)

            # Posterior predictive covariance
            z_star_cov = cov_star_star - cov_star.T @ Sinv_cov_star + \
                numpy.matmul(R.T, numpy.matmul(C, R))

        # Print summary
        if verbose:
            self.timer.toc()
            self.memory.stop()

            self.prediction_result = {
                'config': {
                    'num_training_points': self.z.size,
                    'num_test_points': test_points.shape[0],
                    'cov': cov
                },
                'process': {
                    'wall_time': self.timer.wall_time,
                    'proc_time': self.timer.proc_time,
                    'memory': [self.memory.mem, self.memory.unit]
                }
            }
            print_prediction_summary(self.prediction_result)

        # Plot prediction
        if plot:
            plot_prediction(self.mean.points, test_points, self.z, z_star_mean,
                            z_star_cov, confidence_level, true_data, verbose)

        if cov:
            return z_star_mean, z_star_cov
        else:
            return z_star_mean
