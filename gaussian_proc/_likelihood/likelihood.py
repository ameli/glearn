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

from ._full_likelihood import FullLikelihood
from ._profile_likelihood import ProfileLikelihood
from ._double_profile_likelihood import DoubleProfileLikelihood
import numpy


# ==========
# Likelihood
# ==========

class Likelihood(object):
    """
    """

    # ====
    # init
    # ====

    def __init__(
            self,
            mean,
            cov,
            z,
            profile_hyperparam='var',
            log_hyperparam=True):
        """
        """

        # Input attributes
        self.mean = mean
        self.cov = cov
        self.z = z
        self.profile_hyperparam = profile_hyperparam

        # Member data
        self.X = self.mean.X
        self.mixed_cor = self.cov.mixed_cor

        # Set likelihood method depending on the type of profile.
        if self.profile_hyperparam == 'none':
            self.likelihood_method = FullLikelihood(self.z, self.X, self.cov,
                                                    log_hyperparam)
        elif self.profile_hyperparam == 'var':
            self.likelihood_method = \
                ProfileLikelihood(self.z, self.X, self.cov, log_hyperparam)
        elif self.profile_hyperparam == 'var_noise':
            self.likelihood_method_ = \
                DoubleProfileLikelihood(self.z, self.X, self.cov,
                                        log_hyperparam)
        else:
            raise ValueError('"profile_hyperparam" can be one of "none", ' +
                             '"var", or "var_noise".')

        # Attributes of self.likelihood_method
        self.scale_index = self.likelihood_method.scale_index

    # ============================
    # hyperparam to log hyperparam
    # ============================

    def hyperparam_to_log_hyperparam(self, hyperparam):
        """
        Converts the input hyperparameters to their log10, if this is enabled
        by ``self.use_scale``.

        If is assumed that the input hyperparam is not in log scale, and it
        containts either of the following form:

        * [sigma, sigma0]
        * [sigma, sigma0, scale1, scale2, ...]
        """

        return self.likelihood_method.hyperparam_to_log_hyperparam(hyperparam)

    # ==================
    # extract hyperparam
    # ==================

    def extract_hyperparam(self, hyperparam):
        """
        It is assumed the input hyperparam might be in the log10 scale, and
        may or may not contain scales. The output will be converted to non-log
        format and will include scale, regardless if the input has scale or
        not.
        """

        return self.likelihood_method.extract_hyperparam(hyperparam)

    # ==========
    # likelihood
    # ==========

    def likelihood(self, sign_switch, hyperparam):
        """
        Returns the log-likelihood function.
        """

        return self.likelihood_method.likelihood(sign_switch, hyperparam)

    # ===================
    # likelihood jacobian
    # ===================

    def likelihood_jacobian(self, sign_switch, hyperparam):
        """
        Returns the Jacobian of log-likelihood function.
        """

        return self.likelihood_method.likelihood_jacobian(sign_switch,
                                                          hyperparam)

    # ==================
    # likelihood hessian
    # ==================

    def likelihood_hessian(self, sign_switch, hyperparam):
        """
        Returns the Hessian of log-likelihood function.
        """

        return self.likelihood_method.likelihood_hessian(sign_switch,
                                                         hyperparam)

    # ====
    # plot
    # ====

    def plot(self, result):
        """
        Plot likelihood function and its derivatives.
        """

        if self.profile_hyperparam == 'none':

            # Plot likelihood for scale, fixed sigma and sigma0
            self.likelihood_method.plot_likelihood_versus_scale(
                    result, other_sigmas=numpy.logspace(-1, 1, 3))

            # Plot likelihood for sigma, fixed sigma0 and scale
            self.likelihood_method.plot_likelihood_versus_sigma(
                    result, other_scales=numpy.logspace(-1, 1, 3))

            # Plot likelihood for sigma0, fixed sigma and scale
            self.likelihood_method.plot_likelihood_versus_sigma0(
                    result, other_scales=numpy.logspace(-1, 1, 3))

            # 2d plot of likelihood versus sigma0 and sigma
            self.likelihood_method.plot_likelihood_versus_sigma0_sigma(result)

        elif self.profile_hyperparam == 'var':

            # Plot log-lp versus eta
            self.likelihood_method.plot_likelihood_versus_eta(
                    result, numpy.logspace(-2, 2, 5))

            # Plot log-lp versus scale
            self.likelihood_method.plot_likelihood_versus_scale(
                    result, numpy.logspace(-2, 2, 5))

            # 3D Plot of log-lp function
            self.likelihood_method.plot_likelihood_versus_eta_scale(result)

            # Plot first derivative of log likelihood
            self.likelihood_method.plot_likelihood_der1_eta(result)

        elif self.profile_hyperparam == 'var_noise':

            # Plot log-lp
            self.likelihood_method.plot_likelihood_versus_scale(result)
