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

from .._mixed_correlation import MixedCorrelation
from ._direct_likelihood import DirectLikelihood
from ._profile_likelihood import ProfileLikelihood


# ==========
# Likelihood
# ==========

class Likelihood(object):
    """
    """

    # ====
    # init
    # ====

    def __init__(self, X, K, likelihood_method='direct'):
        """
        """

        self.X = X
        self.K = K
        self.likelihood_method = likelihood_method

        # Mixed correlation object
        interpolate = False
        imate_method = 'eigenvalue'
        # imate_method = 'hutchinson'
        imate_options = {}
        # imate_options = {
        #     'min_num_samples': 100
        # }
        self.K_mixed = MixedCorrelation(self.K, interpolate=interpolate,
                                        imate_method=imate_method,
                                        imate_options=imate_options)

    # ==========
    # likelihood
    # ==========

    def likelihood(self, z, hyperparam):
        """
        """

        sign_switch = False
        return DirectLikelihood.log_likelihood(z, self.X, self.K_mixed,
                                               sign_switch, hyperparam)

    # =======================
    # maximize log likelihood
    # =======================

    def maximize_log_likelihood(self, z, plot=False):
        """
        """

        if self.likelihood_method == 'direct':
            results = DirectLikelihood.maximize_log_likelihood(z, self.X,
                                                               self.K_mixed)

            # Plot log likelihood
            if plot:
                optimal_sigma = results['sigma']
                optimal_sigma0 = results['sigma0']
                optimal_hyperparam = [optimal_sigma, optimal_sigma0]
                DirectLikelihood.plot_log_likelihood(z, self.X, self.K_mixed,
                                                     optimal_hyperparam)

            # return ProfileLikelihood.maximize_log_likelihood_with_sigma_eta(
            #         z, self.X, self.K_mixed)

        elif self.likelihood_method == 'profiled':

            # Note: When using interpolation, make sure the interval below is
            # exactly the end points of eta_i, not less or more.
            interval_eta = [1e-4, 1e+3]

            # Find hyperparam
            results = ProfileLikelihood.find_log_likelihood_der1_zeros(
                    z, self.X, self.K_mixed, interval_eta)

            # Plot first derivative of log likelihood
            if plot:
                optimal_eta = results['eta']
                ProfileLikelihood.plot_log_likelihood_der1_eta(
                        z, self.X, self.K, self.K_mixed, optimal_eta)

        return results
