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
from functools import partial
from ._root_finding import find_interval_with_sign_change, chandrupatla_method
from .._likelihood.likelihood import likelihood
from .._likelihood._profile_likelihood import ProfileLikelihood
from ._minimize import minimize
import warnings


# =========
# Posterior
# =========

class Posterior(object):
    """
    Posterior distribution.
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
        Initialization.
        """

        self.likelihood = likelihood(mean, cov, z, log_hyperparam,
                                     profile_hyperparam)

        if cov.cor.scale_prior is not None:
            self.prior = cov.cor.scale_prior

            if log_hyperparam:
                self.prior.use_log_scale = True
            else:
                self.prior.use_log_scale = False

        else:
            self.prior = None

        # Member data
        self.num_fun_eval = 0
        self.num_jac_eval = 0
        self.num_hes_eval = 0

    # =========
    # posterior
    # =========

    def _posterior(self, sign_switch, hyperparam):
        """
        Returns the log-posterior distribution for a given hyperparameter set.

        It is assumed that hyperparam is either of the forms:
        * [scale0, scale1, scale2, ...]
        * [eta, scale0, scale1, scale2, ...]
        * [sigma, sigma0, scale0, scale1, scale2, ...]

        The prior only accepts scale hyperparam ([scale0, scale1, ...]). The
        index at which ``scale`` starts in ``hyperparam`` is given by
        ``self.likelihood.scale_index``.
        """

        # Counter
        self.num_fun_eval += 1

        # Likelihood uses the full hyperparam, including scale.
        likelihood_ = self.likelihood.likelihood(sign_switch, hyperparam)
        posterior_ = likelihood_

        if self.prior is not None:

            # Extract scale from the hyperparam (depends on likelihood method)
            scale_index = self.likelihood.scale_index
            if hyperparam.size < scale_index:
                raise ValueError('"hyperparam" size should be larger than ' +
                                 '"scale_index" of the hyperparam of the ' +
                                 ' likelihood  method.')

            # Extract a position of hyperparam that is related to scale. Note
            # that hyperparam may or may not be in log form.
            hyperparam_scale = hyperparam[scale_index:]

            # Prior only uses the "scale" part of hyperparam.
            prior_ = self.prior.log_pdf(hyperparam_scale)

            if sign_switch:
                prior_ = -prior_

            # likelihood, prior, posterior here are all in log form.
            posterior_ += prior_

        return posterior_

    # ==================
    # posterior jacobian
    # ==================

    def _posterior_jacobian(self, sign_switch, hyperparam):
        """
        Returns the Jacobian of log-posterior distribution for a given
        hyperparameter set.

        It is assumed that hyperparam is either of the forms:
        * [scale0, scale1, scale2, ...]
        * [eta, scale0, scale1, scale2, ...]
        * [sigma, sigma0, scale0, scale1, scale2, ...]

        The prior only accepts scale hyperparam ([scale0, scale1, ...]). The
        index at which ``scale`` starts in ``hyperparam`` is given by
        ``self.likelihood.scale_index``.
        """

        # Counter
        self.num_jac_eval += 1

        # Likelihood uses the full hyperparam, including scale.
        likelihood_jacobian = self.likelihood.likelihood_jacobian(sign_switch,
                                                                  hyperparam)
        posterior_jacobian = likelihood_jacobian

        if self.prior is not None:

            # Extract scale from the hyperparam (depends on likelihood method)
            scale_index = self.likelihood.scale_index
            if hyperparam.size < scale_index:
                raise ValueError('"hyperparam" size should be larger than ' +
                                 '"scale_index" of the hyperparam of the ' +
                                 ' likelihood  method.')

            # Extract a position of hyperparam that is related to scale. Note
            # that hyperparam may or may not be in log form.
            hyperparam_scale = hyperparam[scale_index:]

            # Prior only uses the "scale" part of hyperparam.
            prior_jacobian = self.prior.log_pdf_jacobian(hyperparam_scale)

            if sign_switch:
                prior_jacobian = -prior_jacobian

            # Index [:scale_index] does not have any scale. On the other hand,
            # index [scale_index] has scale. Add log-prior Jacobian to them.
            posterior_jacobian[scale_index:] += prior_jacobian

        return posterior_jacobian

    # =================
    # posterior hessian
    # =================

    def _posterior_hessian(self, sign_switch, hyperparam):
        """
        Returns the Hessian of log-posterior distribution for a given
        hyperparameter set.

        It is assumed that hyperparam is either of the forms:
        * [scale0, scale1, scale2, ...]
        * [eta, scale0, scale1, scale2, ...]
        * [sigma, sigma0, scale0, scale1, scale2, ...]

        The prior only accepts scale hyperparam ([scale0, scale1, ...]). The
        index at which ``scale`` starts in ``hyperparam`` is given by
        ``self.likelihood.scale_index``.
        """

        # Counter
        self.num_hes_eval += 1

        # Likelihood uses the full hyperparam, including scale.
        likelihood_hessian = self.likelihood.likelihood_hessian(sign_switch,
                                                                hyperparam)
        posterior_hessian = likelihood_hessian

        if self.prior is not None:

            # Extract scale from the hyperparam (depends on likelihood method)
            scale_index = self.likelihood.scale_index
            if hyperparam.size < scale_index:
                raise ValueError('"hyperparam" size should be larger than ' +
                                 '"scale_index" of the hyperparam of the ' +
                                 ' likelihood  method.')

            # Extract a position of hyperparam that is related to scale. Note
            # that hyperparam may or may not be in log form.
            hyperparam_scale = hyperparam[scale_index:]

            # Prior only uses the "scale" part of hyperparam.
            prior_hessian = self.prior.log_pdf_hessian(hyperparam_scale)

            if sign_switch:
                prior_hessian = -prior_hessian

            # Index [:scale_index] does not have any scale. On the other hand,
            # index [scale_index] has scale. Add log-prior Hessian to them.
            posterior_hessian[scale_index:, scale_index:] += prior_hessian

        return posterior_hessian

    # ==================
    # maximize posterior
    # ==================

    def maximize_posterior(
            self,
            tol=1e-3,
            hyperparam_guess=[0.1, 0.1],
            log_hyperparam=True,
            optimization_method='Nelder-Mead',
            use_rel_error=True,
            verbose=False):
        """
        Maximizing the log-likelihood function over the space of parameters
        sigma and sigma0

        In this function, hyperparam = [sigma, sigma0].
        """

        # Reset attributes
        self.num_fun_eval = 0
        self.num_jac_eval = 0
        self.num_hes_eval = 0

        # Convert hyperparam to log of hyperparam (if enabled in configuration)
        hyperparam_guess = self.likelihood.hyperparam_to_log_hyperparam(
                hyperparam_guess)

        if optimization_method == 'chandrupatla':

            if not isinstance(self.likelihood, ProfileLikelihood):
                raise ValueError('"chandrupartla" method can only be ' +
                                 'applied to "ProfileLikelihood" class.')

            if (not numpy.isscalar(hyperparam_guess)) and \
                    (len(hyperparam_guess) > 1):

                warnings.warn('"chandrupatla" method does not optimize ' +
                              '"scale". The "distance scale in the given ' +
                              '"hyperparam_guess" will be ignored. To ' +
                              'optimize distance scale with "chandrupatla"' +
                              'method, set "profile_eta" to True.')

                # Extract eta and scale from hyperparam
                scale_index = self.likelihood.scale_index
                eta_guess = hyperparam_guess[:scale_index]
                scale_guess = hyperparam_guess[scale_index:]

                # Set scale in likelihood object
                if self.likelihood.mixed_cor.get_scale() is None:
                    self.likelihood.mixed_cor.set_scale(scale_guess)
                    warnings.warn('"scale" is set based on the guess value.')

            # Note: When using interpolation, make sure the interval below is
            # exactly the end points of eta_i, not less or more.
            min_eta_guess = numpy.min([1e-4, eta_guess * 1e-2])
            max_eta_guess = numpy.max([1e+3, eta_guess * 1e+2])
            interval_eta = [min_eta_guess, max_eta_guess]

            # Using root finding method on the first derivative w.r.t eta
            result = self._find_likelihood_der1_zeros(interval_eta)

            # Finding the maxima. This isn't necessary and affects run time
            result['optimization']['max_likelihood'] = \
                self.likelihood.likelihood(
                        False, result['hyperparam']['eta'])

            # The distance scale used in this method is the same as its guess.
            result['hyperparam']['scale'] = \
                self.likelihood.mixed_cor.get_scale()

        else:
            # Partial function of likelihood (with minus to make maximization
            # to a minimization).
            sign_switch = True
            posterior_partial_func = partial(self._posterior, sign_switch)

            # Partial function of Jacobian of likelihood (with minus sign)
            jacobian_partial_func = partial(self._posterior_jacobian,
                                            sign_switch)

            # Partial function of Hessian of likelihood (with minus sign)
            hessian_partial_func = partial(self._posterior_hessian,
                                           sign_switch)

            # Minimize
            res = minimize(posterior_partial_func, hyperparam_guess,
                           method=optimization_method, tol=tol,
                           use_rel_error=use_rel_error,
                           jac=jacobian_partial_func,
                           hess=hessian_partial_func, verbose=verbose)

            # convert back log-hyperparam to hyperparam
            sigma, sigma0, eta, scale = self.likelihood.extract_hyperparam(
                    res['optimization']['state_vector'])

            res['hyperparam']['sigma'] = sigma
            res['hyperparam']['sigma0'] = sigma0
            res['hyperparam']['eta'] = eta
            res['hyperparam']['scale'] = scale

            # Number of function, jacobian, and hessian evaluations
            res['optimization']['num_fun_eval'] = self.num_fun_eval
            res['optimization']['num_jac_eval'] = self.num_jac_eval
            res['optimization']['num_hes_eval'] = self.num_hes_eval

        return res

    # ==========================
    # find likelihood der1 zeros
    # ==========================

    def _find_likelihood_der1_zeros(
            self,
            interval_eta,
            tol=1e-6,
            max_iterations=100,
            num_bracket_trials=3):
        """
        root finding of the derivative of ell.

        The log likelihood function is implicitly a function of eta. We have
        substituted the value of optimal sigma, which itself is a function of
        eta.
        """

        # Find an interval that the function changes sign before finding its
        # root (known as bracketing the function)
        log_eta_start = numpy.log10(interval_eta[0])
        log_eta_end = numpy.log10(interval_eta[1])

        # Initial points
        bracket = [log_eta_start, log_eta_end]
        bracket_found, bracket, bracket_values = \
            find_interval_with_sign_change(self._likelihood_der1_eta, bracket,
                                           num_bracket_trials, args=(), )

        if bracket_found:
            # There is a sign change in the interval of eta. Find root of ell
            # derivative

            # Find roots using Brent method
            # method = 'brentq'
            # res = scipy.optimize.root_scalar(self._likelihood_der1_eta,
            #                                  bracket=bracket, method=method,
            #                                  xtol=tol)
            # print('Iter: %d, Eval: %d, Converged: %s'
            #         % (res.iterations, res.function_calls, res.converged))

            # Find roots using Chandraputala method
            res = chandrupatla_method(self._likelihood_der1_eta, bracket,
                                      bracket_values, verbose=False, eps_m=tol,
                                      eps_a=tol, maxiter=max_iterations)

            # Extract results
            # eta = self._hyperparam_to_eta(res.root)   # Use with Brent
            eta = self._hyperparam_to_eta(res['root'])  # Use with Chandrupatla
            sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)
            iter = res['iterations']

            # Check second derivative
            # success = True
            # d2ell_deta2 = self._likelihood_der2_eta(eta)
            # if d2ell_deta2 < 0:
            #     success = True
            # else:
            #     success = False

        else:
            # bracket with sign change was not found.
            iter = 0

            # Evaluate the function in intervals
            eta_left = bracket[0]
            eta_right = bracket[1]
            dell_deta_left = bracket_values[0]
            dell_deta_right = bracket_values[1]

            # Second derivative of log likelihood at eta = zero, using either
            # of the two methods below:
            eta_zero = 0.0
            # method 1: directly from analytical equation
            d2ell_deta2_zero_eta = self._likelihood_der2_eta(eta_zero)

            # method 2: using forward differencing from first derivative
            # dell_deta_zero_eta = self._likelihood_der1_eta(
            #         numpy.log10(eta_zero))
            # d2ell_deta2_zero_eta = \
            #         (dell_deta_lowest_eta - dell_deta_zero_eta) / eta_lowest

            # print('dL/deta   at eta = 0.0:\t %0.2f'%dell_deta_zero_eta)
            print('dL/deta   at eta = %0.2e:\t %0.2f'
                  % (eta_left, dell_deta_left))
            print('dL/deta   at eta = %0.2e:\t %0.16f'
                  % (eta_right, dell_deta_right))
            print('d2L/deta2 at eta = 0.0:\t %0.2f'
                  % d2ell_deta2_zero_eta)

            # No sign change. Can not find a root
            if (dell_deta_left > 0) and (dell_deta_right > 0):
                if d2ell_deta2_zero_eta > 0:
                    eta = 0.0
                else:
                    eta = numpy.inf

            elif (dell_deta_left < 0) and (dell_deta_right < 0):
                if d2ell_deta2_zero_eta < 0:
                    eta = 0.0
                else:
                    eta = numpy.inf

            # Check eta
            if not (eta == 0 or numpy.isinf(eta)):
                raise ValueError('eta must be zero or inf at this point.')

            # Find sigma and sigma0
            sigma, sigma0 = self._find_optimal_sigma_sigma0(eta)

        # Output dictionary
        result = {
            'hyperparam':
            {
                'sigma': sigma,
                'sigma0': sigma0,
                'eta': eta,
                'scale': None
            },
            'optimization':
            {
                'max_likelihood': None,
                'iter': iter
            }
        }

        return result
