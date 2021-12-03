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
from .._likelihood.likelihood import likelihood
from .._likelihood._profile_likelihood import ProfileLikelihood
from .._optimize import minimize, root
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

        # Convert hyperparam to log of hyperparam. Note that if use_log_scale,
        # use_log_eta, or use_log_sigmas are not True, the output is not
        # converted to log, despite we named the outout with "log_" prefix.
        log_hyperparam_guess = self.likelihood.hyperparam_to_log_hyperparam(
                hyperparam_guess)

        if optimization_method in ['chandrupatla', 'brentq']:

            if not isinstance(self.likelihood, ProfileLikelihood):
                raise ValueError('"%s" method can ' % optimization_method +
                                 'only be used when variance profiling is ' +
                                 'enabled. Set "profile_hyperparam" to "var".')

            scale_index = self.likelihood.scale_index
            if (not numpy.isscalar(hyperparam_guess)) and \
                    (len(hyperparam_guess) > 1):

                warnings.warn('"%s" method does not ' % optimization_method +
                              'optimize "scale". The "distance scale in ' +
                              'the given "hyperparam_guess" will be ignored.' +
                              ' To optimize scale with ' +
                              '"%s"' % optimization_method + 'method, set ' +
                              '"profile_hyperparam" to "var".')

                # Extract log_eta and scale from hyperparam. log_eta is either
                # log of eta (if self.likelihood.use_log_eta is True), or eta
                # itself, despite for both cases we named it by "log_" prefix.
                scale_guess = hyperparam_guess[scale_index:]

                # Set scale in likelihood object
                if self.likelihood.mixed_cor.get_scale() is None:
                    self.likelihood.mixed_cor.set_scale(scale_guess)
                    warnings.warn('"scale" is set based on the guess value.')

            # Since root-finding methods do not optimize the scale parameter,
            # the scale prior should not be used in the posterior. Here we
            # overwrite the prior to None
            self.prior = None

            # Partial function of posterior
            sign_switch = False
            posterior_partial_func = partial(self._posterior, sign_switch)

            # Partial function of Jacobian of posterior (with minus sign)
            jacobian_partial_func = partial(self._posterior_jacobian,
                                            sign_switch)

            # Partial function of Hessian of posterior (with minus sign)
            hessian_partial_func = partial(self._posterior_hessian,
                                           sign_switch)

            # Find zeros of the Jacobian (input fun is Jacobian, and the
            # Jacobian of the input is the Hessian).
            log_eta_guess = log_hyperparam_guess[:scale_index]
            res = root(jacobian_partial_func, log_eta_guess,
                       use_log=self.likelihood.use_log_eta, verbose=verbose)
            x = res['optimization']['state_vector']

            # Check second derivative is positive, which the root does not
            # becomes maxima. Don't check if x is inf due to singularity.
            if not numpy.isinf(x):
                hessian = hessian_partial_func(x)
                if hessian > 0:
                    res['optimization']['message'] = 'Root is not a maxima.'
                    res['optimization']['status'] = False

            # Find sigma and sigma0, eta, and scale
            eta = self.likelihood._hyperparam_to_eta(x)
            sigma, sigma0 = self.likelihood._find_optimal_sigma_sigma0(x)
            scale = self.likelihood.mixed_cor.get_scale()

            # Finding the maxima (not necessary). Only evaluated when verbose.
            if verbose:
                max_fun = posterior_partial_func(x)
            else:
                max_fun = 'not evaluated'
            res['optimization']['max_fun'] = max_fun

        else:
            # Partial function of posterior (with minus to turn maximization
            # to a minimization problem).
            sign_switch = True
            posterior_partial_func = partial(self._posterior, sign_switch)

            # Partial function of Jacobian of posterior (with minus sign)
            jacobian_partial_func = partial(self._posterior_jacobian,
                                            sign_switch)

            # Partial function of Hessian of posterior (with minus sign)
            hessian_partial_func = partial(self._posterior_hessian,
                                           sign_switch)

            # Minimize
            res = minimize(posterior_partial_func, log_hyperparam_guess,
                           method=optimization_method, tol=tol,
                           use_rel_error=use_rel_error,
                           jac=jacobian_partial_func,
                           hess=hessian_partial_func, verbose=verbose)

            # convert back log-hyperparam to hyperparam
            sigma, sigma0, eta, scale = self.likelihood.extract_hyperparam(
                    res['optimization']['state_vector'])

        # Append optimal hyperparameter to the result dictionary
        hyperparam = {
            'sigma': sigma,
            'sigma0': sigma0,
            'eta': eta,
            'scale': scale
        }

        res['hyperparam'] = hyperparam

        # Modify number of function, jacobian, and hessian evaluations
        res['optimization']['num_fun_eval'] = self.num_fun_eval
        res['optimization']['num_jac_eval'] = self.num_jac_eval
        res['optimization']['num_hes_eval'] = self.num_hes_eval

        return res
