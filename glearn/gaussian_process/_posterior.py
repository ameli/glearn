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
from .._utilities.device import get_num_cpu_threads, get_num_gpu_devices
from .._utilities.memory import Memory
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

        self.profile_hyperparam = profile_hyperparam
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

        # Record resident memory (rss) of this current process in bytes
        self.memory = Memory()

    # =====
    # reset
    # =====

    def reset(self):
        """
        Resets the number of function and matrix evaluations and all timers.
        """

        # Reset number of function evaluations
        self.num_fun_eval = 0
        self.num_jac_eval = 0
        self.num_hes_eval = 0

        # Reset num evaluations and timers
        self.likelihood.cov.cor.num_cor_eval = 0
        self.likelihood.cov.cor.timer.reset()
        self.likelihood.cov.mixed_cor.logdet_timer.reset()
        self.likelihood.cov.mixed_cor.traceinv_timer.reset()
        self.likelihood.cov.mixed_cor.solve_timer.reset()
        self.likelihood.timer.reset()
        self.memory.reset()

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
            hyperparam_guess,
            tol=1e-3,
            max_iter=1000,
            max_bracket_trials=6,
            log_hyperparam=True,
            optimization_method='Nelder-Mead',
            use_rel_error=True,
            verbose=False):
        """
        Maximizing the log-likelihood function over the space of parameters
        sigma and sigma0

        In this function, hyperparam = [sigma, sigma0].
        """

        # Record the used memory of the current process at this point in bytes
        if verbose:
            self.memory.start()

        # Convert hyperparam to log of hyperparam. Note that if use_log_scale,
        # use_log_eta, or use_log_sigmas are not True, the output is not
        # converted to log, despite we named the output with "log_" prefix.
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
                       use_log=self.likelihood.use_log_eta, tol=tol,
                       max_iter=max_iter,
                       max_bracket_trials=max_bracket_trials, verbose=verbose)
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
                           max_iter=max_iter, use_rel_error=use_rel_error,
                           jac=jacobian_partial_func,
                           hess=hessian_partial_func, verbose=verbose)

            # convert back log-hyperparam to hyperparam
            sigma, sigma0, eta, scale = self.likelihood.extract_hyperparam(
                    res['optimization']['state_vector'])

        # Find the memory used only during the training process
        if verbose:
            self.memory.stop()

        # Create output dictionary
        res = self._create_output_dict(res, sigma, sigma0, eta, scale,
                                       optimization_method, max_iter,
                                       max_bracket_trials, use_rel_error, tol,
                                       verbose)

        return res

    # ==================
    # create output dict
    # ==================

    def _create_output_dict(self, res, sigma, sigma0, eta, scale,
                            optimization_method, max_iter, max_bracket_trials,
                            use_rel_error, tol, verbose):
        """
        Creates a dictionary with a full information about the training
        process.
        """

        # Append optimal hyperparameter to the result dictionary
        hyperparam = {
            'sigma': sigma,
            'sigma0': sigma0,
            'eta': eta,
            'scale': scale
        }

        # Number of function, Jacobian, and Hessian evaluations
        optimization = {
            'num_fun_eval': self.num_fun_eval,
            'num_jac_eval': self.num_jac_eval,
            'num_hes_eval': self.num_hes_eval,
            'num_cor_eval': self.likelihood.cov.cor.num_cor_eval,
            'max_fun': res['optimization']['max_fun'],
            'num_opt_iter': res['optimization']['num_opt_iter'],
            'message': res['optimization']['message']
        }

        # Correlation times
        time = {
            # correlation timer
            'cor_wall_time': self.likelihood.cov.cor.timer.wall_time,
            'cor_proc_time': self.likelihood.cov.cor.timer.proc_time,

            # mixed_cor logdet timer
            'det_wall_time':
                self.likelihood.cov.mixed_cor.logdet_timer.wall_time,
            'det_proc_time':
                self.likelihood.cov.mixed_cor.logdet_timer.proc_time,

            # mixed_cor traceinv timer
            'trc_wall_time':
                self.likelihood.cov.mixed_cor.traceinv_timer.wall_time,
            'trc_proc_time':
                self.likelihood.cov.mixed_cor.traceinv_timer.proc_time,

            # mixed_cor solve timer
            'sol_wall_time':
                self.likelihood.cov.mixed_cor.solve_timer.wall_time,
            'sol_proc_time':
                self.likelihood.cov.mixed_cor.solve_timer.proc_time,

            # Likelihood timer
            'lik_wall_time': self.likelihood.timer.wall_time,
            'lik_proc_time': self.likelihood.timer.proc_time,

            # Optimization timer
            'opt_wall_time': res['time']['wall_time'],
            'opt_proc_time': res['time']['proc_time']
        }

        # Configuration
        config = {
            'profile_hyperparam': self.profile_hyperparam,
            'optimization_method': optimization_method,
            'max_iter': max_iter,
            'max_bracket_trials': max_bracket_trials,
            'use_rel_error': use_rel_error,
            'tol': tol,
        }

        # imate configuration
        imate_config = {
            'imate_method': self.likelihood.cov.imate_method,
            'imate_tol': self.likelihood.cov.tol,
            'imate_interpolate': self.likelihood.cov.mixed_cor.interpolate,
        }
        if self.likelihood.cov.mixed_cor.imate_info != {}:
            imate_info = self.likelihood.cov.mixed_cor.imate_info
            imate_config['min_num_samples'] = imate_info['min_num_samples']
            imate_config['max_num_samples'] = imate_info['max_num_samples']
            imate_config['device'] = imate_info['device']
            imate_config['solver'] = imate_info['solver']

        # Device information. Device is queried only if verbose is enabled,
        # since gpu inquiries can be time consuming.
        if verbose:
            if self.likelihood.cov.mixed_cor.imate_info != {}:
                imate_info = self.likelihood.cov.mixed_cor.imate_info
                num_cpu_threads = imate_info['device']['num_cpu_threads']
                num_gpu_devices = imate_info['device']['num_gpu_devices']
                num_gpu_multiproc = \
                    imate_info['device']['num_gpu_multiprocessors']
                num_gpu_threads_per_multiproc = \
                    imate_info['device']['num_gpu_threads_per_multiprocessor']
            else:
                num_cpu_threads = get_num_cpu_threads()
                num_gpu_devices = get_num_gpu_devices()
                num_gpu_multiproc = 0
                num_gpu_threads_per_multiproc = 0

            device = {
                'num_cpu_threads': num_cpu_threads,
                'num_gpu_devices': num_gpu_devices,
                'num_gpu_multiproc': num_gpu_multiproc,
                'num_gpu_threads_per_multiproc': num_gpu_threads_per_multiproc,
                'memory_usage': [self.memory.mem, self.memory.unit]
            }
        else:
            device = {}

        # Create output dictionary
        res = {
            'hyperparam': hyperparam,
            'optimization': optimization,
            'config': config,
            'imate_config': imate_config,
            'time': time,
            'device': device
        }

        return res
