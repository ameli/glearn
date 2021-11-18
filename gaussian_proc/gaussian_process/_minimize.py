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

import time
import scipy.optimize
from ._minimize_terminator import MinimizeTerminator, MinimizeTerminated


# ========
# minimize
# ========

def minimize(
        func,
        hyperparam_guess,
        method,
        tol,
        jac=None,
        hess=None,
        use_rel_error=True,
        verbose=False):
    """
    Minimizes a function.
    """

    # Minimize Terminator to gracefully terminate scipy.optimize.minimize once
    # tolerance is reached
    minimize_terminator = MinimizeTerminator(tol, use_rel_error=use_rel_error,
                                             verbose=verbose)

    options = {
        'maxiter': 1000,
        'xatol': tol,
        'fatol': tol,
        'disp': False
    }

    # Keeping times
    initial_wall_time = time.time()
    initial_proc_time = time.process_time()

    try:
        # Local optimization method
        res = scipy.optimize.minimize(func, hyperparam_guess, method=method,
                                      tol=tol, jac=jac, hess=hess,
                                      callback=minimize_terminator.__call__,
                                      options=options)

        # Extract results from Res output
        hyperparam = res.x
        max_posterior = -res.fun
        num_opt_iter = res.nit
        num_fun_eval = res.nfev
        num_jac_eval = res.njev
        num_hes_eval = res.nhev
        message = res.message
        success = res.success

    except MinimizeTerminated:

        # Extract results from MinimizeTerminator
        hyperparam = minimize_terminator.hyperparams[-1, :]
        max_posterior = -func(hyperparam)
        num_opt_iter = minimize_terminator.counter
        num_fun_eval = None
        num_jac_eval = None
        num_hes_eval = None
        message = 'Minimization algorithm is terminated successfully for ' + \
                  'reaching the tolerance %+0.5e on all variables ' % tol + \
                  'after %d iterations of the algorithm.' % num_opt_iter
        success = minimize_terminator.all_converged

    # Get convergence of hyperparam and its error
    hyperparams = minimize_terminator.hyperparams
    errors = minimize_terminator.errors
    converged = minimize_terminator.converged

    # Adding time to the results
    wall_time = time.time() - initial_wall_time
    proc_time = time.process_time() - initial_proc_time

    result = {
        'hyperparam':
        {
            'sigma': None,
            'sigma0': None,
            'eta': None,
            'scale': None
        },
        'config':
        {
            'method': method,
            'max_iter': options['maxiter'],
            'tol': tol,
            'use_rel_error': use_rel_error,
        },
        'optimization':
        {
            'state_vector': hyperparam,
            'max_posterior': max_posterior,
            'num_opt_iter': num_opt_iter,
            'num_fun_eval': num_fun_eval,
            'num_jac_eval': num_jac_eval,
            'num_hes_eval': num_hes_eval,
            'message': message,
            'success': success
        },
        'convergence':
        {
            'converged': converged,
            'errors': errors,
            'hyperparams': hyperparams,
        },
        'time':
        {
            'wall_time': wall_time,
            'proc_time': proc_time
        }
    }

    return result
