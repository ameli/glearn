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

import scipy.optimize
from ._minimize_terminator import MinimizeTerminator, MinimizeTerminated
from .._utilities.timer import Timer

__all__ = ['minimize']


# ========
# minimize
# ========

def minimize(
        fun,
        hyperparam_guess,
        method,
        tol,
        max_iter=1000,
        jac=None,
        hess=None,
        use_rel_error=True,
        verbose=False):
    """
    Minimizes a multivariate function.
    """

    # Minimize Terminator to gracefully terminate scipy.optimize.minimize once
    # tolerance is reached.
    minimize_terminator = MinimizeTerminator(tol, use_rel_error=use_rel_error,
                                             verbose=verbose)

    options = {
        'maxiter': max_iter,
        'xtol': tol,
        'xatol': tol,
        'fatol': tol,
        'disp': False
    }

    # Keeping elapsed times of optimization
    timer = Timer()
    timer.tic()

    try:
        # Local optimization method
        res = scipy.optimize.minimize(fun, hyperparam_guess, method=method,
                                      tol=tol, jac=jac, hess=hess,
                                      callback=minimize_terminator.__call__,
                                      options=options)

        # Extract results from Res output
        hyperparam = res.x
        max_fun = -res.fun
        num_opt_iter = res.nit
        message = res.message
        success = res.success

    except MinimizeTerminated:

        # Extract results from MinimizeTerminator
        hyperparam = minimize_terminator.hyperparams[-1, :]
        max_fun = -fun(hyperparam)
        num_opt_iter = minimize_terminator.counter
        message = 'Minimization algorithm is terminated successfully for ' + \
                  'reaching the tolerance %0.3e on all variables ' % tol + \
                  'after %d iterations' % num_opt_iter
        success = minimize_terminator.all_converged

    # Get convergence of hyperparam and its error
    # hyperparams = minimize_terminator.hyperparams
    # errors = minimize_terminator.errors
    # converged = minimize_terminator.converged

    # Adding time to the results
    timer.toc()

    result = {
        # 'convergence':
        # {
        #     'converged': converged,
        #     'errors': errors,
        #     'hyperparams': hyperparams,
        # },
        'optimization':
        {
            'state_vector': hyperparam,
            'max_fun': max_fun,
            'num_opt_iter': num_opt_iter,
            'message': message,
            'success': success
        },
        'time':
        {
            'wall_time': timer.wall_time,
            'proc_time': timer.proc_time
        }
    }

    return result
