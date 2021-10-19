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
        verbose=False):
    """
    Minimizes a function.
    """

    # Minimize Terminator to gracefully terminate scipy.optimize.minimize once
    # tolerance is reached
    minimize_terminator = MinimizeTerminator(tol, verbose=verbose)

    options = {
        'maxiter': 1000,
        'xatol': tol,
        'fatol': tol,
        'disp': True
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
        iter = res.nit
        message = res.message
        success = res.success

    except MinimizeTerminated:

        # Extract results from MinimizeTerminator
        hyperparam = minimize_terminator.get_hyperparam()
        max_posterior = -func(hyperparam)
        iter = minimize_terminator.get_counter()
        message = 'Terminated after reaching the tolerance.'
        success = True

        if verbose:
            print('Minimization terminated after %d iterations.' % iter)

    # Adding time to the results
    wall_time = time.time() - initial_wall_time
    proc_time = time.process_time() - initial_proc_time

    result = {
        'hyperparam': hyperparam,
        'optimization':
        {
            'max_posterior': max_posterior,
            'iter': iter,
            'message': message,
            'success': success
        },
        'time':
        {
            'wall_time': wall_time,
            'proc_time': proc_time
        }
    }

    return result
