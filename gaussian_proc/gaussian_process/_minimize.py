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
import time
import scipy.optimize


# ===================
# Minimize Terminated
# ===================

class MinimizeTerminated(Exception):
    """
    This class is a python exception class to raise when the MinimizeTerminator
    is terminated. In a try-exception clause, this class is caught.
    """

    def __init__(self, *args, **kwargs):
        super(MinimizeTerminated, self).__init__(*args)


# ===================
# Minimize Terminator
# ===================

class MinimizeTerminator(object):
    """
    The scipy.optimize.minimize does not terminate when setting its tolerances
    with ``tol``, ``xatol``, and ``fatol``. Rather, its algorithm runs over all
    iterations till ``maxiter`` is reached, which passes way below the
    specified tolerance.

    To fix this issue, I tried to use its callback function to manually
    terminate the algorithm. If the callback function returns True, according
    to documentation, it should terminate the algorithm. However, it seems (in
    a GitHub issue thread) that this feature is not implemented, i.e., the
    callback is useless.

    To fix the latter issue, this class is written. It stores iterations,
    ``self.counter``, as member data. Its ``__call__()`` function is passed to
    the callback of ``scipy.optimize.minimize``. It updates the hyperparam
    ``xk`` in ``self.xk``, and compares it to the previous stored hyperparam
    to calculate the error, ``self.error``. If all the entries of the
    ``self.error`` vector are below the tolerance, it raises an exception. The
    exception causes the algorithm to terminate. To prevent the exception from
    terminating the whole script, the algorithm should be inside a try/except
    clause to catch the exception and terminate it gracefully.

    Often, the algorithm passes ``xk`` the same as previous hyperparam, which
    than makes the self.error to be absolute zero. To ignore these false
    errors, we check if self.error > 0 to leave the false errors out.
    """

    # ====
    # init
    # ====

    def __init__(self, tol, verbose):
        """
        Initialization.
        """

        # Member data
        self.counter = 0
        self.tol = tol
        self.hyperparam = None
        self.error = numpy.inf
        self.converged = False
        self.verbose = verbose

    # ===========
    # get counter
    # ===========

    def get_counter(self):
        """
        Returns the counter.
        """

        return self.counter

    # ================
    # get state vector
    # ================

    def get_hyperparam(self):
        """
        Returns the state vector.
        """

        return self.hyperparam

    # ====
    # call
    # ====

    def __call__(self, current_hyperparam, *args, **kwargs):
        """
        Overwriting the ``__call__`` function.
        """

        if self.hyperparam is None:
            self.hyperparam = current_hyperparam
            self.counter += 1
        else:
            if not self.converged:

                # Using absolute error
                # self.error = numpy.abs(
                #         current_hyperparam - self.hyperparam)

                # or using relative error
                self.error = numpy.abs(
                    (current_hyperparam - self.hyperparam) /
                    self.hyperparam)
                self.hyperparam = current_hyperparam
                self.counter += 1

                if self.verbose:
                    print('Convergence error: %s'
                          % (', '.join(str(e) for e in self.error.tolist())))

                if numpy.all(self.error < self.tol) and \
                        numpy.all(self.error > 0):
                    self.converged = True
                    raise MinimizeTerminated('Convergence error reached the ' +
                                             'tolerance at %d iterations.'
                                             % (self.counter))


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
            'iter': res.nit,
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
