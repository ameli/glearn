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

    def __init__(self, tol, use_rel_error=True, verbose=False):
        """
        Initialization.
        """

        # Attributes
        self.use_rel_error = use_rel_error
        self.verbose = verbose

        # Member data
        self.counter = 0
        self.tol = tol
        self.hyperparam = None
        self.error = numpy.inf
        self.converged = False

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

            # Initialization
            self.hyperparam = current_hyperparam
            self.error = numpy.empty((self.hyperparam.size, ), dtype=float)
            self.error[:] = numpy.inf
            self.counter += 1

            if self.verbose:
                errors_string = ', '.join(('%+9f' % e) for e in
                                          self.error.tolist())
                print('iter: %03d, conv err: %s'
                      % (self.counter, errors_string))
        else:
            if not self.converged:

                # Using absolute error or relative error
                if self.use_rel_error:
                    # Using relative error
                    self.error = numpy.abs(
                        (current_hyperparam - self.hyperparam) /
                        self.hyperparam)
                else:
                    # Using absolute error
                    self.error = numpy.abs(
                            current_hyperparam - self.hyperparam)

                self.hyperparam = current_hyperparam
                self.counter += 1

                # Print convergence error for each of the variables.
                if self.verbose:
                    errors_string = ', '.join(('%+0.2e' % e) for e in
                                              self.error.tolist())
                    print('iter: %03d, conv err: %s'
                          % (self.counter, errors_string))

                if numpy.all(self.error < self.tol) and \
                        numpy.all(self.error > 0):
                    self.converged = True
                    raise MinimizeTerminated('Convergence error reached the ' +
                                             'tolerance after %d iterations.'
                                             % (self.counter))
