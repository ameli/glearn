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

_all__ = ['MinimizeTerminator', 'MinimizeTerminated']


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
    This class interrupts ``scipy.optimize.minimize`` function to terminate
    the optimization algorithm iterations if the termination criteria is
    reached.

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

    To disable the effect of this class and leave the control of algorithm
    iterations back to ``scipy.optimize.minimize`` set ``terminate`` to
    False.
    """

    # ====
    # init
    # ====

    def __init__(
            self,
            tol,
            use_rel_error=True,
            terminate=False,
            verbose=False):
        """
        Initialization.
        """

        # Attributes
        self.use_rel_error = use_rel_error
        self.terminate = terminate
        self.verbose = verbose

        # Member data
        self.counter = 0
        self.tol = tol
        self.hyperparams = None
        self.errors = None
        self.converged = None
        self.all_converged = False

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

        delimiter = ' ' * 3

        if self.hyperparams is None:

            # Initialization
            self.converged = numpy.empty((current_hyperparam.size, ),
                                         dtype=bool)
            self.converged[:] = False
            self.hyperparams = numpy.empty((1, current_hyperparam.size),
                                           dtype=float)
            self.hyperparams[0, :] = current_hyperparam
            self.errors = numpy.empty((1, current_hyperparam.size),
                                      dtype=float)
            self.errors[0, :] = numpy.inf
            self.counter += 1

            if self.verbose:
                errors_string = delimiter.join(('%8f' % e) for e in
                                               self.errors[-1, :].tolist())
                line = '%03d%s%s' % (self.counter, delimiter, errors_string)
                header = '=' * len(line)
                title = 'Convergence'
                left_space = ' ' * ((len(header) - len(title)) // 2)
                right_space = ' ' * \
                    (len(header) - len(title) - len(left_space))
                print('')
                print(left_space + title + right_space)
                print(header)
                subtitle = 'itr'
                subheader = '---'
                for i in range(self.hyperparams.size):
                    subtitle += delimiter + 'param %2d' % (i+1)
                    subheader += delimiter + '--------'
                print(subtitle)
                print(subheader)
                print(line)
        else:
            if not self.all_converged:

                # Using absolute error or relative error
                if self.use_rel_error:
                    # Using relative error
                    error = numpy.abs(
                            (current_hyperparam - self.hyperparams[-1, :]) /
                            self.hyperparams[-1, :])
                else:
                    # Using absolute error
                    error = numpy.abs(
                            current_hyperparam - self.hyperparams[-1, :])

                # Insert 1d current hyperparam to last row of self.hyperparams
                self.hyperparams = numpy.insert(
                        self.hyperparams, self.hyperparams.shape[0],
                        current_hyperparam, axis=0)

                # Insert error 1d array after last row of 2d self.errors array
                self.errors = numpy.insert(self.errors, self.errors.shape[0],
                                           error, axis=0)

                self.counter += 1

                # Print convergence error for each of the variables.
                if self.verbose:
                    errors_string = delimiter.join(('%0.2e' % e) for e in
                                                   self.errors[-1, :].tolist())
                    print('%03d%s%s'
                          % (self.counter, delimiter,  errors_string))

                self.converged[:] = self.errors[-1, :] < self.tol
                self.all_converged = numpy.all(self.converged)

                # Terminate when all converged
                if self.all_converged and self.terminate:
                    raise MinimizeTerminated('Convergence error reached the ' +
                                             'tolerance after %d iterations.'
                                             % (self.counter))
