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
import numpy
import scipy
from ._chandrupatla import chandrupatla

__all__ = ['root']


# ============
# find bracket
# ============

def _find_bracket(
        fun,
        bracket,
        num_bracket_trials,
        use_log,
        verbose=False):
    """
    Finds an interval ``[x0, x1]`` in which ``f(x0)`` and ``f(x1)`` have
    opposite signs. The interval is used for some root finding algorithms, such
    as Brent and Chandrupatla method. Finding such interval is known as
    *bracketing* the function.

    If the initial interval is not a suitable *bracket*, then it iterates
    *num_bracket_trials* times. If within the iterations, the bracket is yet
    not found, it returns False status.
    """

    # Initialization
    bracket_found = False

    # Interval bounds
    x0 = bracket[0]
    x1 = bracket[1]

    # Initial bracket
    f0 = fun(x0)
    f1 = fun(x1)

    # Trials
    iterations = 0
    while (not bracket_found) and (iterations < num_bracket_trials):
        iterations += 1

        if numpy.sign(f0) != numpy.sign(f1):

            # bracket was found
            bracket_found = True
            bracket = [x0, x1]
            bracket_value = [f0, f1]
            break

        else:

            # bracket was not found. Investigate the inner mid point
            t = 0.5
            x_new = x0*(1-t) + x1*t
            f_new = fun(x_new)

            # TODO
            if verbose:
                print('bracket was not found. Search for bracket. ' +
                      'Iteration: %d' % iterations)
                print('x0: %0.2f, f0: %0.16f' % (x0, f0))
                print('x1: %0.2f, f1: %0.16f' % (x1, f1))
                print('x_new: %0.2f, f_new: %0.16f' % (x_new, f_new))

            if numpy.sign(f0) != numpy.sign(f_new):

                # bracket was found
                bracket_found = True

                # Determine to choose [x0, x_inner] or [x_inner, x1] interval
                # based on whichever has smaller f
                if numpy.abs(f0) < numpy.abs(f1):
                    bracket = [x0, x_new]
                    bracket_value = [f0, f_new]
                else:
                    bracket = [x_new, x1]
                    bracket_value = [f_new, f1]
                break

            elif numpy.abs(f_new) < numpy.min([numpy.abs(f0), numpy.abs(f1)]):

                # if the new point has less f than both f0 and f1, keep
                # searching within the refined inner interval
                if numpy.abs(f0) < numpy.abs(f1):
                    # search mid-left inner interval in the next iteration
                    x1 = x_new
                    f1 = f_new
                else:
                    # search mid-right inner interval in the next iteration
                    x0 = x_new
                    f0 = f_new

                continue

            else:

                # bracket was not found yet. Try a point outside of interval
                if numpy.abs(f0) > numpy.abs(f1):
                    # Extend to the right side of interval
                    t = 0.5 + 1
                else:
                    # Extend to the left side of interval
                    t = 0.5 - 1
                x_new = x0*(1-t)+x1*t
                f_new = fun(x_new)

                print('!!!!!!!')
                print(t)
                print(f0)
                print(f_new)
                print(f1)

                if numpy.sign(f0) != numpy.sign(f_new):

                    # bracket was found
                    bracket_found = True
                    if t < 0:
                        bracket = [x_new, x0]
                        bracket_value = [f_new, f0]
                    else:
                        bracket = [x1, x_new]
                        bracket_value = [f1, f_new]
                    break

                else:
                    if t > 0:
                        # Search right side outer interval in next iteration
                        x0 = x1
                        f0 = f1
                        x1 = x_new
                        f1 = f_new
                    else:
                        # Search left side outer interval in next iteration
                        print('@@@@@@@@@@@@@')
                        x1 = x0
                        f1 = f0
                        x0 = x_new
                        f0 = f_new

                    continue

    # Exit with no success
    if not bracket_found:
        bracket = [x0, x1]
        bracket_value = [f0, f1]

    return bracket_found, bracket, bracket_value


# ==========
# find zeros
# ==========

def root(
        fun,
        x_guess,
        use_log=False,
        method='chandrupatla',
        tol=1e-6,
        max_iter=100,
        num_bracket_trials=4,
        verbose=False):
    """
    Finds roots of a function. If the Jacobian is given, it also checks if the
    root is maxima or minima.

    This function is used to find the maxima of posterior function by finding
    the zeros of the Jacobian of posterior. Hence, here, ``fun``: is the
    Jacobian of posterior. Also, The Jacobian ``jac`` of ``fun`` is the
    Hessian of the posterior.
    """

    # Keeping times
    initial_wall_time = time.time()
    initial_proc_time = time.process_time()

    # Ensure x_guess is a scalar
    if not numpy.isscalar(x_guess):
        if numpy.asarray(x_guess).size > 1:
            raise ValueError('"x_guess" should be a 1d array.')
        else:
            x_guess = x_guess[0]

    # Note: When using traceinv interpolation, make sure the interval below is
    # exactly the end points of eta_i, not less or more.
    window = 1.0
    threshold = 4.0
    if use_log:
        # x is in the log scale
        min_x_guess = numpy.max([-threshold, x_guess - window])
        max_x_guess = numpy.min([threshold, x_guess + window])
    else:
        # x is not in the log scale
        min_x_guess = numpy.max([10**(-threshold), x_guess * 10**(-window)])
        max_x_guess = numpy.min([10**threshold, x_guess * 10**(window)])

    # Interval to search for optimal value of x (eta or log of eta)
    interval = numpy.array([min_x_guess, max_x_guess], dtype=float)

    # Search for bracket (an interval with sign-change)
    bracket_found, bracket, bracket_values = _find_bracket(
            fun, interval, num_bracket_trials, use_log, verbose=verbose)

    # If bracket was not found, check if fun at zero has opposite sign
    if not bracket_found:

        # Function value at zero
        if use_log:
            zero = -numpy.inf
        else:
            zero = 0.0
        fun_zero = fun(zero)

        # Function value at the left side of bracket
        left_index = numpy.argmin(bracket)
        fun_left = bracket_values[left_index]

        if numpy.sign(fun_zero) * numpy.sign(fun_left):
            bracket_found
            bracket = [zero, bracket[left_index]]

    # Find root based on whether the bracket was found or not.
    if bracket_found:
        # There is a sign change in the interval of eta. Find root of ell
        # derivative

        if method == 'brentq':
            # Find roots using Brent method
            res = scipy.optimize.root_scalar(fun, bracket=bracket,
                                             method=method, xtol=tol)
            x = res.root
            num_opt_iter = res.iterations
            num_fun_eval = res.function_calls
            success = res.converged

        elif method == 'chandrupatla':
            # Find roots using Chandraputala method
            res = chandrupatla(fun, bracket, bracket_values, verbose=False,
                               eps_m=tol, eps_a=tol, maxiter=max_iter)
            x = res['root']
            num_opt_iter = res['num_opt_iter']
            num_fun_eval = res['num_fun_eval']
            success = res['converged']

        else:
            raise ValueError('"method" should be one of "chandrupatla" or ' +
                             '"brentq".')

    else:
        # bracket with sign change was not found.

        # No algorithm iteration. Also num_fun_eval will be found in later.
        num_opt_iter = 0
        num_fun_eval = None
        success = True

        if verbose:
            print('No bracket with sign change was found. Assume root at inf.')
            x = numpy.inf

    # Adding time to the results
    wall_time = time.time() - initial_wall_time
    proc_time = time.process_time() - initial_proc_time

    # Output dictionary
    result = {
        'config':
        {
            'method': method,
            'max_iter': max_iter,
            'num_bracket_trials': num_bracket_trials,
            'tol': tol,
        },
        'optimization':
        {
            'state_vector': x,
            'max_fun': 'not evaluated',
            'num_opt_iter': num_opt_iter,
            'num_fun_eval': None,
            'num_jac_eval': num_fun_eval,  # fun here is Jacobian of posterior
            'num_hes_eval': None,
            'message': '',
            'success': success
        },
        'time':
        {
            'wall_time': wall_time,
            'proc_time': proc_time
        }
    }

    return result
