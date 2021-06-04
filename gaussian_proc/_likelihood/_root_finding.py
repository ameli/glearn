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


# ==============================
# find interval with sign change
# ==============================

def find_interval_with_sign_change(f, bracket, num_bracket_trials, args=(), ):
    """
    Finds an interval ``[x0, x1]`` in which ``f(x0)`` and ``f(x1)`` have
    opposite signs. The interval is used for some root finding algorithms, such
    as Brent and Chandrupatla method. Finding such interval is known as
    *bracketing* the function.

    If the initial interval is not a sitable *bracket*, then it iterates
    *num_bracket_trials* times. If within the iterations the bracket is yet not
    found, it exits with false status.
    """

    # Initialization
    bracket_found = False

    # Interval bounds
    x0 = bracket[0]
    x1 = bracket[1]

    # Initial bracket
    f0 = f(x0, *args)
    f1 = f(x1, *args)

    # Trials
    iterations = 0
    while (not bracket_found) and (iterations < num_bracket_trials):
        iterations += 1

        if numpy.sign(f0) != numpy.sign(f1):

            # bracket wad found
            bracket_found = True
            bracket = [x0, x1]
            bracket_value = [f0, f1]
            break

        else:
            print('bracket was not found. Search for bracket. Iteration: %d'
                  % iterations)

            # Test
            print('x0: %0.2f, f0: %0.16f' % (x0, f0))
            print('x1: %0.2f, f1: %0.16f' % (x1, f1))

            # bracket was not found. Investigate the inner mid point
            t = 0.5
            x_new = x0*(1-t) + x1*t
            f_new = f(x_new, *args)

            # Test
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
                f_new = f(x_new, *args)

                if numpy.sign(f0) != numpy.sign(f_new):

                    # bracket wad found
                    bracket_found = True
                    if numpy.abs(f0) > numpy.abs(f1):
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


# ===================
# chandrupatla method
# ===================

def chandrupatla_method(f, bracket, bracket_values, verbose=False, eps_m=None,
                        eps_a=None, maxiter=50, args=(), ):
    """
    This function is obtained from:
    https://github.com/scipy/scipy/issues/7242#issuecomment-290548427
    More to read at: https://www.embeddedrelated.com/showarticle/855.php
    """

    x0 = bracket[0]
    x1 = bracket[1]

    # Initialization
    b = x0
    a = x1

    # Evaluate function on the intervals
    if bracket_values is None:
        fa = f(a, *args)
        fb = f(b, *args)
    else:
        fa = bracket_values[1]
        fb = bracket_values[0]

    # Make sure we know the size of the result
    shape = numpy.shape(fa)
    assert shape == numpy.shape(fb)

    # In case x0, x1 are scalars, make sure we broadcast them to become the
    # size of the result
    b += numpy.zeros(shape)
    a += numpy.zeros(shape)

    fc = fa
    c = a

    # Make sure we are bracketing a root in each case
    assert (numpy.sign(fa) * numpy.sign(fb) <= 0).all()
    t = 0.5
    # Initialize an array of False,
    # determines whether we should do inverse quadratic interpolation
    iqi = numpy.zeros(shape, dtype=bool)

    # jms: some guesses for default values of the eps_m and eps_a settings
    # based on machine precision... not sure exactly what to do here
    eps = numpy.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2*eps

    iterations = 0
    terminate = False

    while maxiter > 0:
        maxiter -= 1
        # use t to linearly interpolate between a and b,
        # and evaluate this function as our newest estimate xt
        xt = a + t*(b-a)
        ft = f(xt, *args)
        if verbose:
            output = 'IQI? %s\nt=%s\nxt=%s\nft=%s\na=%s\nb=%s\nc=%s' \
                    % (iqi, t, xt, ft, a, b, c)
            if verbose is True:
                print(output)
            else:
                print(output, file=verbose)
        # update our history of the last few points so that
        # - a is the newest estimate (we're going to update it from xt)
        # - c and b get the preceding two estimates
        # - a and b maintain opposite signs for f(a) and f(b)
        samesign = numpy.sign(ft) == numpy.sign(fa)
        c = numpy.choose(samesign, [b, a])
        b = numpy.choose(samesign, [a, b])
        fc = numpy.choose(samesign, [fb, fa])
        fb = numpy.choose(samesign, [fa, fb])
        a = xt
        fa = ft

        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        fa_is_smaller = numpy.abs(fa) < numpy.abs(fb)
        xm = numpy.choose(fa_is_smaller, [b, a])
        fm = numpy.choose(fa_is_smaller, [fb, fa])

        """
        the preceding lines are a vectorized version of:

        samesign = numpy.sign(ft) == numpy.sign(fa)
        if samesign
            c = a
            fc = fa
        else:
            c = b
            b = a
            fc = fb
            fb = fa

        a = xt
        fa = ft
        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        if numpy.abs(fa) < numpy.abs(fb):
            xm = a
            fm = fa
        else:
            xm = b
            fm = fb
        """

        tol = 2*eps_m*numpy.abs(xm) + eps_a
        tlim = tol/numpy.abs(b-c)
        terminate = numpy.logical_or(terminate,
                                     numpy.logical_or(fm == 0, tlim > 0.5))
        if verbose:
            output = "fm=%s\ntlim=%s\nterm=%s" % (fm, tlim, terminate)
            if verbose is True:
                print(output)
            else:
                print(output, file=verbose)

        if numpy.all(terminate):
            break
        iterations += 1-terminate

        # Figure out values xi and phi
        # to determine which method we should use next
        xi = (a-b)/(c-b)
        phi = (fa-fb)/(fc-fb)
        iqi = numpy.logical_and(phi**2 < xi, (1-phi)**2 < 1-xi)

        if not shape:
            # scalar case
            if iqi:
                # inverse quadratic interpolation
                t = fa / (fb-fa) * fc / (fb-fc) + \
                        (c-a)/(b-a)*fa/(fc-fa)*fb/(fc-fb)
            else:
                # bisection
                t = 0.5
        else:
            # array case
            t = numpy.full(shape, 0.5)
            a2, b2, c2, fa2, fb2, fc2 = a[iqi], b[iqi], c[iqi], fa[iqi], \
                fb[iqi], fc[iqi]
            t[iqi] = fa2 / (fb2-fa2) * fc2 / (fb2-fc2) + \
                (c2-a2)/(b2-a2)*fa2/(fc2-fa2)*fb2/(fc2-fb2)

        # limit to the range (tlim, 1-tlim)
        t = numpy.minimum(1-tlim, numpy.maximum(tlim, t))

    # Results
    results = {
        'root': xm,
        'iterations': iterations,
    }

    return results
