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

__all__ = ['chandrupatla']


# ============
# chandrupatla
# ============

def chandrupatla(
        fun,
        bracket,
        bracket_values,
        verbose=False,
        eps_m=None,
        eps_a=None,
        maxiter=50):
    """
    Finds roots of a uni-variate function using Chandrupatla method.

    This function is used to find extrema of posterior function where the
    function f is the Jacobian of posterior.

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
        fa = fun(a)
        fb = fun(b)
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

    # Some guesses for default values of the eps_m and eps_a settings based on
    # machine precision.
    eps = numpy.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2*eps

    iterations = 0
    terminate = False
    converged = False

    while maxiter > 0:
        maxiter -= 1
        # Use t to linearly interpolate between a and b, and evaluate this
        # function as our newest estimate xt.
        xt = a + t*(b-a)
        ft = fun(xt)
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
            converged = True
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
    res = {
        'root': xm,
        'num_opt_iter': iterations,
        'converged': converged
    }

    return res
