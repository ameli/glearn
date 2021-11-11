#! /usr/bin/env python

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

import sys
import numpy
from gaussian_proc.sample_data import generate_points, generate_data
from gaussian_proc.mean import LinearModel
from gaussian_proc.kernels import Matern, Exponential, SquareExponential, \
        RationalQuadratic, Linear
from gaussian_proc.priors import Uniform, Cauchy, StudentT, Erlang, \
        Gamma, InverseGamma, Normal, BetaPrime
from gaussian_proc import Correlation
from gaussian_proc import Covariance
from gaussian_proc import GaussianProcess


# ====
# main
# ====

def main():

    # Generate points
    # num_points = 30
    num_points = 50
    dimension = 1
    grid = True
    points = generate_points(num_points, dimension, grid)

    # Generate noisy data
    # noise_magnitude = 0.2
    noise_magnitude = 0.05
    z = generate_data(points, noise_magnitude, plot=False)

    # Mean
    b = numpy.zeros((6, ))
    B = numpy.random.rand(b.size, b.size)
    B = 1e+5 * B.T @ B
    # b = None
    # B = None
    polynomial_degree = 5
    # trigonometric_coeff = [0.2]
    # trigonometric_coeff = [0.1, 0.2, 0.3, 1.0]
    trigonometric_coeff = None
    # hyperbolic_coeff = [0.4, 0.7, 1.0]
    hyperbolic_coeff = None
    mean = LinearModel.design(points, polynomial_degree=polynomial_degree,
                              trigonometric_coeff=trigonometric_coeff,
                              hyperbolic_coeff=hyperbolic_coeff, b=b, B=B)

    # Prior for scale of correlation
    scale_prior = Uniform()
    # scale_prior = Cauchy()
    # scale_prior = StudentT()
    # scale_prior = InverseGamma()
    # scale_prior = Normal()
    # scale_prior = Erlang()
    # scale_prior = BetaPrime()
    # scale_prior.plot()

    # Kernel
    # kernel = Matern()
    kernel = Exponential()
    # kernel = Linear()
    # kernel = SquareExponential()
    # kernel = RationalQuadratic()

    # Correlation
    # cor = Correlation(points, kernel=kernel, scale=0.07, sparse=False)
    # cor = Correlation(points, kernel=kernel, sparse=False)
    cor = Correlation(points, kernel=kernel, scale=scale_prior, sparse=False)

    # Covariance
    # imate_method = 'eigenvalue'
    imate_method = 'cholesky'
    # imate_method = 'hutchinson'
    # imate_method = 'slq'
    cov = Covariance(cor, imate_method=imate_method)

    # Gaussian process
    gp = GaussianProcess(mean, cov)

    # Training options
    # profile_hyperparam = 'none'
    profile_hyperparam = 'var'
    # profile_hyperparam = 'var_noise'

    # optimization_method = 'chandrupatla'  # requires jacobian
    # optimization_method = 'Nelder-Mead'     # requires func
    # optimization_method = 'BFGS'          # requires func, jacobian
    # optimization_method = 'CG'            # requires func, jacobian
    optimization_method = 'Newton-CG'     # requires func, jacobian, hessian
    # optimization_method = 'dogleg'        # requires func, jacobian, hessian
    # optimization_method = 'trust-exact'   # requires func, jacobian, hessian
    # optimization_method = 'trust-ncg'     # requires func, jacobian, hessian

    # hyperparam_guess = [1.0]
    # hyperparam_guess = [0, 0.1, 0.1]
    # hyperparam_guess = [-1, 1e-1]
    # hyperparam_guess = [1.0]
    # hyperparam_guess = [0.1, 0.1]
    # hyperparam_guess = [1.0, 0.1]
    # hyperparam_guess = [0.1, 0.1, 0.1, 0.1]
    # hyperparam_guess = [0.01, 0.01, 0.1]
    hyperparam_guess = None

    # gp.train(z, options=options, plot=False)
    gp.train(z, profile_hyperparam=profile_hyperparam, log_hyperparam=True,
             optimization_method=optimization_method, tol=1e-5,
             hyperparam_guess=hyperparam_guess, verbose=False, plot=True)

# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
