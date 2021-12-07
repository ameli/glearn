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
import time
from glearn.sample_data import generate_points, generate_data
from glearn.mean import LinearModel
from glearn.kernels import Matern, Exponential, SquareExponential, \
        RationalQuadratic, Linear
from glearn import Correlation
from glearn import Covariance
from glearn import GaussianProcess

import matplotlib.pyplot as plt
import numpy


# ====
# main
# ====

def main():

    # Generate points
    # num_points = 30
    # num_points = 95
    num_points = 200
    dimension = 1
    grid = True
    points = generate_points(num_points, dimension, grid)

    # Generate noisy data
    # noise_magnitude = 0.2
    noise_magnitude = 0.1
    z = generate_data(points, noise_magnitude, plot=False)

    # Mean
    mean = LinearModel(points, polynomial_degree=2)

    scale = 0.2

    # Correlation
    # kernel = Matern()
    kernel = Exponential()
    # kernel = Linear()
    # kernel = SquareExponential()
    # kernel = RationalQuadratic()
    cor = Correlation(points, kernel=kernel, scale=scale,
                      sparse=False)
    # cor = Correlation(points, kernel=kernel, sparse=False)

    # Covariance
    cov = Covariance(cor, imate_method='cholesky')

    # K0 = cor.get_matrix(derivative=[])
    # K1 = cor.get_matrix(derivative=[0])
    # K2 = cor.get_matrix(derivative=[0, 0])

    sigma = 3.0
    sigma0 = 0.0
    K0 = cov.get_matrix(sigma, sigma0, derivative=[])
    K1 = cov.get_matrix(sigma, sigma0, derivative=[0])
    K2 = cov.get_matrix(sigma, sigma0, derivative=[0, 0])

    x = numpy.zeros((points.shape[0], ), dtype=float)
    k0 = numpy.zeros((points.shape[0], ), dtype=float)
    k1 = numpy.zeros((points.shape[0], ), dtype=float)
    k2 = numpy.zeros((points.shape[0], ), dtype=float)
    for i in range(x.size):
        x[i] = numpy.abs(points[i, 0] - points[0, 0])
        k0[i] = sigma**2 * kernel.kernel(x[i]/scale, derivative=0)
        k1[i] = k0[i] * x[i] / scale**2
        k2[i] = k1[i] * (-2.0 + x[i]/scale) / scale

    fig, ax = plt.subplots(ncols=3, figsize=(13, 4))
    ax[0].plot(x, K0[0, :], color='gray')
    ax[1].plot(x, K1[0, :], color='skyblue')
    ax[2].plot(x, K2[0, :], color='salmon')
    ax[0].plot(x, k0[:], '--', color='black')
    ax[1].plot(x, k1[:], '--', color='blue')
    ax[2].plot(x, k2[:], '--', color='red')
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

    plt.show()



    # # Gaussian process
    # gp = GaussianProcess(mean, cov)
    #
    # # Trainign options
    # # profile_param = 'none'
    # profile_param = 'var'
    # # profile_param = 'var_noise'
    #
    # # optimization_method = 'chandrupatla'  # requires jacobian
    # optimization_method = 'Nelder-Mead'     # requires func
    # # optimization_method = 'BFGS'          # requires func, jacobian
    # # optimization_method = 'CG'            # requires func, jacobian
    # # optimization_method = 'Newton-CG'     # requires func, jacobian, hessian
    # # optimization_method = 'dogleg'        # requires func, jacobian, hessian
    # # optimization_method = 'trust-exact'   # requires func, jacobian, hessian
    # # optimization_method = 'trust-ncg'     # requires func, jacobian, hessian
    #
    # # hyperparam_guess = [0.0]
    # # hyperparam_guess = [0, 0.1, 0.1]
    # # hyperparam_guess = [0.1, 0.1]
    # # hyperparam_guess = [0.1]
    # hyperparam_guess = [0, 0.1]
    # # hyperparam_guess = [0.1, 0.1]
    # # hyperparam_guess = [0.1, 0.1, 0.1, 0.1]
    # # hyperparam_guess = [0.01, 0.01, 0.1]
    #
    # t0 = time.time()
    # # gp.train(z, options=options, plot=False)
    # gp.train(z, profile_param=profile_param,
    #          optimization_method=optimization_method,
    #          hyperparam_guess=hyperparam_guess, verbose=False, plot=True)
    # t1 = time.time()
    # print('Elapsed time: %0.2f' % (t1 - t0))


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main())
