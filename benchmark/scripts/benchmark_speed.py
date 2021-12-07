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

import getopt
import os
from os.path import join
import sys
import pickle
import numpy
import multiprocessing
from datetime import datetime

from glearn.sample_data import generate_points, generate_data
from glearn import get_processor_name, get_gpu_name, get_num_gpu_devices
from glearn.mean import LinearModel
from glearn.kernels import Matern, Exponential, SquareExponential  # noqa: F401
from glearn.kernels import RationalQuadratic, Linear               # noqa: F401
from glearn.priors import Uniform, Cauchy, StudentT, Erlang        # noqa: F401
from glearn.priors import Gamma, InverseGamma, Normal, BetaPrime   # noqa: F401
from glearn import Correlation
from glearn import Covariance
from glearn import GaussianProcess


# ===============
# parse arguments
# ===============

def parse_arguments(argv):
    """
    Parses the argument of the executable and obtains the filename.
    """

    # -----------
    # print usage
    # -----------

    def print_usage(exec_name):
        usage_string = "Usage: " + exec_name + " <arguments>"
        options_string = """
At least, one of the following arguments are required:

    -c --cpu      Runs the benchmark on CPU. Default is not to run on cpu.
    -g --gpu      Runs the benchmark on GPU. Default is not to run in gpu.
        """

        print(usage_string)
        print(options_string)

    # -----------------

    # Initialize variables (defaults)
    arguments = {
        'use_cpu': False,
        'use_gpu': False
    }

    # Get options
    try:
        opts, args = getopt.getopt(argv[1:], "cg", ["cpu", "gpu"])
    except getopt.GetoptError:
        print_usage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:
        if opt in ('-c', '--cpu'):
            arguments['use_cpu'] = True
        elif opt in ('-g', '--gpu'):
            arguments['use_gpu'] = True

    if len(argv) < 2:
        print_usage(argv[0])
        sys.exit()

    return arguments


# =========
# benchmark
# =========

def benchmark(argv):
    """
    Test for :mod:`imate.traceinv` sub-package.
    """

    # arguments = parse_arguments(argv)
    benchmark_dir = '..'

    # Settings
    config = {
        'dimension': 1,
        'data_sizes': 2**numpy.arange(8, 14),
        'grid': True,
        'noise_magnitude': 0.05,
        'polynomial_degree': 2,
        'trigonometric_coeff': None,
        'hyperbolic_coeff': None,
        'b': None,
        'B': None,
        'scale': 0.07,
        'scale_prior': 'Uniform',
        'kernel': 'Exponential',
        'sparse': True,
        'kernel_threshold': 0.03,
        # 'imate_method': 'cholesky',
        'imate_method': 'slq',
        'hyperparam_guess': None,
        'profile_hyperparam': ['none', 'var'],
        'optimization_method': 'Nelder-Mead',
        'verbose': False,
    }

    devices = {
        'cpu_name': get_processor_name(),
        'gpu_name': get_gpu_name(),
        'num_all_cpu_threads': multiprocessing.cpu_count(),
        'num_all_gpu_devices': get_num_gpu_devices()
    }

    # For reproducibility
    numpy.random.seed(0)

    # Loop variables
    data_sizes = config['data_sizes']
    profile_hyperparams = config['profile_hyperparam']
    results = []

    # Loop over data filenames
    for i in range(data_sizes.size):

        data_size = data_sizes[i]
        print('data size: %d' % data_size)

        # Generate data points
        dimension = config['dimension']
        grid = config['grid']
        points = generate_points(data_size, dimension, grid)

        # Generate noisy data
        noise_magnitude = config['noise_magnitude']
        z_noisy = generate_data(points, noise_magnitude, plot=False)

        # Mean
        b = config['b']
        B = config['B']
        polynomial_degree = config['polynomial_degree']
        trigonometric_coeff = config['trigonometric_coeff']
        hyperbolic_coeff = config['hyperbolic_coeff']
        mean = LinearModel(points, polynomial_degree=polynomial_degree,
                           trigonometric_coeff=trigonometric_coeff,
                           hyperbolic_coeff=hyperbolic_coeff, b=b, B=B)

        # Prior for scale of correlation
        # scale_prior = Uniform()
        # scale_prior = Cauchy()
        # scale_prior = StudentT()
        # scale_prior = InverseGamma()
        # scale_prior = Normal()
        # scale_prior = Erlang()
        # scale_prior = BetaPrime()
        # scale_prior_name = config['scale_prior']
        # scale_prior = eval(scale_prior_name)

        # Kernel
        # kernel = Matern()
        # kernel = Exponential()
        # kernel = Linear()
        # kernel = SquareExponential()
        # kernel = RationalQuadratic()
        kernel_name = config['kernel']
        kernel = eval(kernel_name + "()")

        # Correlation
        scale = config['scale']
        sparse = config['sparse']
        kernel_threshold = config['kernel_threshold']
        # cor = Correlation(points, kernel=kernel, scale=scale_prior,
        #                   sparse=sparse)
        cor = Correlation(points, kernel=kernel, scale=scale, sparse=sparse,
                          kernel_threshold=kernel_threshold)

        # Covariance
        imate_method = config['imate_method']
        cov = Covariance(cor, imate_method=imate_method)

        # Gaussian process
        gp = GaussianProcess(mean, cov)

        # Training
        hyperparam_guess = config['hyperparam_guess']
        optimization_method = config['optimization_method']
        verbose = config['verbose']

        full_likelihood_res = None
        profile_likelihood_res = None

        for profile_hyperparam in profile_hyperparams:
            res = gp.train(z_noisy, profile_hyperparam=profile_hyperparam,
                           log_hyperparam=True,
                           optimization_method=optimization_method, tol=1e-6,
                           hyperparam_guess=hyperparam_guess, verbose=verbose,
                           plot=False)

            if profile_hyperparam == 'none':
                full_likelihood_res = res
                print('\t full likelihood')
            else:
                profile_likelihood_res = res
                print('\t prof likelihood')

        result = {
                'data_size': data_size,
                'full_likelihood': full_likelihood_res,
                'prof_likelihood': profile_likelihood_res,
        }

        results.append(result)

    now = datetime.now()

    # Final object of all results
    benchmark_results = {
        'config': config,
        'devices': devices,
        'results': results,
        'date': now.strftime("%d/%m/%Y %H:%M:%S")
    }

    # Save to file
    pickle_dir = 'pickle_results'
    output_filename = 'benchmark_speed'
    output_filename += '.pickle'
    output_full_filename = join(benchmark_dir, pickle_dir, output_filename)
    with open(output_full_filename, 'wb') as file:
        pickle.dump(benchmark_results, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Results saved to %s.' % output_full_filename)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(benchmark(sys.argv))
