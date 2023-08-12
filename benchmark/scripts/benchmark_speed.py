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
from os.path import join
import sys
import pickle
import numpy
from datetime import datetime

import glearn
from glearn import sample_data, LinearModel, Covariance, GaussianProcess


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
    # config = {
    #     'dimension': 1,
    #     'data_sizes': 2**numpy.arange(8, 13),
    #     'grid': True,
    #     'noise_magnitude': 0.2,
    #     'polynomial_degree': 2,
    #     'trigonometric_coeff': None,
    #     'hyperbolic_coeff': None,
    #     'b': None,
    #     'B': None,
    #     'scale': 0.07,
    #     'kernel': 'Exponential',
    #     'sparse': False,
    #     'kernel_threshold': 0.03,
    #     'imate_options': {'method': 'eigenvalue'},
    #     'hyperparam_guess': None,
    #     'profile_hyperparam': ['none', 'var'],
    #     'optimization_method': 'Nelder-Mead',
    #     'verbose': False,
    # }

    config = {
        'repeat': 5,
        'dimension': 2,
        'data_sizes': (2**numpy.arange(6, 7.01, 1.0/6.0)).astype(int),
        'grid': True,
        'noise_magnitude': 0.2,
        'polynomial_degree': 2,
        'trigonometric_coeff': None,
        'hyperbolic_coeff': None,
        'b': None,
        'B': None,
        'scale': 0.005,
        'kernel': 'Exponential',
        'sparse': True,
        'kernel_threshold': 0.02,
        'imate_options': {
            'var': {
                'method': 'slq',
                'min_num_samples': 100,
                'max_num_samples': 200,
                'lanczos_degree': 50,
                },
            'none': {
                # 'method': 'cholesky',
                'method': 'slq',
                'min_num_samples': 100,
                'max_num_samples': 200,
                'lanczos_degree': 50,
                },
            },
        'hyperparam_guesses': {'var': [1.0], 'none': [0.1, 0.1]},
        # 'hyperparam_guesses': {'var': None, 'none': None},
        # 'profile_hyperparam': ['var', 'none'],
        'profile_hyperparam': ['none', 'var'],
        # 'optimization_method': {
        #     'var': 'chandrupatla',
        #     'none': 'Nelder-Mead'},
        # 'optimization_method': {'var': 'BFGS', 'none': 'BFGS'},
        # 'optimization_method': {'var': 'Newton-CG', 'none': 'Newton-CG'},
        'optimization_method': {'var': 'CG', 'none': 'CG'},
        # 'optimization_method': {'var': 'BFGS', 'none': 'BFGS'},
        'tol': 1e-4,
        'verbose': False,
    }

    devices = glearn.info(print_only=False)

    # Loop variables
    data_sizes = config['data_sizes']
    profile_hyperparams = config['profile_hyperparam']
    results = []

    # Loop over data filenames
    for i in range(data_sizes.size):

        data_size = data_sizes[i]
        if config['grid']:
            num_all_points = data_size**config['dimension']
        else:
            num_all_points = data_size
        log2_num_all_points = numpy.log2(num_all_points)
        print('%2d/%2d  size: 2**%0.3f'
              % (i+1, data_sizes.size, log2_num_all_points), flush=True)

        # Generate data points
        dimension = config['dimension']
        grid = config['grid']
        points = sample_data.generate_points(data_size, dimension=dimension,
                                             grid=grid)

        # Generate noisy data
        noise_magnitude = config['noise_magnitude']
        z_noisy = sample_data.generate_data(
                points, noise_magnitude=noise_magnitude)

        # Mean
        b = config['b']
        B = config['B']
        polynomial_degree = config['polynomial_degree']
        trigonometric_coeff = config['trigonometric_coeff']
        hyperbolic_coeff = config['hyperbolic_coeff']
        # mean = LinearModel(points, polynomial_degree=polynomial_degree,
        #                    trigonometric_coeff=trigonometric_coeff,
        #                    hyperbolic_coeff=hyperbolic_coeff, b=b, B=B)

        # Prior for scale of correlation
        scale_name = config['scale']
        if scale_name is not str:
            scale = scale_name
        else:
            scale = eval('priors.' + scale_name)

        # Kernel
        kernel_name = config['kernel']
        kernel = eval('kernels.' + kernel_name + "()")

        # Covariance
        scale = config['scale']
        sparse = config['sparse']
        kernel_threshold = config['kernel_threshold']
        # cov = Covariance(points, kernel=kernel, scale=scale, sparse=sparse,
        #                  kernel_threshold=kernel_threshold)

        # Gaussian process
        # gp = GaussianProcess(mean, cov)

        # Training
        verbose = config['verbose']
        full_likelihood_res = []
        profile_likelihood_res = []
        tol = config['tol']

        for profile_hyperparam in profile_hyperparams:

            if profile_hyperparam == 'none':
                hyperparam_guess = config['hyperparam_guesses']['none']
                optimization_method = config['optimization_method']['none']
                imate_options = config['imate_options']['none']
            elif profile_hyperparam == 'var':
                hyperparam_guess = config['hyperparam_guesses']['var']
                optimization_method = config['optimization_method']['var']
                imate_options = config['imate_options']['var']

            if profile_hyperparam == 'none':
                print('       full likelihood ', end='', flush=True)
            else:
                print('       prof likelihood ', end='', flush=True)

            for j in range(config['repeat']):
                print('.', end='', flush=True)

                mean = LinearModel(points, polynomial_degree=polynomial_degree,
                                   trigonometric_coeff=trigonometric_coeff,
                                   hyperbolic_coeff=hyperbolic_coeff, b=b, B=B)
                cov = Covariance(points, kernel=kernel, scale=scale,
                                 sparse=sparse,
                                 kernel_threshold=kernel_threshold)
                gp = GaussianProcess(mean, cov)

                res = gp.train(z_noisy, profile_hyperparam=profile_hyperparam,
                               log_hyperparam=True,
                               optimization_method=optimization_method,
                               tol=tol, hyperparam_guess=hyperparam_guess,
                               verbose=verbose, imate_options=imate_options,
                               plot=False)

                if profile_hyperparam == 'none':
                    full_likelihood_res.append(res)
                else:
                    profile_likelihood_res.append(res)

                del mean
                del cov
                del gp

            print(' Done.', flush=True)

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
