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
from numpy import format_float_scientific as ffs
import scipy
from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, save_plot, plt, \
    matplotlib

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =====================
# print training result
# =====================

def print_training_result(posterior, res):
    """
    Prints the training results.
    """

    # Optimization config
    profile_hyperparam = res['config']['profile_hyperparam']
    optimization_method = res['config']['optimization_method']
    tol = res['config']['tol']
    max_iter = res['config']['max_iter']
    max_bracket_trials = res['config']['max_bracket_trials']
    # use_rel_error = res['config']['use_rel_error']

    # Imate config
    imate_method = res['imate_config']['imate_method']
    imate_interpolate = res['imate_config']['imate_interpolate']
    imate_tol = res['imate_config']['imate_tol']
    imate_min_num_samples = res['imate_config']['min_num_samples']
    imate_max_num_samples = res['imate_config']['max_num_samples']

    # Hyperparam
    hyperparam = res['hyperparam']
    eta = hyperparam['eta']
    sigma = hyperparam['sigma']
    sigma0 = hyperparam['sigma0']
    scale = numpy.asarray(hyperparam['scale'])

    # Optimization
    max_fun = res['optimization']['max_fun']
    num_opt_iter = res['optimization']['num_opt_iter']
    num_fun_eval = res['optimization']['num_fun_eval']
    num_jac_eval = res['optimization']['num_jac_eval']
    num_hes_eval = res['optimization']['num_hes_eval']
    num_cor_eval = res['optimization']['num_cor_eval']

    # Time
    cor_wall_time = res['time']['cor_wall_time']
    cor_proc_time = res['time']['cor_proc_time']
    det_wall_time = res['time']['det_wall_time']
    det_proc_time = res['time']['det_proc_time']
    trc_wall_time = res['time']['trc_wall_time']
    trc_proc_time = res['time']['trc_proc_time']
    sol_wall_time = res['time']['sol_wall_time']
    sol_proc_time = res['time']['sol_proc_time']
    # lik_wall_time = res['time']['lik_wall_time']
    # lik_proc_time = res['time']['lik_proc_time']
    opt_wall_time = res['time']['opt_wall_time']
    opt_proc_time = res['time']['opt_proc_time']

    # Device
    num_cpu_threads = res['device']['num_cpu_threads']
    num_gpu_devices = res['device']['num_gpu_devices']
    num_gpu_multiproc = res['device']['num_gpu_multiproc']
    num_gpu_threads_per_multiproc = \
        res['device']['num_gpu_threads_per_multiproc']
    mem_usage, mem_unit = res['device']['memory_usage']

    # Convert scale (theta) to string
    if scale.size == 1:
        theta_string = '%0.10f' % scale[0]
    else:
        if scale.size == 2:
            digit = 2
        else:
            digit = 1
        theta_string = ''
        for i in range(scale.size):
            theta_string += '%s' \
                % ffs(scale[i], precision=digit, min_digits=digit,
                      exp_digits=1)
            if i < scale.size-1:
                theta_string += ', '

    # Print hyperparameters
    print('')
    print('                                Training Summary                 ' +
          '               ')
    print('=================================================================' +
          '===============')
    print('       posterior/param                optimization              i' +
          'mate solver    ')
    print('-----------------------------      -------------------      -----' +
          '---------------')
    colspace = '      '
    print('posterior    %+0.9e' % max_fun, end=colspace)
    print('method %12s' % optimization_method, end=colspace)
    print('method  %12s' % imate_method)

    print('eta          %0.10e' % eta, end=colspace)
    print('tol        %0.2e' % tol, end=colspace)
    print('tol         %0.2e' % imate_tol)

    print('sigma        %0.10e' % sigma, end=colspace)
    print('max iter       %3d' % max_iter, end=colspace)
    print('interpolate    %5s' % imate_interpolate)

    print('sigma0       %0.10e' % sigma0, end=colspace)
    print('max bracket try  %2d' % max_bracket_trials, end=colspace)
    print('min num samples %4d' % imate_min_num_samples)

    print('theta %23s' % theta_string, end=colspace)
    print('profile param %5s' % profile_hyperparam, end=colspace)
    print('max num samples %4d' % imate_max_num_samples)

    # Print process info (elapsed times, number of function evaluations, cpu
    # and gpu device info).
    print('')
    print('                                    Process                      ' +
          '               ')
    print('=================================================================' +
          '===============')
    print('         time (sec)                    evaluations               ' +
          'processor      ')
    print('-----------------------------      -------------------      -----' +
          '---------------')
    print('task         clock    process      task              #      devic' +
          'e             #')
    print('=================================================================' +
          '===============')
    colspace = '      '
    print('correlation  %s  %s'
          % (ffs(cor_wall_time, precision=2, min_digits=2, exp_digits=1),
             ffs(cor_proc_time, precision=2, min_digits=2, exp_digits=1)),
          end=colspace)
    print('correlation   %5d' % num_cor_eval, end=colspace)
    print('cpu threads   %6d' % num_cpu_threads)

    print('logdet       %s  %s'
          % (ffs(det_wall_time, precision=2, min_digits=2, exp_digits=1),
             ffs(det_proc_time, precision=2, min_digits=2, exp_digits=1)),
          end=colspace)
    print('likelihood    %5d' % num_fun_eval, end=colspace)
    print('gpu devices   %6d' % num_gpu_devices)

    print('traceinv     %s  %s'
          % (ffs(trc_wall_time, precision=2, min_digits=2, exp_digits=1),
             ffs(trc_proc_time, precision=2, min_digits=2, exp_digits=1)),
          end=colspace)
    print('jacobian      %5d' % num_jac_eval, end=colspace)
    print('gpu multiproc %6d' % num_gpu_multiproc)

    print('solver       %s  %s'
          % (ffs(sol_wall_time, precision=2, min_digits=2, exp_digits=1),
             ffs(sol_proc_time, precision=2, min_digits=2, exp_digits=1)),
          end=colspace)
    print('hessian       %5d' % num_hes_eval, end=colspace)
    print('gpu thrds/sm  %6d' % num_gpu_threads_per_multiproc)

    print('overall      %s  %s'
          % (ffs(opt_wall_time, precision=2, min_digits=2, exp_digits=1),
             ffs(opt_proc_time, precision=2, min_digits=2, exp_digits=1)),
          end=colspace)
    print('optimization  %5d' % num_opt_iter, end=colspace)
    if not isinstance(mem_usage, str):
        print('mem used (%s) %6.0f' % (mem_unit, mem_usage))
    else:
        # At this point, memory_usage should be "n/a"
        print('mem used %10s ' % mem_usage)
    print('')


# =========================
# plot training convergence
# =========================

def plot_training_convergence(posterior, res, verbose):
    """
    """

    load_plot_settings()

    fig, ax = plt.subplots(figsize=(6, 4.8))
    markersize = 3

    use_rel_error = res['config']['use_rel_error']
    tol = res['config']['tol']
    errors = res['convergence']['errors'][1:]
    iter = numpy.arange(errors.shape[0]) + 2

    scale_index = posterior.likelihood.scale_index

    if scale_index == 1:

        # label of eta
        if posterior.likelihood.use_log_eta:
            eta_label = r'$\ln \eta$'
        else:
            eta_label = r'$\eta$'

        # Plot convergence for eta hyperparameter
        ax.plot(iter, errors[:, 0], '-o', markersize=markersize, color='black',
                label=eta_label)
    elif scale_index == 2:

        # label of sigmas
        if posterior.likelihood.use_log_sigmas:
            sigma_label = r'$\ln \sigma$'
            sigma0_label = r'$\ln \sigma_0$'
        else:
            sigma_label = r'$\ln \sigma$'
            sigma0_label = r'$\ln \sigma_0$'

        # Plot convergence for sigma and sigma0 hyperparameter
        ax.plot(iter, errors[:, 0], '-o', markersize=markersize, color='black',
                label=sigma_label)
        ax.plot(iter, errors[:, 1], '-o', markersize=markersize, color='gray',
                label=sigma0_label)

    # label of theta (scale)
    if posterior.likelihood.use_log_scale:
        theta_label = r'$\ln \theta'
    else:
        theta_label = r'$\theta'

    # Plot convergence for scale hyperparameters
    num_scales = errors.shape[1] - scale_index
    colors = plt.cm.ocean(numpy.linspace(0.5, 0.95, num_scales))
    for i in range(scale_index, errors.shape[1]):
        ax.plot(iter, errors[:, i], '-o', markersize=markersize,
                color=colors[i-scale_index, :],
                label=theta_label + r'%d$' % (scale_index - i + 1))

    # Plot tolerance line
    ax.plot([iter[0], iter[-1]], [tol, tol], '--', color='black',
            label=r'tolerance')

    if use_rel_error:
        ax.set_ylabel(r'Relative Error')
    else:
        ax.set_ylabel(r'Absolute Error')

    ax.set_xlabel(r'Iterations')
    ax.set_title(r'Convergence of Hyperparameters')
    ax.set_xlim([iter[0], iter[-1]])
    ax.set_yscale('log')
    ax.grid(True, which='major', axis='y')
    ax.legend(fontsize='small', loc='lower left')

    # Save plots
    plt.tight_layout()
    filename = 'training_convergence'
    save_plot(plt, filename, transparent_background=False, pdf=True)

    if verbose:
        print('Plot saved to %s.' % filename)

    plt.show()


# ===============
# plot prediction
# ===============

def plot_prediction(
        points,
        test_points,
        z,
        z_star_mean,
        z_star_cov=None,
        confidence_level=0.95,
        true_data=None,
        verbose=False):
    """
    Plots prediction mean and covariance for 1D or 2D data.
    """

    if points.ndim == 1 or points.shape[1] == 1:
        # Plot 1D data
        plot_prediction_1d(points, test_points, z, z_star_mean, z_star_cov,
                           confidence_level, true_data, verbose)
    elif points.shape[1] == 2:

        if true_data is not None:
            raise RuntimeError('"true_data" can be plotted for only 1D data.')

        # Plot 2D data
        plot_prediction_2d(points, test_points, z, z_star_mean, z_star_cov,
                           confidence_level, verbose)
    else:
        raise ValueError('Predictions can be plotted for only 1D and 2D data.')


# ==================
# plot prediction 1D
# ==================

def plot_prediction_1d(
        points,
        test_points,
        z,
        z_star_mean,
        z_star_cov=None,
        confidence_level=0.95,
        true_data=None,
        verbose=False):
    """
    Plots prediction mean and covariance for 1D data.
    """

    load_plot_settings()

    # Short names, also since data are 1D, use vector of points than 2D array
    x = points[:, 0]
    x_star = test_points[:, 0]

    # Sort training points
    x_sorting_index = numpy.argsort(x)
    x = x[x_sorting_index]
    z = z[x_sorting_index]

    # Sort test points
    x_star_sorting_index = numpy.argsort(x_star)
    x_star = x_star[x_star_sorting_index]
    z_star_mean = z_star_mean[x_star_sorting_index]
    z_star_cov_ = z_star_cov[x_star_sorting_index, :]
    z_star_cov_ = z_star_cov_[:, x_star_sorting_index]

    fig, ax = plt.subplots(figsize=(6, 4.8))
    markersize = 3

    # Plot training data (possibly with noise)
    ax.plot(x, z, 'o', markersize=markersize, color='gray',
            label='training (noisy) data')

    # Plot true data (without noise) on test points
    if true_data is not None:
        ax.plot(x_star, true_data, '--', color='black',
                label='true (noise-free) data')

    # Plot predicted data on test points
    ax.plot(x_star, z_star_mean, color='black',
            label='posterior predictive mean')

    # Plot uncertainty of the test points
    if z_star_cov is not None:

        z_score = numpy.sqrt(2.0) * scipy.special.erfinv(confidence_level)

        # Get the diagonal of matrix
        if scipy.sparse.isspmatrix(z_star_cov):
            z_star_var = z_star_cov.diagonal()
        else:
            z_star_var = numpy.diag(z_star_cov)

        # Standard deviation
        z_star_std = numpy.sqrt(z_star_var)

        # Error, based on confidence level
        error = z_score * z_star_std

        ax.fill_between(
                x_star, z_star_mean - error, z_star_mean + error,
                color='black', alpha=0.25,
                label=(str(100.0*confidence_level).strip('0').strip('.') +
                       r'$\%$ confidence region'))

    x_min = numpy.min(numpy.r_[x, x_star])
    x_max = numpy.max(numpy.r_[x, x_star])
    ax.set_xlim([x_min, x_max])
    ax.set_xlabel(r'$x^*$')
    ax.set_ylabel(
            r'$z^*(x^*|z, \beta, \sigma, \sigma_0, \boldsymbol{\theta})$')
    ax.set_title('Prediction')
    ax.legend(fontsize='small')

    # Save plots
    plt.tight_layout()
    filename = 'prediction'
    save_plot(plt, filename, transparent_background=False, pdf=True,
              verbose=verbose)

    plt.show()


# ==================
# plot prediction 2D
# ==================

def plot_prediction_2d(
        points,
        test_points,
        z,
        z_star_mean,
        z_star_cov=None,
        confidence_level=0.95,
        verbose=False):
    """
    Plots prediction mean and covariance for 2D data.

    .. warning::

        matplotlib 3D has bugs in 3D plots that make the plots of this function
        look erroneous. For example, if two plots overlap, at certain angles,
        one plot is rendered completely above the other plot, even if some
        parts of one plot is behind the other.

        In this function, we have two plots, (1) a scatter plot of training
        data points, and (2) a mean surface plot of test points. The mean
        surface should lie between training points, that is, some of the
        training points should be above the mean surface, and some below.

        However, due to the matplotlib bug, all training points are rendered
        either above or below the mean surface (depending on the view angle).
        Unfortunately, matplotlib way of rendering two or more 3D plots are by
        zorder, not by OpenGL engine. This problem is unavoidable with
        matplotlib at the moment. Best solution is to use mayavi.

        The matplotlib bug is described here:
        https://matplotlib.org/2.2.2/mpl_toolkits/mplot3d/faq.html
    """

    load_plot_settings()
    colormap = 'magma_r'
    cmap = plt.cm.get_cmap(colormap)

    triang = matplotlib.tri.Triangulation(
            test_points[:, 0], test_points[:, 1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], z, marker='.', s=7, c='black',
               label='training data')

    # Plot uncertainty of the test points
    if z_star_cov is not None:

        z_score = numpy.sqrt(2.0) * scipy.special.erfinv(confidence_level)

        # Get the diagonal of matrix
        if scipy.sparse.isspmatrix(z_star_cov):
            z_star_var = z_star_cov.diagonal()
        else:
            z_star_var = numpy.diag(z_star_cov)

        # Standard deviation
        z_star_std = numpy.sqrt(z_star_var)

        # Error, based on confidence level
        error = z_score * z_star_std

        # vertices of triangulation
        X = numpy.c_[test_points, z_star_mean]
        triang_vertices = X[triang.triangles]

        # Get mid points of triangular cells
        midpoints = numpy.average(triang_vertices, axis=1)
        midx = midpoints[:, 0]
        midy = midpoints[:, 1]

        # Interpolate the value of error at the center of triangular cells
        triang_interpolator = matplotlib.tri.LinearTriInterpolator(
            triang, error)
        face_error = triang_interpolator(midx, midy)

        # Normalize the errors to be between 0 and 1 to map to colors
        face_error_min = numpy.min(face_error)
        face_error_max = numpy.max(face_error)
        norm = (face_error - face_error_min) / \
            (face_error_max - face_error_min)

        # Map to colors
        facecolors = cmap(norm)

        # Plot a 3D patch collection
        collection = Poly3DCollection(triang_vertices, facecolors=facecolors,
                                      edgecolors=(0, 0, 0, 0),
                                      antialiased=False,
                                      label='posterior predictive mean')
        surf = ax.add_collection(collection)

        # Colorbar
        cbar_norm = matplotlib.colors.Normalize(vmin=face_error_min,
                                                vmax=face_error_max)
        cbar = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=cbar_norm,
                                  cmap=cmap), ax=ax, pad=0.05, fraction=0.05,
                                  shrink=0.5)
        cbar.ax.set_ylabel('Posterior predictive standard deviation')

    else:

        # Just plot the mean (a gray surface) without the colored by the errors
        surf = ax.plot_trisurf(triang, z_star_mean, edgecolor=(0, 0, 0, 0),
                               antialiased=True, color='gray',
                               label='posterior predictive mean')

    # To avoid a bug in matplotlib
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot limit
    x_min = numpy.min(numpy.r_[points[:, 0], test_points[:, 0]])
    x_max = numpy.max(numpy.r_[points[:, 0], test_points[:, 0]])
    y_min = numpy.min(numpy.r_[points[:, 1], test_points[:, 1]])
    y_max = numpy.max(numpy.r_[points[:, 1], test_points[:, 1]])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    ax.view_init(elev=20, azim=-155)
    ax.set_xlabel(r'$x^*_1$')
    ax.set_ylabel(r'$x^*_2$')
    ax.set_zlabel(r'$z^*(\boldsymbol{x}^*|z, \beta, \sigma, \sigma_0, ' +
                  r'\boldsymbol{\theta})$')
    ax.set_title('Prediction')
    ax.legend(fontsize='small')

    # Save plots
    plt.tight_layout()
    filename = 'prediction'
    save_plot(plt, filename, transparent_background=False, pdf=True,
              verbose=verbose)

    plt.show()
