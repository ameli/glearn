# =======
# Imports
# =======

from cython import boundscheck, wraparound
from scipy.special.cython_special cimport gamma, kv
from libc.math cimport sqrt, exp, isnan, isinf
from libc.stdio cimport printf

__all__ = ['matern_kernel', 'euclidean-distance']


# =============
# matern kernel
# =============

cdef double matern_kernel(
        const double x,
        const double nu) nogil:
    """
    Computes the Matern class correlation function for a given Euclidean
    distance of two spatial points.

    The Matern correlation function defined by

    .. math::
        K(\\boldsymbol{x},\\boldsymbol{x}'|\\rho,\\nu) =
            \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}
            \\left( \\sqrt{2 \\nu} \\frac{\\| \\boldsymbol{x} -
            \\boldsymbol{x}' \\|}{\\rho} \\right)
            K_{\\nu}\\left(\\sqrt{2 \\nu}  \\frac{\\|\\boldsymbol{x} -
            \\boldsymbol{x}' \\|}{\\rho} \\right)

    where

        * :math:`\\rho` is the correlation scale of the function,
        * :math:`\\Gamma` is the Gamma function,
        * :math:`\\| \\cdot \\|` is the Euclidean distance,
        * :math:`\\nu` is the smoothness parameter.
        * :math:`K_{\\nu}` is the modified Bessel function of the second kind
          of order :math:`\\nu`

    .. warning::

        When the distance :math:`\\| \\boldsymbol{x} - \\boldsymbol{x}' \\|` is
        zero, the correlation function produces :math:`\\frac{0}{0}` but in the
        limit, the correlation function is :math:`1`. If the distance is not
        exactly zero, but close to zero, this function might produce unstable
        results.

    In this function, it is assumed that :math:`\\nu = \\frac{5}{2}`, and the
    Matern correlation in this case can be represented by:

    .. math::
        K(\\boldsymbol{x},\\boldsymbol{x}'|\\rho,\\nu) =
        \\left( 1 + \\sqrt{5} \\frac{\\| \\boldsymbol{x} -
        \\boldsymbol{x}'\\|}{\\rho} + \\frac{5}{3} \\frac{\\| \\boldsymbol{x} -
        \\boldsymbol{x}'\\|^2}{\\rho^2} \\right)
        \\exp \\left( -\\sqrt{5} \\frac{\\| \\boldsymbol{x} -
        \\boldsymbol{x}'\\|}{\\rho} \\right)

    :param x: The distance  that represents the Euclidean distance between
        mutual points.
    :type x: ndarray

    :return: Matern correlation
    :rtype: double
    """

    # scaled distance
    cdef double correlation

    if x == 0:
        correlation = 1.0
    else:
        if nu == 0.5:
            correlation = exp(-x)
        elif nu == 1.5:
            correlation = (1.0 + sqrt(3.0) * x) * exp(-sqrt(3.0) * x)
        elif nu == 2.5:
            correlation = (1.0 + sqrt(5.0) * x + (5.0 / 3.0) * (x**2)) * \
                    exp(-sqrt(5.0) * x)
        elif nu < 100:

            # Change zero elements of x to a dummy number, to avoid
            # multiplication of zero by Inf in Bessel function below
            correlation = ((2.0**(1.0-nu)) / gamma(nu)) * \
                    ((sqrt(2.0*nu) * x)**nu) * kv(nu, sqrt(2.0*nu)*x)

        else:
            # For nu > 100, assume nu is Inf. In this case, Matern function
            # approaches Gaussian kernel
            correlation = exp(-0.5*x**2)

        if isnan(correlation):
            printf('correlation is nan.\n')
        if isinf(correlation):
            printf('correlation is inf.\n')

    return correlation


# ==================
# euclidean_distance
# ==================

@boundscheck(False)
@wraparound(False)
cdef double euclidean_distance(
        const double[:] point1,
        const double[:] point2,
        const double[:] correlation_scale,
        const int dimension) nogil:
    """
    Returns the weighted Euclidean distance between two points.

    :param point1: 1D array of the coordinates of a point
    :type point1: cython memoryview (double)

    :param point2: 1D array of the coordinates of a point
    :type point2: cython memoryview (double)

    :param dimension: Dimension of the coordinates of the points.
    :type dimension: int

    :return: Euclidean distance betwrrn point1 and point2
    :rtype: double
    """

    cdef double distance2 = 0
    cdef int dim

    for dim in range(dimension):
        distance2 += ((point1[dim] - point2[dim]) / correlation_scale[dim])**2

    return sqrt(distance2)
