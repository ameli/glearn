#! /usr/bin/env python

"""
Configurations:

1. Before running this code, make sure in TraceEstimation.py, the ComputeTraceOfInverse() is set to
   Cholesky method, with either UseInverse or without it. With UseInverse, the code is faster but the
   results for small n (~2000 more or less) is the samewith and without computing inverse directly.

2. Also, for accurate results, disable interpolation of trace, rather compute trace for every eta 
   directly. The results of the paper for the table 1 is obtained this way. But the rest of the 
   paper, especially those reuslts for performance, the trace is interpolated.
   to disable interpolation of trace, do as follow:
       in LikelihoodEstimaton.py > LogLikelihoodFirstDerivative(), comment and uncomment these two likes as below:
           # TraceKninv = TraceEstimation.EstimateTrace(TraceEstimationUtilities, eta)
           TraceKninv = TraceEstimation.ComputeTraceOfInverse(Kn)    # Use direct method without interpolation, Test
"""

# =======
# Imports
# =======

# Classes
import Data
from LikelihoodEstimation import LikelihoodEstimation
from TraceEstimation import TraceEstimation
from PlotSettings import *

# =============================
# Compute Noise For Single Data
# =============================

def ComputeNoiseForSingleData():
    """
    This function uses three methods
        1. Maximizing log likelihood with parameters sigma and sigma0
        2. Maximizing log likelihood with parameters sigma and eta
        3. Finding zeros of derivative of log likelihood

    This script uses a single data, for which the random noise with a given standard deviation is added to the data once.
    It plots
        1. liklelihood in 3D as function of parameters sigma and eta
        2. Trace estimation using interpolation
        3. Derivative of log likelihood.
    """

    # Generate noisy data
    NumPointsAlongAxis = 50
    NoiseMagnitude = 0.2
    GridOfPoints = True
    x, y, z = Data.GenerateData(NumPointsAlongAxis, NoiseMagnitude, GridOfPoints)

    # Generate Linear Model
    DecorrelationScale = 0.1
    UseSparse = False
    nu = 0.5
    K = Data.GenerateCorrelationMatrix(x, y, z, DecorrelationScale, nu, UseSparse)

    # BasisFunctionsType = 'Polynomial-0'
    # BasisFunctionsType = 'Polynomial-1'
    BasisFunctionsType = 'Polynomial-2'
    # BasisFunctionsType = 'Polynomial-3'
    # BasisFunctionsType = 'Polynomial-4'
    # BasisFunctionsType = 'Polynomial-5'
    # BasisFunctionsType = 'Polynomial-2-Trigonometric-1'
    X = Data.GenerateLinearModelBasisFunctions(x, y, BasisFunctionsType)

    # Trace estimation weights
    UseEigenvaluesMethod = False    # If set to True, it overrides the interpolation estimation methods
    # TraceEstimationMethod = 'NonOrthogonalFunctionsMethod'   # highest condtion number
    # TraceEstimationMethod = 'OrthogonalFunctionsMethod'      # still high condition number
    TraceEstimationMethod = 'OrthogonalFunctionsMethod2'     # best (lowest) condition number
    # TraceEstimationMethod = 'RBFMethod'

    # Precompute trace interpolation function
    TraceEstimationUtilities = TraceEstimation.ComputeTraceEstimationUtilities(K, UseEigenvaluesMethod, TraceEstimationMethod, None, [1e-4, 4e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3])

    # Finding optimal parameters with maximum likelihood using parameters (sigma, sigma0)
    # Results = LikelihoodEstimation.MaximizeLogLikelihoodWithSigmaSigma0(z, X, K, TraceEstimationUtilities)
    # print(Results)

    # Finding optimal parameters with maximum likelihood using parameters (sigma, eta)
    # Results = LikelihoodEstimation.MaximizeLogLikelihoodWithSigmaEta(z, X, K, TraceEstimationUtilities)
    # print(Results)

    # Finding optimal parameters with derivative of likelihood
    Interval_eta = [1e-4, 1e+3]   # Note: make sure the interval is exactly the end points of eta_i, not less or more.
    Results = LikelihoodEstimation.FindZeroOfLogLikelihoodFirstDerivative(z, X, K, TraceEstimationUtilities, Interval_eta)
    print(Results)

    # Plot likelihood and its derivative
    # LikelihoodEstimation.PlotLogLikelihood(z, X, K, TraceEstimationUtilities)
    LikelihoodEstimation.PlotLogLikelihoodFirstDerivative(z, X, K, TraceEstimationUtilities, Results['eta'])

# ====
# Main
# ====

if __name__ == "__main__":

    ComputeNoiseForSingleData()
