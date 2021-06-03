#! /usr/bin/env python

"""
Configurations before runing this script:

    -In Data.py: disable Ray paralleism by:
        In GenerateCorrelationMatrix(), set RunInParallel to False.
        Before the signature of ComputeCorrelationForAProcess(), comment @ray.remote.

    - In TraceEstimation.py > ComputeTraceOfInverse(), set the method to the Stochastic Lanczos Quadrature Method, and
    - In LikelihoodEstimation, set the estimate of trace to the interpolation method, like this
            TraceKninv = TraceEstimation.EstimateTrace(TraceEstimationUtilities,eta)
            # TraceKninv = TraceEstimation.ComputeTraceOfInverse(Kn)    # Use direct method without interpolation, Test
"""

# =======
# Imports
# =======

import numpy
import scipy
from scipy import ndimage
from scipy import interpolate
from functools import partial
import multiprocessing
import pickle
import time

import matplotlib
from matplotlib import cm
from  matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Classes
import Data
from LikelihoodEstimation import LikelihoodEstimation
from TraceEstimation import TraceEstimation
from PlotSettings import *

# =========================
# Find Optimal Sigma Sigma0
# =========================

def FindOptimalSigmaSigma0(x,y,z,X,UseEigenvaluesMethod,UseSparse,TraceEstimationMethod,DecorrelationScale,nu):
    """
    For a given DecorrelationScale and nu, it finds optimal sigma and sigma0
    """

    K = Data.GenerateCorrelationMatrix(x,y,z,DecorrelationScale,nu,UseSparse)

    # Precompute trace interpolation function
    TraceEstimationUtilities = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,None,[1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3])

    # Finding optimal parameters with maximum likelihood using parameters (sigma,sigma0)
    # Results = LikelihoodEstimation.MaximizeLogLikelihoodWithSigmaSigma0(z,X,K,TraceEstimationUtilities_1)

    # Finding optimal parameters with maximum likelihood using parameters (sigma,eta)
    # Results = LikelihoodEstimation.MaximizeLogLikelihoodWithSigmaEta(z,X,K,TraceEstimationUtilities_1)

    # Finding optimal parameters with derivative of likelihood
    Interval_eta = [1e-3,1e+3]   # Note: make sure the interval is exactly the end points of eta_i, not less or more.
    Results = LikelihoodEstimation.FindZeroOfLogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities,Interval_eta)
    Optimal_sigma = Results['sigma']
    Optimal_sigma0 = Results['sigma0']

    return Optimal_sigma,Optimal_sigma0,TraceEstimationUtilities,K

# =============
# Uniform Prior
# =============

def UniformPrior(Parameter,Bounds):
    """
    Uniform prior to limit a parameter within a bound
    """

    if Parameter < Bounds[0] or Parameter > Bounds[1]:
        return 0
    else:
        return 1

# ===========================
# Partial Likelihood Function
# ===========================

def PartialLikelihoodFunction( \
        NumPoints, \
        NoiseMagnitude, \
        GridOfPoints, \
        BasisFunctionsType, \
        UseEigenvaluesMethod, \
        TraceEstimationMethod, \
        UseSparse, \
        Parameters):
    """
    The correlation K is a function of
        - Decorrelaton scale
        - nu
        - sigma
        - sigma0

    Given DecorrelationScale and nu, we find optimal values for sigma and sigma0 using our method.
    The log likelihood function is then computed based on the optimal sigma, sigma0 and the given Decottrlation scale and nu.

    This function is used by a caller function to find optimal values for Decorelation scale and nu.
    """

    x,y,z = Data.GenerateData(NumPoints,NoiseMagnitude,GridOfPoints)
    X = Data.GenerateLinearModelBasisFunctions(x,y,BasisFunctionsType)

    # If parameters are only DecorrelationScale and nu, use our method.
    if len(Parameters) == 2:

        # Extract paramneters
        DecorrelationScale = Parameters[0]
        nu = Parameters[1]

        # Uniform prior # SETTING
        # Prior1 = UniformPrior(DecorrelationScale,[0.1,0.3])
        # Prior2 = UniformPrior(nu,[0.5,25])

        # Uniform prior
        Prior1 = UniformPrior(DecorrelationScale,[0,numpy.inf])
        Prior2 = UniformPrior(nu,[0,25])

        # Inverse square prior
        # Prior1 = 1.0 / (1.0 + DecorrelationScale)**2
        # Prior2 = 1.0 / (1.0 + nu/25)**2

        # If prior is zero, do not compute likelihood
        if (Prior1 == 0) or (Prior2 == 0):
            NegativeLogPrior = numpy.inf
            return NegativeLogPrior
        else:
            NegativeLogPrior = -(numpy.log(Prior1) + numpy.log(Prior2))

        # Find optimal sigma and sigma0
        Optimal_sigma,Optimal_sigma0,TraceEstimationUtilities,K = FindOptimalSigmaSigma0(x,y,z,X,UseEigenvaluesMethod,UseSparse,TraceEstimationMethod,DecorrelationScale,nu)
        # Likelihood function with minus to make maximization to a minimization
        NegativeLogLikelihood = LikelihoodEstimation.LogLikelihoodWithSigmaSigma0(z,X,K,TraceEstimationUtilities,True,[Optimal_sigma,Optimal_sigma0])

        # Posterior
        NegativeLogPosterior = NegativeLogLikelihood + NegativeLogPrior

        print("LogPosterior: %0.4f, Decorrelation: %0.4f, nu: %0.4f, Sigma: %0.4f, Sigma0: %0.4f"%(-NegativeLogPosterior,Parameters[0],Parameters[1],Optimal_sigma,Optimal_sigma0))

    elif len(Parameters) == 4:
        # When more parameters are provided, we use the full direct optimization without our method

        # Extract parameters
        DecorrelationScale = Parameters[0]
        nu = Parameters[1]
        Sigma = Parameters[2]
        Sigma0 = Parameters[3]

        # Prior probability density # SETTING
        # Prior1 = UniformPrior(DecorrelationScale,[0.1,0.3])
        # Prior2 = UniformPrior(nu,[0.5,25])
        # Prior3 = UniformPrior(Sigma,[0,1])
        # Prior4 = UniformPrior(Sigma0,[0,1])

        # Uniform prior
        Prior1 = UniformPrior(DecorrelationScale,[0,numpy.inf])
        Prior2 = UniformPrior(nu,[0,25])

        # Inverse square prior
        # Prior1 = UniformPrior(DecorrelationScale,[0,numpy.inf]) / (1.0 + DecorrelationScale)**2
        # Prior2 = UniformPrior(nu,[0,numpy.inf]) / (1.0 + nu/25)**2

        Prior3 = UniformPrior(Sigma,[0,numpy.inf])
        Prior4 = UniformPrior(Sigma0,[0,numpy.inf])

        # If prior is zero, do not compute likelihood
        if (Prior1 == 0) or (Prior2 == 0) or (Prior3 == 0) or (Prior4 == 0):
            NegativeLogPrior = numpy.inf
            return NegativeLogPrior
        else:
            NegativeLogPrior = -(numpy.log(Prior1) + numpy.log(Prior2) + numpy.log(Prior3) + numpy.log(Prior4))

        # Obtain correlation
        K = Data.GenerateCorrelationMatrix(x,y,z,DecorrelationScale,nu,UseSparse)

        # Trace estimation utilities
        TraceEstimationUtilities = \
        {
            'UseEigenvaluesMethod': False
        }

        # Likelihood function with minus to make maximization to a minimization
        NegativeLogLikelihood = LikelihoodEstimation.LogLikelihoodWithSigmaSigma0(z,X,K,TraceEstimationUtilities,True,[Sigma,Sigma0])

        # Posterior
        NegativeLogPosterior = NegativeLogLikelihood + NegativeLogPrior

        print("LogPosterior: %0.4f, Decorrelation: %0.4f, nu: %0.4f, Sigma: %0.4f, Sigma0: %0.4f"%(-NegativeLogPosterior,Parameters[0],Parameters[1],Parameters[2],Parameters[3]))

    else:
        raise ValueError('Parameter is not recognized.')

    return NegativeLogPosterior

# ===================
# Minimize Terminated
# ===================

class MinimizeTerminated(Exception):
    """
    This class is a python exception class to raise when the MinimizeTerminator is terminated.
    In a try-exception clause, this class is cought.
    """

    def __init__(self,*args,**kwargs):
        super(MinimizeTerminated,self).__init__(*args)

# ===================
# Minimize Terminator
# ===================

class MinimizeTerminator(object):
    """
    The scipy.optimize.minimize does not terminate when setting its tolerances with tol, xatol, and fatol.
    Rather, its algorithm runs over all iterations till maxiter is reached, which passes way below the specified tolerance.

    To fix this issue, I tried to use its callack function to manually terminate the algorithm. If the callback function
    returns True, according to documentation, it should terminate the algorithm. However, it seems (in a github issue thread) 
    that this feature is not implemented, ie., the callback is useless.

    To fix the latter issue, this class is written. It stores iterations, self.coounter, as member data.
    Its __call__() function is passed to the callback of scipy.optimize.minimize. It updates the state vector xk in
    self.xk, and compares it to the previous stored state vector to calculate the error, self.Error.
    If all the entries of the self.Error vector are below the tolerance, it raises an exception. The exception causes the
    algorithm to terminate. To prevent the excpetion to terminate the whole script, the algorithm should be inside a try,except
    clause to catch the exception and terminate it gracefully.
     
    Often, the algorithm passes xk the same as previous state vector, which than makes the self.Error to be absolute zero.
    To ignore these false errors, we check if self.Error > 0 to leave the false errors out.
    """

    def __init__(self,Tolerance,Verbose):

        # Member data
        self.Counter = 0
        self.Tolerance = Tolerance
        self.StateVector = None
        self.Error = numpy.inf
        self.Converged = False
        self.Verbose = Verbose

    def GetCounter(self):
        return self.Counter

    def GetStateVector(self):
        return self.StateVector

    def __call__(self,CurrentStateVector,*args,**kwargs):
        if self.StateVector is None:
            self.StateVector = CurrentStateVector
            self.Counter += 1
        else:
            if self.Converged == False:
                # self.Error = numpy.abs(CurrentStateVector - self.StateVector)                   # Absolute error
                self.Error = numpy.abs((CurrentStateVector - self.StateVector)/self.StateVector)  # Relative error
                self.StateVector = CurrentStateVector
                self.Counter += 1

                if self.Verbose == True:
                    print('Convergence error: %s'%(', '.join(str(e) for e in self.Error.tolist())))

                if numpy.all(self.Error < self.Tolerance) and numpy.all(self.Error > 0):
                    self.Converged = True
                    raise MinimizeTerminated('Convergence error reached the tolerance at %d iterations.'%(self.Counter))

# ==================================
# Find Optimal Covariance Parameters
# ==================================

def FindOptimalCovarianceParameters(ResultsFilename):

    # Generate noisy data
    NumPoints = 30
    NoiseMagnitude = 0.2
    GridOfPoints = True
    UseSparse = False

    # Basis functions
    # BasisFunctionsType = 'Polynomial-2-Trigonometric-1'
    # BasisFunctionsType = 'Polynomial-5'
    # BasisFunctionsType = 'Polynomial-4'
    # BasisFunctionsType = 'Polynomial-3'
    BasisFunctionsType = 'Polynomial-2'
    # BasisFunctionsType = 'Polynomial-1'
    # BasisFunctionsType = 'Polynomial-0'

    # Trace estimation method
    UseEigenvaluesMethod = True    # If set to True, it overrides the interpolation estimation methods
    # TraceEstimationMethod = 'NonOrthogonalFunctionsMethod'   # highest condtion number
    # TraceEstimationMethod = 'OrthogonalFunctionsMethod'      # still high condition number
    TraceEstimationMethod = 'OrthogonalFunctionsMethod2'     # best (lowest) condition number
    # TraceEstimationMethod = 'RBFMethod'

    LogLikelihood_PartialFunction = partial( \
            PartialLikelihoodFunction, \
            NumPoints,NoiseMagnitude,GridOfPoints,BasisFunctionsType,UseEigenvaluesMethod,TraceEstimationMethod,UseSparse)

    # Guesses for the search parameters
    UseDirectMethod = True   # SETTING
    if UseDirectMethod == True:

        # uses Direct method, optimizing over the space of 4 parameters
        Guess_DecorrelationScale = 0.1
        Guess_nu = 1
        Guess_Sigma0 = 0.05
        Guess_Sigma = 0.05
        GuessParameters = [Guess_DecorrelationScale,Guess_nu,Guess_Sigma,Guess_Sigma0]
        Bounds = [(0.1,0.3),(0.5,25),(0.001,1),(0.001,1)]

    else:

        # uses our method, optimizing over the space of two parameters
        Guess_DecorrelationScale = 0.1
        Guess_nu = 1
        GuessParameters = [Guess_DecorrelationScale,Guess_nu]
        Bounds = [(0.1,0.3),(0.5,25)]

    # Local optimization settings
    # Method = 'BFGS'
    # Method = 'L-BFGS-B'
    # Method = 'SLSQP'
    # Method = 'trust-constr'
    # Method = 'CG'
    Method = 'Nelder-Mead'
    Tolerance = 1e-4

    # Minimize Terminator to gracefully terminate scipy.optimize.minimize once tolerance is reached
    MinimizeTerminatorObj = MinimizeTerminator(Tolerance,Verbose=True)

    # Optimization methods
    time0 = time.process_time()
    try:
        # Local optimization method (use for both direct and presented method)
        # Res = scipy.optimize.minimize(LogLikelihood_PartialFunction,GuessParameters,method=Method,tol=Tolerance,
        #         callback=MinimizeTerminatorObj.__call__,
        #         options={'maxiter':1000,'xatol':Tolerance,'fatol':Tolerance,'disp':True})

        # Global optimization methods (use for direct method)
        numpy.random.seed(31)   # for repeatability of results
        Res = scipy.optimize.differential_evolution(LogLikelihood_PartialFunction,Bounds,workers=-1,tol=Tolerance,atol=Tolerance,
                updating='deferred',polish=True,strategy='best1bin',popsize=50,maxiter=200) # Works well
        # Res = scipy.optimize.dual_annealing(LogLikelihood_PartialFunction,Bounds,maxiter=500)
        # Res = scipy.optimize.shgo(LogLikelihood_PartialFunction,Bounds,
        #         options={'minimize_every_iter': True,'local_iter': True,'minimizer_kwargs':{'method': 'Nelder-Mead'}})
        # Res = scipy.optimize.basinhopping(LogLikelihood_PartialFunction,x0=GuessParameters)

        # Extract results from Res output
        StateVector = Res.x
        max_lp = -Res.fun
        Iterations = Res.nit
        Message = Res.message
        Success = Res.success

        print(Res)

        # Brute Force optimization method (use for direct method)
        # rranges = ((0.1,0.3),(0.5,25))
        # Res = scipy.optimize.brute(LogLikelihood_PartialFunction,ranges=rranges,full_output=True,finish=scipy.optimize.fmin,workers=-1,Ns=30)
        # Optimal_DecorrelationScale = Res[0][0]
        # Optimal_nu = Res[0][1]
        # max_lp = -Res[1]
        # Iterations = None
        # Message = "Using bute force"
        # Sucess = True

    except MinimizeTerminated:

        # Extract results from MinimizeTerminator
        StateVector = MinimizeTerminatorObj.GetStateVector()
        max_lp = -LogLikelihood_PartialFunction(StateVector)
        Iterations = MinimizeTerminatorObj.GetCounter()
        Message = 'Terminated after reaching the tolerance.'
        Success = True

        print('Minimization terminated after %d iterations.'%(Iterations))

    time1 = time.process_time()
    ElapsedTime = time1 - time0

    # Unpack state vector
    Optimal_DecorrelationScale = StateVector[0]
    Optimal_nu = StateVector[1]

    if UseDirectMethod:

        Optimal_sigma = StateVector[2]
        Optimal_sigma0 = StateVector[3]

    else:

        # Find what was the optimal sigma and sigma0
        x,y,z = Data.GenerateData(NumPoints,NoiseMagnitude,GridOfPoints)
        X = Data.GenerateLinearModelBasisFunctions(x,y,BasisFunctionsType)
        Optimal_sigma,Optimal_sigma0,TraceEstimationUtilities,K = FindOptimalSigmaSigma0(x,y,z,X,UseEigenvaluesMethod,UseSparse,TraceEstimationMethod,Optimal_DecorrelationScale,Optimal_nu)

    # Output distionary
    Results =  \
    {
        'DataSetup': \
        {
            'NumPoints': NumPoints,
            'NoiseMagnitude': NoiseMagnitude,
            'UseSparse': UseSparse,
            'BasisFunctionsType': BasisFunctionsType
        },
        'OptimizationSetup':
        {
            'UseDirectMethod': UseDirectMethod,
            'Tolerance': Tolerance,
            'GuessParameters': GuessParameters,
            'Bounds': Bounds,
        },
        'Parameters': \
        {
            'sigma': Optimal_sigma,
            'sigma0' : Optimal_sigma0,
            'DecorrelationScale': Optimal_DecorrelationScale,
            'nu': Optimal_nu,
        },
        'Convergence': \
        {
            'max_lp': max_lp,
            'Iterations': Iterations,
            'ElapsedTime': ElapsedTime,
            'Message': Message,
            'Success': Success
        }
    }

    print(Results)

    # Save the results
    with open(ResultsFilename,'wb') as handle:
        pickle.dump(Results,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to %s.'%ResultsFilename)

# ============================
# Log Likelihood Grid Function
# ============================

def LogLikelihood_GridFunction( \
        NumPoints, \
        NoiseMagnitude, \
        GridOfPoints, \
        BasisFunctionsType, \
        UseEigenvaluesMethod, \
        TraceEstimationMethod, \
        UseSparse, \
        DecorrelationScale, \
        nu, \
        Index):

    N = DecorrelationScale.size
    i = numpy.mod(Index,N)
    j = int(Index / N)

    Parameters = [DecorrelationScale[i],nu[j]]
    Lp = PartialLikelihoodFunction(NumPoints,NoiseMagnitude,GridOfPoints,BasisFunctionsType, \
            UseEigenvaluesMethod,TraceEstimationMethod,UseSparse,Parameters)

    return Lp,i,j

# =====================================
# Plot Log Likelihood Versus Parameters
# =====================================

def PlotLogLikelihoodVersusParameters(ResultsFilename,PlotFilename,PlotDataWithPrior):
    """
    This function plots the results of the "ComputeLogLikelihoodVersusParameters" function.
    """

    print('Plot results ...')

    if PlotDataWithPrior == False:
        # Plots for data without prior
        CutData = 0.92
        Clim = 0.87
    else:
        # Plots for data with prior
        CutData = numpy.inf
        Clim = None

    # Open file
    with open(ResultsFilename,'rb') as handle:
        Results = pickle.load(handle)

    DecorrelationScale = Results['DecorrelationScale']
    nu = Results['nu']
    Lp = Results['Lp']

    # Smooth the data with Gaussian filter.
    sigma = [2,2]  # in unit of data pixel size
    Lp = scipy.ndimage.filters.gaussian_filter(Lp,sigma,mode='nearest')

    # Increase resolution for better contour plot
    N = 300
    f = scipy.interpolate.interp2d(nu,DecorrelationScale,Lp,kind='cubic')
    DecorrelationScale_HighRes = numpy.linspace(DecorrelationScale[0],DecorrelationScale[-1],N)
    nu_HighRes = numpy.linspace(nu[0],nu[-1],N)
    x,y = numpy.meshgrid(DecorrelationScale_HighRes,nu_HighRes)
    Lp = f(nu_HighRes,DecorrelationScale_HighRes)

    # We will plot the difference of max of Lp to Lp, called z
    MaxLp = numpy.abs(numpy.max(Lp))
    z = MaxLp - Lp
    z[z>CutData] = CutData   # Used for plotting data without prior
    Min = numpy.min(z)
    Max = numpy.max(z)

    # Figure
    fig,ax=plt.subplots(figsize=(6.2,4.8))

    # Adjust bounds of a colormap
    def truncate_colormap(cmap, minval=0.0,maxval=1.0,n=2000):
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(numpy.linspace(minval, maxval, n)))
        return new_cmap

    # cmap = plt.get_cmap('gist_stern_r')
    # cmap = plt.get_cmap('rainbow_r')
    # cmap = plt.get_cmap('nipy_spectral_r')
    # cmap = plt.get_cmap('RdYlGn')
    # cmap = plt.get_cmap('ocean')
    # cmap = plt.get_cmap('gist_stern_r')
    # cmap = plt.get_cmap('RdYlBu')
    # cmap = plt.get_cmap('gnuplot_r')
    # cmap = plt.get_cmap('Spectral')
    cmap = plt.get_cmap('gist_earth')
    ColorMap = truncate_colormap(cmap,0,1)
    # ColorMap = truncate_colormap(cmap,0.2,0.9)  # for ocean

    # # Custom colormap
    # from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    # # colors = ["black", "darkblue", "purple", "orange", "orangered"]
    # colors = ["black", "darkblue", "mediumblue", "purple", "orange", "gold"]
    # nodes = [0.0, 0.2, 0.4, 0.75, 0.95, 1.0]
    # ColorMap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    # Contour fill Plot
    Levels = numpy.linspace(Min,Max,2000)
    c = ax.contourf(x,y,z.T,Levels,cmap=ColorMap,zorder=-9)
    cbar = fig.colorbar(c,pad=0.025)
    if Clim is not None:
        c.set_clim(0,Clim)   # Used to plot data without prior
    if PlotDataWithPrior == False:
        cbar.set_ticks([0,0.3,0.6,0.9,1])
    else:
        cbar.set_ticks([0,0.5,1,1.5,1.9])

    # Contour plot
    # Levels = numpy.r_[numpy.linspace(Max,Max+(Min-Max)*0.93,10),numpy.linspace(Max+(Min-Max)*0.968,Max,1)][::-1]

    if PlotDataWithPrior == False:
        Levels = numpy.r_[0.03,numpy.arange(0.1,0.9,0.1)]
    else:
        Levels = numpy.r_[0.05,0.15,numpy.arange(0.3,1.9,0.2)]
    c = ax.contour(x,y,z.T,Levels,colors='silver',linewidths=1)
    ax.clabel(c,inline=True,fontsize=10,fmt='%1.2f',colors='silver')
    c.monochrome = True

    # Find location of min point of the data (two options below)
    # Option I: Find max from user input data
    # Optimal_Lp = 958.306
    # Optimal_DecorrelationScale = 0.17695437557900218
    # Optimal_nu = 3.209863002872277
    # DecorrelationScale_OptimalIndex = numpy.argmin(numpy.abs(Optimal_DecorrelationScale - DecorrelationScale_HighRes))
    # nu_OptimalIndex = numpy.argmin(numpy.abs(Optimal_nu - nu_HighRes))
    # x_optimal = DecorrelationScale_HighRes[DecorrelationScale_OptimalIndex]
    # y_optimal = nu_HighRes[nu_OptimalIndex]

    # Option II: Find max from the plot data
    MaxIndex = numpy.argmin(z)
    MaxIndices = numpy.unravel_index(MaxIndex,z.shape)
    x_optimal = DecorrelationScale_HighRes[MaxIndices[0]]
    y_optimal = nu_HighRes[MaxIndices[1]]

    print('Max L: %f'%MaxLp)
    print('Optimal point at x: %f, y: %f'%(x_optimal,y_optimal))

    # Plot min point of the data
    ax.plot(x_optimal,y_optimal,marker='o',color='white',markersize=4,zorder = 100)
    if PlotDataWithPrior == False:
        # Without prior. Places text below the max point
        ax.text(x_optimal,y_optimal-0.7,r'$(\hat{\alpha},\hat{\nu})$',va='top',ha='center',zorder=100,color='white')
    else:
        # With prior. Places the text above the max point
        ax.text(x_optimal-0.006,y_optimal+0.49,r'$(\hat{\alpha},\hat{\nu})$',va='bottom',ha='center',zorder=100,color='white')

    # Axes
    ax.set_xticks(numpy.arange(0.1,0.31,0.05))
    ax.set_yticks(numpy.r_[1,numpy.arange(5,26,5)])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\nu$')
    # ax.set_yscale('log')

    if PlotDataWithPrior == False:
        # Plot data without prior. The data is likelihood
        ax.set_title('Profile Log Marginal Likelihood')
        cbar.set_label(r'$\ell_{\hat{\sigma}^2,\hat{\sigma}_0^2}(\hat{\alpha},\hat{\nu}) - \ell_{\hat{\sigma}^2,\hat{\sigma}_0^2}(\alpha,\nu)$')
    else:
        # Plot data with prior. The data is posteror
        ax.set_title('Profile Log Posterior')
        cbar.set_label(r'$\log p_{\hat{\sigma}^2,\hat{\sigma}_0^2}(\hat{\alpha},\hat{\nu}|\boldsymbol{z}) - \log p_{\hat{\sigma}^2,\hat{\sigma}_0^2}(\alpha,\nu|\boldsymbol{z})$')

    # To reduce file size, rasterize contour fill plot
    plt.gca().set_rasterization_zorder(-1)

    # Save plots
    plt.tight_layout()
    SaveDir = './doc/images/'
    SaveFilename_PDF = SaveDir + PlotFilename + '.pdf'
    SaveFilename_SVG = SaveDir + PlotFilename + '.svg'
    plt.savefig(SaveFilename_PDF,transparent=True,bbox_inches='tight')
    plt.savefig(SaveFilename_SVG,transparent=True,bbox_inches='tight')
    print('Plot saved to %s.'%(SaveFilename_PDF))
    print('Plot saved to %s.'%(SaveFilename_SVG))
    # plt.show()

# ========================================
# Compute Log Likelihood Versus Parameters
# ========================================

def ComputeLogLikelihoodVersusParameters(ResultsFilename):
    """
    This function computes the Log Likelihood over a 2D grid for varying two parameters
    of the Matern correlation function. The two parameters are the decorrelation scale, and
    the smoothness.

    The output of this function will be saved in Results dictionary.
    The Reesults dictionary can be plotted with "PlotLogLikelihoodVersusParameters" function.
    """

    # # Generate noisy data
    NumPoints = 30
    NoiseMagnitude = 0.2
    GridOfPoints = True
    UseSparse = False
    
    # Basis functions
    # BasisFunctionsType = 'Polynomial-2-Trigonometric-1'
    # BasisFunctionsType = 'Polynomial-5'
    # BasisFunctionsType = 'Polynomial-4'
    # BasisFunctionsType = 'Polynomial-3'
    BasisFunctionsType = 'Polynomial-2'
    # BasisFunctionsType = 'Polynomial-1'
    # BasisFunctionsType = 'Polynomial-0'

    # Trace estimation method
    UseEigenvaluesMethod = True    # If set to True, it overrides the interpolation estimation methods
    # TraceEstimationMethod = 'NonOrthogonalFunctionsMethod'   # highest condtion number
    # TraceEstimationMethod = 'OrthogonalFunctionsMethod'      # still high condition number
    TraceEstimationMethod = 'OrthogonalFunctionsMethod2'       # best (lowest) condition number
    # TraceEstimationMethod = 'RBFMethod'
 
    # Axes arrays # SETTING
    DecorrelationScale = numpy.linspace(0.1,0.3,61)
    nu = numpy.linspace(1,25,60)

    # Log likelihood partial function
    LogLikelihood_PartialGridFunction = partial( \
            LogLikelihood_GridFunction, \
            NumPoints,NoiseMagnitude,GridOfPoints,BasisFunctionsType,UseEigenvaluesMethod,TraceEstimationMethod,UseSparse,DecorrelationScale,nu)

    # Mesh
    Lp_Grid = numpy.zeros((DecorrelationScale.size,nu.size))

    # Parallel processing with multiprocessing
    NumProcessors = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=NumProcessors)
    NumIterations = Lp_Grid.shape[0]*Lp_Grid.shape[1]
    ChunkSize = int(NumIterations / NumProcessors)
    if ChunkSize < 1:
        ChunkSize = 1

    Iterations = range(NumIterations)
    for Lp,i,j in pool.imap_unordered(LogLikelihood_PartialGridFunction,Iterations,chunksize=ChunkSize):

        # Return back positive sign since we worked with negative Lp to convert maximization to minimization
        Lp_Grid[i,j] = -Lp 

    pool.close()

    Results = \
    {
        'DecorrelationScale': DecorrelationScale,
        'nu': nu,
        'Lp': Lp_Grid
    }

    # Save the results
    with open(ResultsFilename,'wb') as handle:
        pickle.dump(Results,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to %s.'%ResultsFilename)

# ====
# Main
# ====

if __name__ == "__main__":
    """
    Before runing this code, make sure in TraceEstimation.py, the ComputeTraceOfInverse() is set to
    LanczosQuadrature with Golub-Kahn-Lanczos method.
    """

    # Settings
    PlotDataWithPrior = False     # Plots data without prior
    # PlotDataWithPrior = True        # Plots data with prior

    # UseSavedResults = False       # Computes new results
    UseSavedResults = True          # Plots previously computed data from pickle files

    ComputePlotData = True        # If UseSavedResults is False, this computes the data of the plot
    # ComputePlotData = False         # If UseSavedResults is False, this computes optimal parameters

    # Filenames
    if PlotDataWithPrior:

        # With prior
        ResultsFilename = './doc/data/OptimalCovariance_WithPrior.pickle'
        PlotFilename = 'OptimalCovariance_WithPrior'

    else:

        # Without prior
        ResultsFilename = './doc/data/OptimalCovariance_WithoutPrior.pickle'
        PlotFilename = 'OptimalCovariance_WithoutPrior'

    # Compute or plot
    if UseSavedResults:
        
        # Plot previously generated data
        PlotLogLikelihoodVersusParameters(ResultsFilename,PlotFilename,PlotDataWithPrior)

    else:

        if ComputePlotData:

            # Generate new data for plot (may take long time)
            ComputeLogLikelihoodVersusParameters(ResultsFilename)
            PlotLogLikelihoodVersusParameters(ResultsFilename,PlotFilename,PlotDataWithPrior)

        else:

            # Find optimal parameters (may take long time)
            FindOptimalCovarianceParameters(ResultsFilename)
