#! /usr/bin/env python

"""
Notes:

    Run this script with exact computation of Trace. Do not use trace estimation, like the interpolation method.
    This is because at low noise values, the plot that will be produced will be distirted.
    However, if the trace estimation is used, the results are still valid, but not for vert small
    noise magnitudes.

    To use exact computation of eigenvalues, either:

    1. Set UseEigenvaluesMethod to True.

    2. Or, (This is faster) compute trace directly (without eigenvalues method). Set UseEigenvaluesMethod to False,
       and in LikelihoodEstimation.py, replace

            TraceKninv = TraceEstimation.EstimateTrace(TraceEstimationUtilities,eta)

       with

            TraceKninv = TraceEstimation.ComputeTraceOfInverse(Kn)

       Then, in TraceEstimation.py, make sure the ComputeTraceOfInverse uses Cholesky method.
       The Cholesky method may or may not use UseInverse. Both works.

    The plot in the paper is produced by NumPoints = 50, Decorrleation scale = 0.1.
"""

# ======
# Import
# ======

import numpy
import pickle
from functools import partial
import multiprocessing

# Classes, Files
import Data
from LikelihoodEstimation import LikelihoodEstimation
from TraceEstimation import TraceEstimation
from PlotSettings import *

# ====================================
# Compute Sigma Simga0 Eta Per Process
# ====================================

def ComputeSigmaSigma0EtaPerProcess(NoiseMagnitudes,BasisFunctionsTypes,j,i):
    """
    This function computes sigma, sigma0 and eta for a given data.
    This function is intedned to be called on a single process, in parallel computation.
    """

    # Generate noisy data
    GridOfPoints = True
    NumPoints = 50
    x,y,z = Data.GenerateData(NumPoints,NoiseMagnitudes[i],GridOfPoints)

    # Generate Correlation Matrix
    DecorrelationScale = 0.1
    nu = 0.5
    UseSparse = False
    K = Data.GenerateCorrelationMatrix(x,y,z,DecorrelationScale,nu,UseSparse)

    X = Data.GenerateLinearModelBasisFunctions(x,y,BasisFunctionsTypes[j])

    # Trace estimation weights
    UseEigenvaluesMethod = True    # If set to True, it overrides the interpolation estimation methods
    # TraceEstimationMethod = 'NonOrthogonalFunctionsMethod'   # highest condtion number
    # TraceEstimationMethod = 'OrthogonalFunctionsMethod'      # still high condition number
    TraceEstimationMethod = 'OrthogonalFunctionsMethod2'       # best (lowest) condition number
    # TraceEstimationMethod = 'RBFMethod'

    # Use Derivative Method
    UseDerivativeMethod = True   # SETTING
    if UseDerivativeMethod:
        # Precompute trace interpolation function
        TraceEstimationUtilities = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,None,[1e-4,1e-3,1e-2,1e-1,1,1e+1,4e+1,1e+2,1e+3])   # SETTING

        # Finding optimal parameters with derivative of likelihood
        Interval_eta = [1e-4,1e+3]  # SETTING
        Results = LikelihoodEstimation.FindZeroOfLogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities,Interval_eta)

    else:

        # Use Direct method

        # Use Eigenvalues Method
        if UseEigenvaluesMethod == True:
            if UseSparse:

                n = K.shape[0]
                K_eigenvalues = numpy.zeros(n)

                # find 90% of eigenvalues and assume the rest are very close to zero.
                NumNoneZeroEig = int(n*0.9)
                K_eigenvalues[:NumNoneZeroEig] = scipy.sparse.linalg.eigsh(K,NumNoneZeroEig,which='LM',tol=1e-3,return_eigenvectors=False)

            else:
                K_eigenvalues = scipy.linalg.eigh(K,eigvals_only=True,check_finite=False)
            EigenvaluesMethodUtilities = \
            {
                'K_eigenvalues': K_eigenvalues
            }
        else:
            EigenvaluesMethodUtilities = {}

        TraceEstimationUtilities = \
        {
            'UseEigenvaluesMethod': UseEigenvaluesMethod,
            'EstimationMethod': EstimationMethod,
            'EigenvaluesMethodUtilities': EigenvaluesMethodUtilities,
            'NonOrthogonalFunctionsMethodUtilities': {},
            'OrthogonalFunctionsMethodUtilities': {},
            'OrthogonalFunctionsMethodUtilities2': {},
            'RBFMethodUtilities': {},
            'AuxilliaryEstimationMethodUtilities': {}
        }

        # Finding optimal parameters with maximum likelihood using parameters (sigma,sigma0)
        Results = LikelihoodEstimation.MaximizeLogLikelihoodWithSigmaSigma0(z,X,K,TraceEstimationUtilities)

    print('i: %d, Noise: %0.4f, Results: %s'%(i,NoiseMagnitudes[i],Results))

    return NoiseMagnitudes[i],Results['sigma'],Results['sigma0'],Results['eta'],i

# ============================================
# Compare Computation With Various Noise Level
# ============================================

def CompareComputationWithVariousNoiselevel(ResultsFilename):

    # Vary noise magnitude. Noise magnitude is sigma0
    NoiseMagnitudes = numpy.logspace(-2,1,200)

    # Generate basis functions
    BasisFunctionsTypes = [
    'Polynomial-0',
    'Polynomial-2',
    'Polynomial-4',
    'Polynomial-2-Trigonometric-1']

    AllResults = []

    for j in range(len(BasisFunctionsTypes)):

        # Partial function
        ComputeSigmaSigma0EtaPerProcess_PartialFunction = partial( \
                ComputeSigmaSigma0EtaPerProcess, \
                NoiseMagnitudes,BasisFunctionsTypes,j)

        # Output results
        NoiseMagnitudeArray = numpy.empty((NoiseMagnitudes.size,))
        sigmaArray = numpy.empty((NoiseMagnitudes.size,))
        sigma0Array = numpy.empty((NoiseMagnitudes.size,))
        etaArray = numpy.empty((NoiseMagnitudes.size,))

        # Parallel processing with multiprocessing
        NumProcessors = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=NumProcessors)
        NumIterations = NoiseMagnitudes.size
        ChunkSize = int(NumIterations / (3.0*NumProcessors))
        if ChunkSize < 1:
            ChunkSize = 1

        Iterations = range(NumIterations)

        # Loop through noise magnitudes
        for NoiseMagnitude,Sigma,Sigma0,Eta,i in pool.imap_unordered(ComputeSigmaSigma0EtaPerProcess_PartialFunction,Iterations,chunksize=ChunkSize):
            NoiseMagnitudeArray[i] = NoiseMagnitude
            sigmaArray[i] = Sigma
            sigma0Array[i] = Sigma0
            etaArray[i] = Eta

        pool.close()

        # NoiseMagnitude = numpy.array(NoiseMagnitudeList)
        # sigma = numpy.array(sigmaList)
        # sigma0 = numpy.array(sigma0List)
        # eta = numpy.array(etaList)

        Results = \
        {
            'NoiseMagnitude': NoiseMagnitudeArray,
            'sigma': sigmaArray,
            'sigma0': sigma0Array,
            'eta': etaArray
        }

        AllResults.append(Results)

    # Save the results
    with open(ResultsFilename,'wb') as handle:
        pickle.dump(AllResults,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to %s.'%ResultsFilename)

    # Plot
    PlotResults(AllResults)

# ============
# Plot Results
# ============

def PlotResults(AllResults):
    """
    Plots sigma hat, sigma0 hat and eta hat versus noise magnitude sigma0.
    """

    print('Plotting results ...')

    NumPlots = len(AllResults)
    Colors1 = sns.color_palette("OrRd_d",NumPlots)[::-1]
    Colors2 = sns.color_palette("YlGn_d",NumPlots)[::-1]
    Colors3 = sns.color_palette("PuBuGn_d",NumPlots)[::-1]

    fig,ax = plt.subplots(figsize=(8.8,4.2))
    ax2 = ax.twinx()

    PlotHandles = []

    for i in range(NumPlots):
        p, = ax.semilogx(AllResults[i]['NoiseMagnitude'],AllResults[i]['sigma0']/AllResults[i]['NoiseMagnitude'],label=r'$\hat{\sigma}_0/\sigma$',color=Colors1[i])
        # q, = ax.semilogx(AllResults[i]['NoiseMagnitude'],AllResults[i]['sigma']/AllResults[i]['NoiseMagnitude'],'--',label=r'$\hat{\sigma}/\sigma$',color=Colors2[i])
        # r, = ax2.loglog(AllResults[i]['NoiseMagnitude'],AllResults[i]['eta'],label=r'$\eta$',color=Colors3[i])
        r, = ax2.semilogx(AllResults[i]['NoiseMagnitude'],AllResults[i]['sigma']/AllResults[i]['NoiseMagnitude'],label=r'$\eta$',color=Colors3[i])
        # PlotHandles.append([p,q,r])
        PlotHandles.append([p,r])

    p11 = PlotHandles[0][0]
    p12 = PlotHandles[1][0]
    p13 = PlotHandles[2][0]
    p14 = PlotHandles[3][0]
    # p15 = PlotHandles[4][0]
    p21 = PlotHandles[0][1]
    p22 = PlotHandles[1][1]
    p23 = PlotHandles[2][1]
    p24 = PlotHandles[3][1]
    # p25 = PlotHandles[4][1]

    # Axes
    Red = 'maroon'
    ax.set_xlabel(r'Data noise level, $\sigma_0$')
    # ax.set_ylabel(r'$\{\hat{\sigma}$,$\hat{\sigma}_0\}/\sigma_0$')
    ax.set_ylabel(r'$\hat{\sigma}_0/\sigma_0$',color=Red)
    ax.set_title(r'Estimates $\hat{\sigma}_0$, $\hat{\sigma}$ versus data noise level')
    ax.set_xlim([AllResults[0]['NoiseMagnitude'][0],AllResults[0]['NoiseMagnitude'][-1]])
    # ax.legend(frameon=False)

    ax.set_ylim([0,1.2]) 
    ax.set_yticks([0,1,1.2],)
    ax.tick_params(axis='y',labelcolor=Red)
    # ax.grid(True,axis='y')
    ax.grid(True)

    # Twin Axes
    # Blue = '#277995'
    Blue = '#20647B'
    ax2.set_ylabel(r'$\hat{\sigma}/\sigma_0$',color=Blue)
    ax2.tick_params(axis='y')
    ax2.set_ylim([0,15])
    ax2.set_yticks([0,5,10,15])
    ax2.tick_params(axis='y',labelcolor=Blue)
    # ax2.set_ylim([1e-1,1e+4])

    # create blank rectangle
    EmptyHandle = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    EmptyLabel = ""

    # Prepare Legend handles and labels
    legend_handles = [EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle,p11,p12,p13,p14,EmptyHandle,p21,p22,p23,p24]
    legend_labels = [EmptyLabel,r'Polynomial, $0^{\mathrm{th}}$ order:',r'Polynomial, $2^{\mathrm{nd}}$ order:',r'Polynomial, $4^{\mathrm{th}}$ order:',r'Trigonometric:',r'$\hat{\sigma}_0/\sigma_0$',EmptyLabel,EmptyLabel,EmptyLabel,EmptyLabel,r'$\hat{\sigma}/\sigma_0$',EmptyLabel,EmptyLabel,EmptyLabel,EmptyLabel]

    # Plot legends
    legend = ax.legend(legend_handles,legend_labels,frameon=False,fontsize='x-small',ncol=3,loc='upper left',handletextpad=-2,bbox_to_anchor=(1.2,1.04),labelspacing=1)
    legend._legend_box.align = "left"

    # Align label and legend line vertically
    shift = 2
    Increment = 1.5
    legend.texts[1].set_position((0,shift))
    legend.texts[2].set_position((0,shift+Increment))
    legend.texts[3].set_position((0,shift+Increment*2))
    legend.texts[4].set_position((0,shift+Increment*2))

    # Save plots
    plt.tight_layout()
    SaveDir = './doc/images/'
    SaveFilename = 'NoiseLevel'
    SaveFilename_PDF = SaveDir + SaveFilename + '.pdf'
    SaveFilename_SVG = SaveDir + SaveFilename + '.svg'
    plt.savefig(SaveFilename_PDF,transparent=True,bbox_inches='tight')
    plt.savefig(SaveFilename_SVG,transparent=True,bbox_inches='tight')
    print('Plot saved to %s.'%(SaveFilename_PDF))
    print('Plot saved to %s.'%(SaveFilename_SVG))
    # plt.show()

# ====
# Main
# ====

if __name__ == "__main__":

    # Setting
    # UseSavedResults = False    # Computes new reuslts
    UseSavedResults = True       # Plots previously generated data from pickle file

    # Reuslts filename
    ResultsFilename = './doc/data/NoiseLevelResults.pickle'

    # Compute or plot
    if UseSavedResults:
 
        # Load file
        print('Loading %s.'%ResultsFilename)
        with open(ResultsFilename,'rb') as handle:
            AllResults = pickle.load(handle)

        # Plot
        PlotResults(AllResults)

    else:

        # Generate new data
        CompareComputationWithVariousNoiselevel(ResultsFilename)
