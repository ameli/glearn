#! /usr/bin/env python

# ======
# Import
# ======

# Python packages
import sys
import pickle
import time
import numpy
import scipy
from scipy import linalg
from scipy import interpolate

# Classes
import Data
from LikelihoodEstimation import LikelihoodEstimation
from TraceEstimation import TraceEstimation
from PlotSettings import *

# ==========================================
# Maximize Likelihood With Derivative Method
# ==========================================

def MaximizeLikelihoodWithDerivativeMethod(NumPoints,UseSparse,DecorrelationScale):
    """
    Maximies L using the derivative of L with respect to eta.
    To compute the trace of K inverse it uses the interpolation estimation method.
    UseEigenvaluesMethod = False
    """

    # Vary noise magnitude
    ElapsedTime_Interpolation_List = []
    ElapsedTime_NoiseEstimation_List = []
    ElapsedTime_Total_List = []
    NumPoints_List = []
    for i in range(NumPoints.size):

        # Generate noisy data
        NoiseMagnitude = 0.2
        GridOfPoints = True
        x,y,z = Data.GenerateData(NumPoints[i],NoiseMagnitude,GridOfPoints)

        # Generate Correlation Matrix
        nu = 0.5
        K = Data.GenerateCorrelationMatrix(x,y,z,DecorrelationScale,nu,UseSparse)

        # Generate basis functions
        # BasisFunctionsType = 'Polynomial-2-Trigonometric-1'
        # BasisFunctionsType = 'Polynomial-5'
        # BasisFunctionsType = 'Polynomial-4'
        # BasisFunctionsType = 'Polynomial-3'
        BasisFunctionsType = 'Polynomial-2'
        # BasisFunctionsType = 'Polynomial-1'
        # BasisFunctionsType = 'Polynomial-0'
        X = Data.GenerateLinearModelBasisFunctions(x,y,BasisFunctionsType)

        # Trace estimation method
        UseEigenvaluesMethod = False    # If set to True, it overrides the interpolation estimation methods
        # TraceEstimationMethod = 'NonOrthogonalFunctionsMethod'   # highest condtion number
        # TraceEstimationMethod = 'OrthogonalFunctionsMethod'      # still high condition number
        TraceEstimationMethod = 'OrthogonalFunctionsMethod2'     # best (lowest) condition number
        # TraceEstimationMethod = 'RBFMethod'

        # Precompute trace interpolation function
        Time0 = time.process_time()
        TraceEstimationUtilities = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,TraceEstimationMethod,None,[1,1e+1,4e+1,1e+2,1e+3])   # SETTING
        Time1 = time.process_time()

        # Finding optimal parameters with derivative of likelihood
        Interval_eta = [1,1e+3]  # SETTING
        Results = LikelihoodEstimation.FindZeroOfLogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities,Interval_eta)
        Time2 = time.process_time()

        ElapsedTime_Interpolation = Time1 - Time0
        ElapsedTime_NoiseEstimation = Time2 - Time1
        ElapsedTime_Total = Time2 - Time0
        print('i: %d, NumPoints: %d, Elapsed times: %0.2f,%0.2f,%0.2f, Results: %s'%(i,NumPoints[i],ElapsedTime_Interpolation,ElapsedTime_NoiseEstimation,ElapsedTime_Total,Results))

        if ((numpy.isinf(Results['eta']) != True) and (Results['eta'] != 0)):
            if GridOfPoints == True:
                NumPoints_List.append(NumPoints[i]**2)
            else:
                NumPoints_List.append(NumPoints[i])
            ElapsedTime_Interpolation_List.append(ElapsedTime_Interpolation)
            ElapsedTime_NoiseEstimation_List.append(ElapsedTime_NoiseEstimation)
            ElapsedTime_Total_List.append(ElapsedTime_Total)

    ElapsedTime_Interpolation_Array = numpy.array(ElapsedTime_Interpolation_List)
    ElapsedTime_NoiseEstimation_Array = numpy.array(ElapsedTime_NoiseEstimation_List)
    ElapsedTime_Total_Array = numpy.array(ElapsedTime_Total_List)
    NumPoints_Array = numpy.array(NumPoints_List)

    Results = \
    {
        'ElapsedTime_Interpolation': ElapsedTime_Interpolation_Array,
        'ElapsedTime_NoiseEstimation': ElapsedTime_NoiseEstimation_Array,
        'ElapsedTime_Total': ElapsedTime_Total_Array,
        'NumPoints': NumPoints_Array
    }

    return Results

# ============================
# Maximize Likelihood Directly
# ============================

def MaximizeLikelihoodWithDirectMethod(NumPoints,UseSparse,DecorrelationScale,UseEigenvaluesMethod):
    """
    Maximizes L using the space of parameters sigma and sigma0,
    To compute determinant of K, it uses the eigenvalue method.
    """

    # Vary noise magnitude
    ElapsedTime_Interpolation_List = []
    ElapsedTime_NoiseEstimation_List = []
    ElapsedTime_Total_List = []
    NumPoints_List = []
    for i in range(NumPoints.size):

        # Generate noisy data
        NoiseMagnitude = 0.2
        GridOfPoints = True
        x,y,z = Data.GenerateData(NumPoints[i],NoiseMagnitude,GridOfPoints)

        # Generate Correlation Matrix
        nu = 0.5
        K = Data.GenerateCorrelationMatrix(x,y,z,DecorrelationScale,nu,UseSparse)

        # BasisFunctionsType = 'Polynomial-2-Trigonometric-1'
        # BasisFunctionsType = 'Polynomial-5'
        # BasisFunctionsType = 'Polynomial-4'
        # BasisFunctionsType = 'Polynomial-3'
        BasisFunctionsType = 'Polynomial-2'
        # BasisFunctionsType = 'Polynomial-1'
        # BasisFunctionsType = 'Polynomial-0'
        X = Data.GenerateLinearModelBasisFunctions(x,y,BasisFunctionsType)

        # Trace estimation method
        # TraceEstimationMethod = 'NonOrthogonalFunctionsMethod'   # highest condtion number
        # TraceEstimationMethod = 'OrthogonalFunctionsMethod'      # still high condition number
        TraceEstimationMethod = 'OrthogonalFunctionsMethod2'     # best (lowest) condition number
        # TraceEstimationMethod = 'RBFMethod'

        # Precompute trace interpolation function
        Time0 = time.process_time()
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
        Time1 = time.process_time()

        TraceEstimationUtilities = \
        {
            'UseEigenvaluesMethod': UseEigenvaluesMethod,
            'EstimationMethod': TraceEstimationMethod,
            'EigenvaluesMethodUtilities': EigenvaluesMethodUtilities,
            'NonOrthogonalFunctionsMethodUtilities': {},
            'OrthogonalFunctionsMethodUtilities': {},
            'OrthogonalFunctionsMethodUtilities2': {},
            'RBFMethodUtilities': {},
            'AuxilliaryEstimationMethodUtilities': {}
        }

        # Finding optimal parameters with maximum likelihood using parameters (sigma,sigma0)
        Results = LikelihoodEstimation.MaximizeLogLikelihoodWithSigmaSigma0(z,X,K,TraceEstimationUtilities)
        Time2 = time.process_time()

        ElapsedTime_Interpolation = Time1 - Time0
        ElapsedTime_NoiseEstimation = Time2 - Time1
        ElapsedTime_Total = Time2 - Time0
        print('i: %d, NumPoints: %d, Elapsed times: %0.2f,%0.2f,%0.2f, Results: %s'%(i,NumPoints[i],ElapsedTime_Interpolation,ElapsedTime_NoiseEstimation,ElapsedTime_Total,Results))

        if ((Results['eta'] < 1e+6) and (Results['eta'] > 1e-5) and (Results['sigma'] > 0) and Results['sigma0'] > 0):
            if GridOfPoints == True:
                NumPoints_List.append(NumPoints[i]**2)
            else:
                NumPoints_List.append(NumPoints[i])
            ElapsedTime_Interpolation_List.append(ElapsedTime_Interpolation)
            ElapsedTime_NoiseEstimation_List.append(ElapsedTime_NoiseEstimation)
            ElapsedTime_Total_List.append(ElapsedTime_Total)

    ElapsedTime_Interpolation_Array = numpy.array(ElapsedTime_Interpolation_List)
    ElapsedTime_NoiseEstimation_Array = numpy.array(ElapsedTime_NoiseEstimation_List)
    ElapsedTime_Total_Array = numpy.array(ElapsedTime_Total_List)
    NumPoints_Array = numpy.array(NumPoints_List)

    Results = \
    {
        'ElapsedTime_Interpolation': ElapsedTime_Interpolation_Array,
        'ElapsedTime_NoiseEstimation': ElapsedTime_NoiseEstimation_Array,
        'ElapsedTime_Total': ElapsedTime_Total_Array,
        'NumPoints': NumPoints_Array
    }

    return Results

# ==============
# Log Regression
# ==============

def LogRegression(x,y,IndexStart,IndexEnd,N):
    if IndexEnd == -1:
        logx = numpy.log10(x[IndexStart:])
        logy = numpy.log10(y[IndexStart:])
    else:
        logx = numpy.log10(x[IndexStart:IndexEnd+1])
        logy = numpy.log10(y[IndexStart:IndexEnd+1])
    xi = numpy.logspace(numpy.log10(x[IndexStart]),numpy.log10(x[IndexEnd]),N)

    logxi = numpy.log10(xi)

    p = numpy.polyfit(logx,logy,1)
    Slope = p[0]
    poly = numpy.poly1d(p)
    logyi = poly(logxi)
    yi = 10**(logyi)

    return Slope,xi,yi

# =================================================
# Compare Computation With Various Number Of Points
# =================================================

def CompareComputationWithVariousNumberOfPoints(ResultsFilename,UseSparse):

    if UseSparse:

        # Settings for sparse matrix
        DecorrelationScale = 0.005  # for sparse matrix
        NumPointsPowers = numpy.arange(12,20.1,0.5)    # For derivative method
        NumPointsPowers2 = numpy.arange(12,16.1,0.5)   # For direct method

    else:

        # Settings for dense matrix
        DecorrelationScale = 0.1    # for dense matrix
        # NumPointsPowers = numpy.arange(9,12.1,0.5)    # For derivative method
        # NumPointsPowers2 = numpy.arange(9,12.1,0.5)   # For direct method
        NumPointsPowers = numpy.arange(6,13.1,0.5)    # For derivative method
        NumPointsPowers2 = numpy.arange(6,13.1,0.5)   # For direct method

    # Number of points
    NumPoints = numpy.rint(numpy.sqrt(2**NumPointsPowers)).astype(int)
    NumPoints2 = numpy.rint(numpy.sqrt(2**NumPointsPowers2)).astype(int)
   
    # Computations
    DerivativeMethodResults =  MaximizeLikelihoodWithDerivativeMethod(NumPoints,UseSparse,DecorrelationScale)
    DirectMethodResults1 =  MaximizeLikelihoodWithDirectMethod(NumPoints2,UseSparse,DecorrelationScale,True)     # Use eigenvalues to compute determinant
    # DirectMethodResults1 = {}
    DirectMethodResults2 =  MaximizeLikelihoodWithDirectMethod(NumPoints2,UseSparse,DecorrelationScale,False)    # do not use eigenalues method, compute determinant with Cholesky
    # DirectMethodResults2 = {}

    # Pickle results
    Results = \
    {
            'NumPointsPowers': NumPointsPowers,
            'DerivativeMethodResults': DerivativeMethodResults,
            'DirectMethodResults1': DirectMethodResults1,
            'DirectMethodResults2': DirectMethodResults2,
    }
    with open(ResultsFilename,'wb') as handle:
        pickle.dump(Results,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to %s.'%ResultsFilename)

# ========================
# Average Multiple Results
# ========================

def AverageMultipleResults(MultipleResults):
    """
    Each run may record different runtime. To moderate the fluctuations,
    we averaga emultiple runtimes over all variables.
    """

    # -------------
    # Average Array
    # -------------

    def AverageArray(List,MethodName):

        if bool(List[0][MethodName]) == False:
            return {}

        NumData = len(List)
        if NumData == 1:
            return List[0][MethodName]

        ArrayNames = ['ElapsedTime_Interpolation','ElapsedTime_NoiseEstimation','ElapsedTime_Total']
        AveragedArraysList = []

        for i in range(len(ArrayNames)):
            
            # Get the first array
            Array = List[0][MethodName][ArrayNames[i]]

            for j in range(1,NumData):
                NewArray = List[j][MethodName][ArrayNames[i]]

                try:
                    Array = Array + NewArray
                except:
                    raise ValueError('Array size: %d, NewArray size: %d'%(Array.size,NewArray.size))

            Array = Array / NumData
            AveragedArraysList.append(Array)

        AveragedResults = \
        {
            'NumPoints': List[0][MethodName]['NumPoints'],
            'ElapsedTime_Interpolation': AveragedArraysList[0],
            'ElapsedTime_NoiseEstimation': AveragedArraysList[1],
            'ElapsedTime_Total': AveragedArraysList[2]
        }

        return AveragedResults

    # ------------

    NumResults = len(MultipleResults)

    # If ony one results, do not average
    if NumResults == 1:
        return MultipleResults[0]

    # Unpack the first results
    NumPointsPowers = MultipleResults[0]['NumPointsPowers']

    DerivativeMethodResults = AverageArray(MultipleResults,'DerivativeMethodResults')
    DirectMethodResults1    = AverageArray(MultipleResults,'DirectMethodResults1')
    DirectMethodResults2    = AverageArray(MultipleResults,'DirectMethodResults2')

    AveragedResults = \
    {
        'DerivativeMethodResults': DerivativeMethodResults,
        'DirectMethodResults1': DirectMethodResults1,
        'DirectMethodResults2': DirectMethodResults2,
    }

    return AveragedResults

# ===============
# Snap To Power 2
# ===============

def SnapToPower2(Dictionary,NumPointsPowers):
    """
    This purpose of this function is to have a cleaner plots so the data are snapped to the axis ticks.

    Dictionary has four fields,

        Dictionary['NumPoints']
        Dictionary['ElapsedTime_Interpolation']
        Dictionary['ElapsedTime_MaxLikelihood']
        Dictionary['ElapsedTime_Total']

    The Dictionary['NumPoints'] are often square integers. We find the closets power 2 (2**x) to these numbers. For example, 23**2 = 529 will be snapped to 2**9 = 512.
    Once, the closets power 2 numbers are found on Dictionary['NumPoints'], the data on the other three arrays
    will be also adjusted by linear interpolation.
    """

    # Check if dictionary is empty
    if bool(Dictionary) == False:
        return {}

    # Data on x axis
    x = Dictionary['NumPoints']
    y1 = Dictionary['ElapsedTime_Interpolation']
    y2 = Dictionary['ElapsedTime_NoiseEstimation']
    y3 = Dictionary['ElapsedTime_Total']

    # Find closets powers of 2
    x_powers = numpy.log(x) / numpy.log(2.0)

    # Snap x_powers to NumPointsPowers. These two arrays might have different lengths.
    x_new = numpy.empty(x.shape)
    for i in range(x.size):

        # Find which element in NumPointsPowers is the closest to x[i]
        MinIndex = numpy.argmin(numpy.abs(NumPointsPowers - x_powers[i]))
        x_new[i] = 2**NumPointsPowers[MinIndex]

    y1_new = scipy.interpolate.interp1d(x,y1,kind='linear',fill_value='extrapolate')(x_new)
    y2_new = scipy.interpolate.interp1d(x,y2,kind='linear',fill_value='extrapolate')(x_new)
    y3_new = scipy.interpolate.interp1d(x,y3,kind='linear',fill_value='extrapolate')(x_new)

    # Update dictionary with the new data
    Dictionary = \
    {
        'NumPoints': x_new,
        'ElapsedTime_Interpolation': y1_new,
        'ElapsedTime_NoiseEstimation': y2_new,
        'ElapsedTime_Total': y3_new
    }

    return Dictionary

# ============
# Plot Results
# ============

def PlotResults(MultipleResultsFilenames,UseSparse):

    MultipleResults = []

    # Load files
    for i in range(len(MultipleResultsFilenames)):

        print('Loading %s.'%MultipleResultsFilenames[i])
        with open(MultipleResultsFilenames[i],'rb') as handle:
            MultipleResults.append(pickle.load(handle))

    # Average Results
    AveragedResults = AverageMultipleResults(MultipleResults)

    # Unpack results
    NumPointsPowers = MultipleResults[0]['NumPointsPowers']
    DerivativeMethodResults = AveragedResults['DerivativeMethodResults']
    DirectMethodResults1 = AveragedResults['DirectMethodResults1']
    DirectMethodResults2 = AveragedResults['DirectMethodResults2']

    # Snap the NumPoints in the data from square integers to powers of 2 integers.
    DerivativeMethodResults = SnapToPower2(DerivativeMethodResults,NumPointsPowers)
    DirectMethodResults1 = SnapToPower2(DirectMethodResults1,NumPointsPowers)
    DirectMethodResults2 = SnapToPower2(DirectMethodResults2,NumPointsPowers)

    N = 100

    if UseSparse:
        # SParse matrices

        # For derivative method
        StartIndex = 6
        EndIndex = -1

        # For direct method 1
        StartIndex1 = 3
        EndIndex1 = -1

        # For direct method 2
        StartIndex2 = 3
        EndIndex2 = -1

    else:
        # Dense matrices

        # For derivative method
        StartIndex = 1
        EndIndex = -2

        # For direct method 1
        StartIndex1 = 1
        EndIndex1 = -2

        # For direct method 2
        StartIndex2 = 1
        EndIndex2 = -2

    if bool(DerivativeMethodResults):
        DerivativeMethod_Slope_1,DerivativeMethod_xi_1,DerivativeMethod_yi_1 = LogRegression(DerivativeMethodResults['NumPoints'],DerivativeMethodResults['ElapsedTime_Interpolation'],StartIndex,EndIndex,N)  # Start at 10
        DerivativeMethod_Slope_2,DerivativeMethod_xi_2,DerivativeMethod_yi_2 = LogRegression(DerivativeMethodResults['NumPoints'],DerivativeMethodResults['ElapsedTime_NoiseEstimation'],StartIndex,EndIndex,N)
        DerivativeMethod_Slope_3,DerivativeMethod_xi_3,DerivativeMethod_yi_3 = LogRegression(DerivativeMethodResults['NumPoints'],DerivativeMethodResults['ElapsedTime_Total'],StartIndex,EndIndex,N)

    if bool(DirectMethodResults1):
        DirectMethod_Slope_1,DirectMethod_xi_1,DirectMethod_yi_1 = LogRegression(DirectMethodResults1['NumPoints'],DirectMethodResults1['ElapsedTime_Interpolation'],StartIndex1,EndIndex1,N)   # Start at 19
        DirectMethod_Slope_2,DirectMethod_xi_2,DirectMethod_yi_2 = LogRegression(DirectMethodResults1['NumPoints'],DirectMethodResults1['ElapsedTime_NoiseEstimation'],StartIndex1,EndIndex1,N)
        DirectMethod_Slope_3,DirectMethod_xi_3,DirectMethod_yi_3 = LogRegression(DirectMethodResults1['NumPoints'],DirectMethodResults1['ElapsedTime_Total'],StartIndex1,EndIndex1,N)

    if bool(DirectMethodResults2):
        DirectMethod2_Slope_2,DirectMethod2_xi_2,DirectMethod2_yi_2 = LogRegression(DirectMethodResults2['NumPoints'],DirectMethodResults2['ElapsedTime_NoiseEstimation'],StartIndex2,EndIndex2,N)

    # Plots
    fig,ax = plt.subplots(figsize=(9.8,4.2))
    LineWidth = 1
    Alpha = 1

    if bool(DerivativeMethodResults):
        colors1 = sns.color_palette("OrRd_d",3)[::-1]
        p11, = ax.loglog(DerivativeMethodResults['NumPoints'],DerivativeMethodResults['ElapsedTime_Interpolation'],'-o',label='Pre-computation',markersize=3,color=colors1[0],linewidth=LineWidth,alpha=Alpha)
        p12, = ax.loglog(DerivativeMethodResults['NumPoints'],DerivativeMethodResults['ElapsedTime_NoiseEstimation'],'-o',label='Max Likelihood',markersize=3,color=colors1[1],linewidth=LineWidth,alpha=Alpha)
        p13, = ax.loglog(DerivativeMethodResults['NumPoints'],DerivativeMethodResults['ElapsedTime_Total'],'-o',label='Total',markersize=3,color=colors1[2],linewidth=LineWidth,alpha=Alpha)

        p21, = ax.loglog(DerivativeMethod_xi_1,DerivativeMethod_yi_1,'--',label='Line fit, slope: %0.2f'%(DerivativeMethod_Slope_1),color=colors1[0],zorder=20)
        p22, = ax.loglog(DerivativeMethod_xi_2,DerivativeMethod_yi_2,'--',label='Line fit, slope: %0.2f'%(DerivativeMethod_Slope_2),color=colors1[1],zorder=20)
        p23, = ax.loglog(DerivativeMethod_xi_3,DerivativeMethod_yi_3,'--',label='Line fit, slope: %0.2f'%(DerivativeMethod_Slope_3),color=colors1[2],zorder=20)

    if bool(DirectMethodResults1):
        colors2 = sns.color_palette("PuBuGn_d",3)[::-1]
        q11, = ax.loglog(DirectMethodResults1['NumPoints'],DirectMethodResults1['ElapsedTime_Interpolation'],'-o',label='Pre-computation',markersize=3,color=colors2[0],linewidth=LineWidth,alpha=Alpha)
        q12, = ax.loglog(DirectMethodResults1['NumPoints'],DirectMethodResults1['ElapsedTime_NoiseEstimation'],'-o',label='Max Likelihood',markersize=3,color=colors2[1],linewidth=LineWidth,alpha=Alpha)
        q13, = ax.loglog(DirectMethodResults1['NumPoints'],DirectMethodResults1['ElapsedTime_Total'],'-o',label='Total',markersize=3,color=colors2[2],linewidth=LineWidth,alpha=Alpha)

        q21, = ax.loglog(DirectMethod_xi_1,DirectMethod_yi_1,'--',label='Line fit, slope: %0.2f'%(DirectMethod_Slope_1),color=colors2[0],zorder=20)
        q22, = ax.loglog(DirectMethod_xi_2,DirectMethod_yi_2,'--',label='Line fit, slope: %0.2f'%(DirectMethod_Slope_2),color=colors2[1],zorder=20)
        q23, = ax.loglog(DirectMethod_xi_3,DirectMethod_yi_3,'--',label='Line fit, slope: %0.2f'%(DirectMethod_Slope_3),color=colors2[2],zorder=20)

    if bool(DirectMethodResults2):
        colors3 = 'goldenrod'
        r13, = ax.loglog(DirectMethodResults2['NumPoints'],DirectMethodResults2['ElapsedTime_NoiseEstimation'],'-o',label='Max Likelihood',markersize=3,color=colors3,linewidth=LineWidth,alpha=Alpha)
        r23, = ax.loglog(DirectMethod2_xi_2,DirectMethod2_yi_2,'--',label='Line fit, slope: %0.2f'%(DirectMethod2_Slope_2),color=colors3,zorder=20)

    # x ticks
    x_pow = numpy.arange(NumPointsPowers[0],NumPointsPowers[-1]+1)
    x_range = numpy.array(2**numpy.array(x_pow),dtype=int)
    ax.set_xticks(x_range)
    ax.set_xticklabels([r'$2^{%d}$'%y for y in x_pow])
    ax.set_xlim([x_range[0],x_range[-1]])
    ax.tick_params(axis='x',which='minor',bottom=False)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.set_xlabel(r'Number of points, $n$')
    ax.set_ylabel('CPU time (second)')
    ax.set_title('Computation time versus number of points')

    if UseSparse:
        ax.set_ylim([1e-1,1e+4])
    else:
        ax.set_ylim([1e-2,1e+3])
    # ax.grid(True,axis='x')
    ax.grid(True,axis='y')

    # create blank rectangle
    EmptyHandle = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    EmptyLabel = ""

    # First legend
    LegendHeight = 1.03
    if bool(DerivativeMethodResults):
        legend_handles1 = [EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle,p11,p12,p13,EmptyHandle,p21,p22,p23,EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle]
        legend_labels1 = [EmptyLabel,'Pre-computation:','Maximize likelihood:','All computations:','Experiment',EmptyLabel,EmptyLabel,EmptyLabel,'Line fit',EmptyLabel,EmptyLabel,EmptyLabel, 
                'Slope','%0.2f'%(DerivativeMethod_Slope_1),'%0.2f'%(DerivativeMethod_Slope_2),'%0.2f'%(DerivativeMethod_Slope_3)]

        # Plot legends
        legend1 = ax.legend(legend_handles1,legend_labels1,frameon=False,fontsize='x-small',ncol=4,loc='upper left',handletextpad=-2,bbox_to_anchor=(1.05,LegendHeight),title='Presented Method')
        legend1._legend_box.align = "left"
        ax.add_artist(legend1)

        # Update legend height for the next legend
        LegendHeight -= 0.41

    # Second legend
    if bool(DirectMethodResults1):
        legend_handles2 = [EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle,q11,q12,q13,EmptyHandle,q21,q22,q23,EmptyHandle,EmptyHandle,EmptyHandle,EmptyHandle]
        legend_labels2 = [EmptyLabel,'Pre-computation:','Maximize likelihood:','All computations:','Experiment',EmptyLabel,EmptyLabel,EmptyLabel,'Line fit',EmptyLabel,EmptyLabel,EmptyLabel,
                'Slope','%0.2f'%(DirectMethod_Slope_1),'%0.2f'%(DirectMethod_Slope_2),'%0.2f'%(DirectMethod_Slope_3)]

        # Plot legends
        legend2 = ax.legend(legend_handles2,legend_labels2,frameon=False,fontsize='x-small',ncol=4,loc='upper left',handletextpad=-2,bbox_to_anchor=(1.05,LegendHeight),title='Direct Method I')
        legend2._legend_box.align = "left"
        ax.add_artist(legend2)

        # Update legend height for the next legend
        LegendHeight -= 0.41

    # Third legend
    if bool(DirectMethodResults2):
        legend_handles3 = [EmptyHandle,EmptyHandle,EmptyHandle,r13,EmptyHandle,r23,EmptyHandle,EmptyHandle]
        legend_labels3 = [EmptyLabel,r'Maximize likelihood:','Experiment',EmptyLabel,'Line fit',EmptyLabel,'Slope','%0.2f'%(DirectMethod2_Slope_2)]

        if bool(DirectMethodResults1):
            TitleMethod2 = 'Direct Method II'
        else:
            TitleMethod2 = 'Direct Method'

        # Plot legends
        legend3 = ax.legend(legend_handles3,legend_labels3,frameon=False,fontsize='x-small',ncol=4,loc='upper left',handletextpad=-2,bbox_to_anchor=(1.05,LegendHeight),title=TitleMethod2)
        legend3._legend_box.align = "left"

    plt.tight_layout()

    # Save plots
    SaveDir = './doc/images/'
    if UseSparse:
        SaveFilename = 'ElapsedTime-Sparse'
    else:
        SaveFilename = 'ElapsedTime-Dense'
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

    UseSavedResults = False    # Computes new results
    # UseSavedResults = True       # Plots previously computed data from pickle file

    # Settings
    # UseSparse = True    # Uses sparse matrices
    UseSparse = False     # Uses dense matrices

    # Filenames
    if UseSparse:
        MultipleResultsFilenames = ['./doc/data/VariousNumberOfPoints-sparse.pickle']  # Input files for plots
        ResultsFilename = './doc/data/VariousNumberOfPoints-sparse.pickle'             # Output file for saving results

    else:
        # MultipleResultsFilenames = ['./doc/data/VariousNumberOfPoints-dense.pickle']   # Input files for plots
        # ResultsFilename = './doc/data/VariousNumberOfPoints-dense.pickle'              # Output file for saving results
        MultipleResultsFilenames = ['./doc/data/VariousNumberOfPoints-dense-test.pickle']   # Input files for plots
        ResultsFilename = './doc/data/VariousNumberOfPoints-dense-test.pickle'              # Output file for saving results

        # Averaging over multiple results files
        # MultipleResultsFilenames = [ \
                # './doc/data/VariousNumberOfPoints_1.pickle',
                # './doc/data/VariousNumberOfPoints_2.pickle',
                # './doc/data/VariousNumberOfPoints_3.pickle',
                # './doc/data/VariousNumberOfPoints_4.pickle']
   
    # Compute or plot
    if UseSavedResults:

        # Plot previously generated results (perhaps, multiple results)
        PlotResults(MultipleResultsFilenames,UseSparse)

    else:

        # Generate new data
        CompareComputationWithVariousNumberOfPoints(ResultsFilename,UseSparse)
