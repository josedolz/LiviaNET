""" 
Copyright (c) 2016, Jose Dolz .All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.

Jose Dolz. Dec, 2016.
email: jose.dolz.upv@gmail.com
LIVIA Department, ETS, Montreal.
"""

import pdb
import os
import numpy as np
import theano
import theano.tensor as T

import sys

# https://github.com/Theano/Theano/issues/689
sys.setrecursionlimit(50000)


#####################################################
## Various activation functions for the CNN layers ##
#####################################################

# Sigmoid activations
def applyActivationFunction_Sigmoid(inputData):
    """ inputData is a tensor5D with shape:
      (batchSize,
       Number of feature Maps,
       convolvedImageShape[0],
       convolvedImageShape[1],
       convolvedImageShape[2]) """
    
    outputData = T.nnet.sigmoid(inputData)
    return ( outputData )

# Tanh activations
def applyActivationFunction_Tanh(inputData):
    """inputData is a tensor5D with shape:
    # (batchSize,
    # Number of feature Maps,
    # convolvedImageShape[0],
    # convolvedImageShape[1],
    # convolvedImageShape[2])"""
    
    outputData= T.tanh(inputData)
    return ( outputData )

# *** There actually exist several ways to implement ReLU activations ***
# --- Version 1 ---  
def applyActivationFunction_ReLU_v1(inputData):
    """ inputData is a tensor5D with shape:
    # (batchSize,
    # Number of feature Maps,
    # convolvedImageShape[0],
    # convolvedImageShape[1],
    # convolvedImageShape[2]) """
    
    return T.maximum(inputData,0)

# --- Version 2 ---    
def applyActivationFunction_ReLU_v2(inputData):

    return T.switch(inputData < 0., 0., inputData)
 
# --- Version 3 ---
def applyActivationFunction_ReLU_v3(inputData):

    return ((inputData + abs(inputData))/2.0)
    
# --- Version 4 ---
def applyActivationFunction_ReLU_v4(inputData):
    
    return (T.sgn(inputData) + 1) * inputData * 0.5    

# *** LeakyReLU *** 
def applyActivationFunction_LeakyReLU( inputData, leakiness ) :
    """leakiness : float
        Slope for negative input, usually between 0 and 1.
        A leakiness of 0 will lead to the standard rectifier,
        a leakiness of 1 will lead to a linear activation function,
        and any value in between will give a leaky rectifier.
        
        [1] Maas et al. (2013):
        Rectifier Nonlinearities Improve Neural Network Acoustic Models,
        http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
       
    
    - The input is a tensor of shape (batchSize, FeatMaps, xDim, yDim, zDim) """

    pos = 0.5 * (1 + leakiness)
    neg = 0.5 * (1 - leakiness)
    
    output = pos * inputData + neg * abs(inputData)
 
    return (output)

# *** There actually exist several ways to implement PReLU activations ***

# PReLU activations (from Kamnitsas)
def applyActivationFunction_PReLU( inputData, PreluActivations ) :
    """Parametric Rectified Linear Unit.
    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alpha` is a learned array with the same shape as x.
    
    - The input is a tensor of shape (batchSize, FeatMaps, xDim, yDim, zDim) """
    preluActivationsAsRow = PreluActivations.dimshuffle('x', 0, 'x', 'x', 'x')

    pos = T.maximum(0, inputData)
    neg = preluActivationsAsRow * (inputData - abs(inputData)) * 0.5
    output = pos + neg

    return (output)

# --- version 2 ---
def applyActivationFunction_PReLU_v2(inputData,PreluActivations) :
    """ inputData is a tensor5D with shape:
     (batchSize,
     Number of feature Maps,
     convolvedImageShape[0],
     convolvedImageShape[1],
     convolvedImageShape[2]) """ 

    # The input is a tensor of shape (batchSize, FeatMaps, xDim, yDim, zDim)
    preluActivationsAsRow = PreluActivations.dimshuffle('x', 0, 'x', 'x', 'x')
    
    pos = ((inputData + abs(inputData)) / 2.0 )
    neg = preluActivationsAsRow * ((inputData - abs(inputData)) / 2.0 )
    output = pos + neg

    return ( output)

# --- version 3 ---
def applyActivationFunction_PReLU_v3(inputData,PreluActivations) :
    """ inputData is a tensor5D with shape:
     (batchSize,
     Number of feature Maps,
     convolvedImageShape[0],
     convolvedImageShape[1],
     convolvedImageShape[2]) """ 

    # The input is a tensor of shape (batchSize, FeatMaps, xDim, yDim, zDim)
    preluActivationsAsRow = PreluActivations.dimshuffle('x', 0, 'x', 'x', 'x')
    
    pos = 0.5 * (1 + preluActivationsAsRow )
    neg = 0.5 * (1 - preluActivationsAsRow )
    output = pos * inputData + neg * abs(inputData)

    return ( output)

# Benchmark on ReLU/PReLU activations:
# http://gforge.se/2015/06/benchmarking-relu-and-prelu/
    
# TODO. Implement some other activation functions:
# Ex: Randomized ReLU
#     S-shape Relu
#     ThresholdedReLU
