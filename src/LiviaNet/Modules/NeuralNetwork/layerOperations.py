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

import theano.tensor as T
import theano
import random
import numpy as np

# ----------------- Apply dropout to a given input ---------------#
def apply_Dropout(rng, dropoutRate, inputShape, inputData, task) :
    """ Task:
    #    0: Training
    #    1: Validation
    #    2: Testing """
    outputData = inputData
    
    if dropoutRate > 0.001 : 
        activationRate = (1-dropoutRate)
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        dropoutMask = srng.binomial(n=1, size=inputShape, p=activationRate, dtype=theano.config.floatX)
        if task == 0:
            outputData = inputData * dropoutMask
        else:
            outputData = inputData * activationRate
    return (outputData)

""" Another dropout version """
""" def applyDropout(rng, inputLayer, inputLayerSize, dropoutRate) :
    # https://iamtrask.github.io/2015/07/28/dropout/
    # https://github.com/mdenil/dropout/blob/master/mlp.py
    
    #srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    #dropoutMask = srng.binomial(n=1, p= 1-dropoutRate, size=inputLayerSize, dtype=theano.config.floatX)
    dropoutMask = numpy.random.binomial([numpy.ones((inputLayer.W.eval().shape))],1-dropoutRate)[0] * (1.0/(1-dropoutRate))
    output = inputLayer.W * dropoutMask
    return (output)"""
    
# ----------------- Convolve an input with a given kernel ---------------#
def convolveWithKernel(W, filter_shape, inputSample, inputSampleShape) :
    wReshapedForConv = W.dimshuffle(0,4,1,2,3)
    wReshapedForConvShape = (filter_shape[0], filter_shape[4], filter_shape[1], filter_shape[2], filter_shape[3])

    #Reshape image for what conv3d2d needs:
    inputSampleReshaped = inputSample.dimshuffle(0, 4, 1, 2, 3)
    inputSampleReshapedShape = (inputSampleShape[0],
                                inputSampleShape[4],
                                inputSampleShape[1],
                                inputSampleShape[2],
                                inputSampleShape[3]) 
    
    convolved_Output = T.nnet.conv3d2d.conv3d(inputSampleReshaped, 
                                              wReshapedForConv,
                                              inputSampleReshapedShape, 
                                              wReshapedForConvShape,
                                              border_mode = 'valid')
                                        
    output = convolved_Output.dimshuffle(0, 2, 3, 4, 1) 

    outputShape = [inputSampleShape[0],
                   filter_shape[0],
                   inputSampleShape[2]-filter_shape[2]+1,
                   inputSampleShape[3]-filter_shape[3]+1,
                   inputSampleShape[4]-filter_shape[4]+1]

    return (output, outputShape)

# ----------------- Apply Batch normalization ---------------#
""" Apply Batch normalization """
""" From Kamnitsas """
def applyBn(numberEpochApplyRolling, inputTrain, inputTest, inputShapeTrain) :
    numberOfChannels = inputShapeTrain[1]
    
    gBn_values = np.ones( (numberOfChannels), dtype = 'float32' )
    gBn = theano.shared(value=gBn_values, borrow=True)
    bBn_values = np.zeros( (numberOfChannels), dtype = 'float32')
    bBn = theano.shared(value=bBn_values, borrow=True)
    
    # For rolling average:
    muArray = theano.shared(np.zeros( (numberEpochApplyRolling, numberOfChannels), dtype = 'float32' ), borrow=True)
    varArray = theano.shared(np.ones( (numberEpochApplyRolling, numberOfChannels), dtype = 'float32' ), borrow=True)
    sharedNewMu_B = theano.shared(np.zeros( (numberOfChannels), dtype = 'float32'), borrow=True)
    sharedNewVar_B = theano.shared(np.ones( (numberOfChannels), dtype = 'float32'), borrow=True)
    
    e1 = np.finfo(np.float32).tiny 

    mu_B = inputTrain.mean(axis=[0,2,3,4]) 
    mu_B = T.unbroadcast(mu_B, (0)) 
    var_B = inputTrain.var(axis=[0,2,3,4])
    var_B = T.unbroadcast(var_B, (0))
    var_B_plusE = var_B + e1
    
    #---computing mu and var for inference from rolling average---
    mu_RollingAverage = muArray.mean(axis=0)
    effectiveSize = inputShapeTrain[0]*inputShapeTrain[2]*inputShapeTrain[3]*inputShapeTrain[4]
    var_RollingAverage = (effectiveSize/(effectiveSize-1))*varArray.mean(axis=0)
    var_RollingAverage_plusE = var_RollingAverage + e1
    
    # training
    normXi_train = (inputTrain - mu_B.dimshuffle('x', 0, 'x', 'x', 'x')) /  T.sqrt(var_B_plusE.dimshuffle('x', 0, 'x', 'x', 'x')) 
    normYi_train = gBn.dimshuffle('x', 0, 'x', 'x', 'x') * normXi_train + bBn.dimshuffle('x', 0, 'x', 'x', 'x') 

    # testing
    normXi_test = (inputTest - mu_RollingAverage.dimshuffle('x', 0, 'x', 'x', 'x')) /  T.sqrt(var_RollingAverage_plusE.dimshuffle('x', 0, 'x', 'x', 'x')) 
    normYi_test = gBn.dimshuffle('x', 0, 'x', 'x', 'x') * normXi_test + bBn.dimshuffle('x', 0, 'x', 'x', 'x')
    
    return (normYi_train,
            normYi_test,
            gBn,
            bBn,
            muArray,
            varArray,
            sharedNewMu_B,
            sharedNewVar_B,
            mu_B, 
            var_B
            )


# ----------------- Apply Softmax ---------------#
def applySoftMax( inputSample, inputSampleShape, numClasses, softmaxTemperature):
   
    inputSampleReshaped = inputSample.dimshuffle(0, 2, 3, 4, 1) 
    inputSampleFlattened = inputSampleReshaped.flatten(1) 

    numClassifiedVoxels = inputSampleShape[2]*inputSampleShape[3]*inputSampleShape[4]
    firstDimOfinputSample2d = inputSampleShape[0]*numClassifiedVoxels
    inputSample2d = inputSampleFlattened.reshape((firstDimOfinputSample2d, numClasses)) 

    # Predicted probability per class.
    p_y_given_x_2d = T.nnet.softmax(inputSample2d/softmaxTemperature)
    
    p_y_given_x_class = p_y_given_x_2d.reshape((inputSampleShape[0],
                                                inputSampleShape[2],
                                                inputSampleShape[3],
                                                inputSampleShape[4],
                                                inputSampleShape[1]))
                                                    
    p_y_given_x = p_y_given_x_class.dimshuffle(0,4,1,2,3) 

    y_pred = T.argmax(p_y_given_x, axis=1) 
    
    return ( p_y_given_x, y_pred )

# ----------------- Apply Bias to feat maps ---------------#                
def applyBiasToFeatureMaps( bias, featMaps ) :
    featMaps = featMaps + bias.dimshuffle('x', 0, 'x', 'x', 'x')
    
    return (featMaps)

