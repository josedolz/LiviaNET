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

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
import theano.tensor.nnet.conv3d2d

from LiviaNet3DConvLayer import LiviaNet3DConvLayer
from Modules.General.Utils import initializeWeights
from Modules.NeuralNetwork.ActivationFunctions import *
from Modules.NeuralNetwork.layerOperations import *

class LiviaSoftmax(LiviaNet3DConvLayer):
    """ Final Classification layer with Softmax """
    
    def __init__(self,
                 rng,
                 layerID,
                 inputSample_Train,
                 inputSample_Test,
                 inputToLayerShapeTrain,
                 inputToLayerShapeTest,
                 filterShape,
                 applyBatchNorm, 
                 applyBatchNormNumberEpochs, 
                 maxPoolingParameters,
                 weights_initialization,
                 weights,
                 activationType=0,
                 dropoutRate=0.0,
                 softmaxTemperature = 1.0) :

        LiviaNet3DConvLayer.__init__(self,
                                     rng,
                                     layerID,
                                     inputSample_Train,
                                     inputSample_Test,
                                     inputToLayerShapeTrain,
                                     inputToLayerShapeTest,
                                     filterShape,
                                     applyBatchNorm, 
                                     applyBatchNormNumberEpochs, 
                                     maxPoolingParameters,
                                     weights_initialization,
                                     weights,
                                     activationType,
                                     dropoutRate)
        
        self._numberOfOutputClasses = None
        self._bClassLayer = None        
        self._softmaxTemperature = None
        
        self._numberOfOutputClasses = filterShape[0]
        self._softmaxTemperature = softmaxTemperature

        # Define outputs
        outputOfConvTrain = self.outputTrain
        outputOfConvTest = self.outputTest

        # define outputs shapes
        outputOfConvShapeTrain = self.outputShapeTrain
        outputOfConvShapeTest = self.outputShapeTest
        

        # Add bias before applying the softmax
        b_values = np.zeros( (self._numberOfFeatureMaps), dtype = 'float32')
        self._bClassLayer = theano.shared(value=b_values, borrow=True)
    
        inputToSoftmaxTrain = applyBiasToFeatureMaps( self._bClassLayer, outputOfConvTrain )
        inputToSoftmaxTest = applyBiasToFeatureMaps( self._bClassLayer, outputOfConvTest ) 

        self.params = self.params + [self._bClassLayer]
        
        # ============ Apply Softmax ==============
        # Training samples
        ( self.p_y_given_x_train, self.y_pred_train ) = applySoftMax(inputToSoftmaxTrain,
                                                                     outputOfConvShapeTrain,
                                                                     self._numberOfOutputClasses,
                                                                     softmaxTemperature)

        # Testing samples
        ( self.p_y_given_x_test, self.y_pred_test ) = applySoftMax(inputToSoftmaxTest,
                                                                   outputOfConvShapeTest,
                                                                   self._numberOfOutputClasses,
                                                                   softmaxTemperature)
        
        
    def negativeLogLikelihoodWeighted(self, y, weightPerClass):      
        #Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
        e1 = np.finfo(np.float32).tiny
        addTinyProbMatrix = T.lt(self.p_y_given_x_train, 4*e1) * e1
        
        weights = weightPerClass.dimshuffle('x', 0, 'x', 'x', 'x')
        log_p_y_given_x_train = T.log(self.p_y_given_x_train + addTinyProbMatrix) 
        weighted_log_probs = log_p_y_given_x_train * weights
   
        wShape =  weighted_log_probs.shape

        # Re-arrange 
        idx0 = T.arange( wShape[0] ).dimshuffle( 0, 'x','x','x')
        idx2 = T.arange( wShape[2] ).dimshuffle('x', 0, 'x','x')
        idx3 = T.arange( wShape[3] ).dimshuffle('x','x', 0, 'x')
        idx4 = T.arange( wShape[4] ).dimshuffle('x','x','x', 0)
        
        return -T.mean( weighted_log_probs[ idx0, y, idx2, idx3, idx4] )
    
    
    def predictionProbabilities(self) :
        return self.p_y_given_x_test
