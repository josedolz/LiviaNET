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


import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
import theano.tensor.nnet.conv3d2d
import pdb

import sys
import os
import numpy as np
import numpy
import random

from Modules.General.Utils import initializeWeights
from Modules.NeuralNetwork.ActivationFunctions import *
from Modules.NeuralNetwork.layerOperations import *

#################################################################
#                         Layer Types                           #
#################################################################

class LiviaNet3DConvLayer(object):
    """Convolutional Layer of the Livia network """
    def __init__(self,
                 rng,
                 layerID,
                 inputSample_Train, 
                 inputSample_Test,
                 inputToLayerShapeTrain,
                 inputToLayerShapeTest,
                 filterShape,
                 useBatchNorm, 
                 numberEpochApplyRolling, 
                 maxPoolingParameters,
                 weights_initMethodType,
                 weights,
                 activationType,
                 dropoutRate=0.0) :
        
        self.inputTrain = None
        self.inputTest = None
        self.inputShapeTrain = None
        self.inputShapeTest = None
       
        self._numberOfFeatureMaps = 0
        self._maxPoolingParameters = None
        self._appliedBnInLayer = None
        self.params = [] 
        self.W = None 
        self._gBn = None 
        self._b = None 
        self._aPrelu = None 
        self.numberOfTrainableParams = 0
        
        self.muBatchNorm = None 
        self._varBnsArrayForRollingAverage = None 
        self.numberEpochApplyRolling = numberEpochApplyRolling
        self.rollingIndex = 0 
        self._sharedNewMu_B = None 
        self._sharedNewVar_B = None
        self._newMu_B = None
        self._newVar_B = None
        
        self.outputTrain = None
        self.outputTest = None
        self.outputShapeTrain = None
        self.outputShapeTest = None

        # === After all the parameters has been initialized, create the layer 
        # Set all the inputs and parameters
        self.inputTrain = inputSample_Train
        self.inputTest = inputSample_Test
        self.inputShapeTrain = inputToLayerShapeTrain
        self.inputShapeTest = inputToLayerShapeTest
        
        self._numberOfFeatureMaps = filterShape[0] 
        assert self.inputShapeTrain[1] == filterShape[1]
        self._maxPoolingParameters = maxPoolingParameters

        print(" --- [STATUS]  --------- Creating layer {} --------- ".format(layerID))
        
         ## Process the input layer through all the steps over the block
             
        (inputToConvTrain,
         inputToConvTest) = self.passInputThroughLayerElements(inputSample_Train,
                                                               inputToLayerShapeTrain,
                                                               inputSample_Test,
                                                               inputToLayerShapeTest,
                                                               useBatchNorm, 
                                                               numberEpochApplyRolling, 
                                                               activationType,
                                                               weights,  
                                                               dropoutRate,
                                                               rng
                                                               )         
        # input shapes for the convolutions
        inputToConvShapeTrain = inputToLayerShapeTrain
        inputToConvShapeTest  = inputToLayerShapeTest
        
        # --------------  Weights initialization -------------
        # Initialize weights with random weights if W is empty
        # Otherwise, use loaded weights

        self.W = initializeWeights(filterShape,
                                   weights_initMethodType,
                                   weights)

        self.params = [self.W] + self.params
        self.numberOfTrainableParams += 1
        
        ##---------- Convolve --------------
        (convolvedOutput_Train, convolvedOutputShape_Train) = convolveWithKernel(self.W, filterShape, inputToConvTrain, inputToConvShapeTrain) 
        (convolvedOutput_Test, convolvedOutputShape_Test)   = convolveWithKernel(self.W , filterShape, inputToConvTest, inputToConvShapeTest) 
        
        self.outputTrain = convolvedOutput_Train
        self.outputTest = convolvedOutput_Test
        self.outputShapeTrain = convolvedOutputShape_Train
        self.outputShapeTest = convolvedOutputShape_Test
        
    
    def updateLayerMatricesBatchNorm(self):

        if self._appliedBnInLayer :
            muArrayValue = self.muBatchNorm.get_value()
            muArrayValue[self.rollingIndex] = self._sharedNewMu_B.get_value()
            self.muBatchNorm.set_value(muArrayValue, borrow=True)
            
            varArrayValue = self._varBnsArrayForRollingAverage.get_value()
            varArrayValue[self.rollingIndex] = self._sharedNewVar_B.get_value()
            self._varBnsArrayForRollingAverage.set_value(varArrayValue, borrow=True)
            self.rollingIndex = (self.rollingIndex + 1) % self.numberEpochApplyRolling
            
    def getUpdatesForBnRollingAverage(self) :
        if self._appliedBnInLayer :
            return [(self._sharedNewMu_B, self._newMu_B),
                    (self._sharedNewVar_B, self._newVar_B) ]
        else :
            return []
    

    def passInputThroughLayerElements(self,
                                      inputSample_Train,
                                      inputSampleShape_Train,
                                      inputSample_Test,
                                      inputSampleShape_Test,
                                      useBatchNorm,
                                      numberEpochApplyRolling,
                                      activationType,
                                      weights,
                                      dropoutRate,
                                      rndState):
        """ Through each block the following steps are applied, according to Kamnitsas:
            1 - Batch Normalization or biases
            2 - Activation function
            3 - Dropout
            4 - (Optional) Max pooling

            Ref:   He et al "Identity Mappings in Deep Residual Networks" 2016 
            https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua """
            
        # ________________________________________________________
        #      1 :  Batch Normalization 
        # ________________________________________________________
        """ Implemenation taken from Kamnitsas work.
        
        A batch normalization implementation in TensorFlow:

        http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

        "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
         Proceedings of the 32nd International Conference on Machine Learning, Lille, France, 2015.
         Journal of Machine Learning Research: W&CP volume 37
        """
        if useBatchNorm > 0 :
           
            self._appliedBnInLayer = True
            
            (inputToNonLinearityTrain,
            inputToNonLinearityTest,
            self._gBn,
            self._b,
            self.muBatchNorm,
            self._varBnsArrayForRollingAverage,
            self._sharedNewMu_B,
            self._sharedNewVar_B,
            self._newMu_B,
            self._newVar_B) = applyBn( numberEpochApplyRolling,
                                       inputSample_Train,
                                       inputSample_Test,
                                       inputSampleShape_Train)
                         
            self.params = self.params + [self._gBn, self._b]
        else : 
            self._appliedBnInLayer = False
            numberOfInputFeatMaps = inputSampleShape_Train[1]

            b_values = np.zeros( (self._numberOfFeatureMaps), dtype = 'float32')
            self._b = theano.shared(value=b_values, borrow=True)
    
            inputToNonLinearityTrain = applyBiasToFeatureMaps( self._b, inputSample_Train )
            inputToNonLinearityTest = applyBiasToFeatureMaps( self._b, inputSample_Test )

            self.params = self.params + [self._b]
            
        # ________________________________________________________
        #      2 :  Apply the corresponding activation function 
        # ________________________________________________________
        def Linear():
            print " --- Activation function: Linear"
            self.activationFunctionType = "Linear"
            output_Train = inputToNonLinearityTrain
            output_Test = inputToNonLinearityTest
            return (output_Train, output_Test)
            
        def ReLU():
            print " --- Activation function: ReLU"
            self.activationFunctionType = "ReLU"
            output_Train = applyActivationFunction_ReLU_v1(inputToNonLinearityTrain)
            output_Test = applyActivationFunction_ReLU_v1(inputToNonLinearityTest)
            return (output_Train, output_Test)
        
        def PReLU():
            print " --- Activation function: PReLU"
            self.activationFunctionType = "PReLU"
            numberOfInputFeatMaps = inputSampleShape_Train[1]
            PReLU_Values = np.ones( (numberOfInputFeatMaps), dtype = 'float32' )*0.01 
            self._aPrelu = theano.shared(value=PReLU_Values, borrow=True) 

            output_Train = applyActivationFunction_PReLU(inputToNonLinearityTrain, self._aPrelu)
            output_Test  = applyActivationFunction_PReLU(inputToNonLinearityTest, self._aPrelu)
            self.params = self.params + [self._aPrelu]
            self.numberOfTrainableParams += 1
            return (output_Train,output_Test)
        
        def LeakyReLU():
            print " --- Activation function: Leaky ReLU "
            self.activationFunctionType = "Leky ReLU"
            leakiness = 0.2 # TODO. Introduce this value in the config.ini
            output_Train = applyActivationFunction_LeakyReLU(inputToNonLinearityTrain,leakiness)
            output_Test = applyActivationFunction_LeakyReLU(inputToNonLinearityTest,leakiness)
            return (output_Train, output_Test)
                
        optionsActFunction = {0 : Linear,
                              1 : ReLU,
                              2 : PReLU,
                              3 : LeakyReLU}

        (inputToDropout_Train, inputToDropout_Test) = optionsActFunction[activationType]()
            
        # ________________________________________________________
        #      3 :  Apply Dropout
        # ________________________________________________________
        output_Train = apply_Dropout(rndState,dropoutRate,inputSampleShape_Train,inputToDropout_Train, 0)
        output_Test  = apply_Dropout(rndState,dropoutRate,inputSampleShape_Train,inputToDropout_Test, 1)
          
        # ________________________________________________________
        #      This will go as input to the convolutions
        # ________________________________________________________   

        return (output_Train, output_Test)
        
    
