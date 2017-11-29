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

NOTES: There are still some functionalities to be implemented.

    - Add pooling layer in 3D
    - Add more activation functions
    - Add more optimizers (ex. Adam)

Jose Dolz. Dec, 2016.
email: jose.dolz.upv@gmail.com
LIVIA Department, ETS, Montreal.
"""

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import random
from math import floor
from math import ceil

from Modules.General.Utils import computeReceptiveField
from Modules.General.Utils import extendLearningRateToParams
from Modules.General.Utils import extractCenterFeatMaps
from Modules.General.Utils import getCentralVoxels
from Modules.General.Utils import getWeightsSet

import LiviaNet3DConvLayer
import LiviaSoftmax
import pdb
    
#####################################################
# ------------------------------------------------- #
##  ##  ##  ##  ##   LIVIANET 3D   ##  ##  ##  ##  ##
# ------------------------------------------------- #
#####################################################


class LiviaNet3D(object):
    def __init__(self):

        # --- containers for Theano compiled functions ----
        self.networkModel_Train = ""
        self.networkModel_Test = ""
        
        # --- shared variables will be stored in the following variables ----
        self.trainingData_x = ""
        self.testingData_x = ""
        self.trainingData_y = ""

        self.lastLayer = ""
        self.networkLayers = []
        self.intermediate_ConnectedLayers = []
             
        self.networkName = ""
        self.folderName = ""
        self.cnnLayers = []
        self.n_classes = -1

        self.sampleSize_Train = []
        self.sampleSize_Test = []
        self.kernel_Shapes = []

        self.pooling_scales = []
        self.dropout_Rates = []
        self.activationType = -1
        self.weight_Initialization = -1
        self.dropoutRates = []
        self.batch_Size = -1
        self.receptiveField = 0
        
        self.initialLearningRate = "" 
        self.learning_rate = theano.shared(np.cast["float32"](0.01))

        # Symbolic variables,
        self.inputNetwork_Train = None  
        self.inputNetwork_Test = None

        self.L1_reg_C = 0
        self.L2_reg_C = 0
        self.costFunction = 0
        
        # Params for optimizers
        self.initialMomentum = "" 
        self.momentum = theano.shared(np.cast["float32"](0.))
        self.momentumNormalized = 0
        self.momentumType = 0
        self.vel_Momentum = [] 
        self.rho_RMSProp = 0
        self.epsilon_RMSProp = 0
        self.params_RmsProp = [] 
        self.numberOfEpochsTrained = 0
        self.applyBatchNorm = ""
        self.numberEpochToApplyBatchNorm = 0
        self.softmax_Temp = 1.0

        self.centralVoxelsTrain = ""
        self.centralVoxelsTest = ""
        
    # -------------------------------------------------------------------- END Function ------------------------------------------------------------------- #

    """ ####### Function to generate the network architecture  ######### """
    def generateNetworkLayers(self,
                            cnnLayers,
                            kernel_Shapes,
                            maxPooling_Layer,
                            sampleShape_Train,
                            sampleShape_Test,
                            inputSample_Train,
                            inputSample_Test,
                            layersToConnect):

        rng = np.random.RandomState(24575)
     
        # Define inputs for first layers (which will be re-used for next layers)
        inputSampleToNextLayer_Train = inputSample_Train
        inputSampleToNextLayer_Test = inputSample_Test
        inputSampleToNextLayerShape_Train = sampleShape_Train
        inputSampleToNextLayerShape_Test = sampleShape_Test

        # Get the convolutional layers
        numLayers = len(kernel_Shapes)
        numberCNNLayers = []
        numberFCLayers = []
        for l_i in range(1,len(kernel_Shapes)):
            if len(kernel_Shapes[l_i]) == 3:
                numberCNNLayers = l_i + 1

        numberFCLayers = numLayers - numberCNNLayers
        
        ######### -------------- Generate the convolutional layers --------------   #########
        # Some checks
        if self.weight_Initialization_CNN == 2:
            if len(self.weightsTrainedIdx) <> numberCNNLayers:
                print(" ... WARNING!!!! Number of indexes specified for trained layers does not correspond with number of conv layers in the created architecture...")

        if self.weight_Initialization_CNN == 2:
            weightsNames = getWeightsSet(self.weightsFolderName, self.weightsTrainedIdx)
            
        for l_i in xrange(0, numberCNNLayers) :
            
            # Get properties of this layer
            # The second element is the number of feature maps of previous layer
            currentLayerKernelShape = [cnnLayers[l_i], inputSampleToNextLayerShape_Train[1]] +  kernel_Shapes[l_i] 

            # If weights are going to be initialized from other pre-trained network they should be loaded in this stage
            # Otherwise
            weights = []
            if self.weight_Initialization_CNN == 2:
                weights = np.load(weightsNames[l_i])

            maxPoolingParameters = []
            dropoutRate = 0.0
            myLiviaNet3DConvLayer = LiviaNet3DConvLayer.LiviaNet3DConvLayer(rng,
                                                                            l_i,
                                                                            inputSampleToNextLayer_Train,
                                                                            inputSampleToNextLayer_Test,
                                                                            inputSampleToNextLayerShape_Train,
                                                                            inputSampleToNextLayerShape_Test,
                                                                            currentLayerKernelShape,
                                                                            self.applyBatchNorm,
                                                                            self.numberEpochToApplyBatchNorm,
                                                                            maxPoolingParameters,
                                                                            self.weight_Initialization_CNN,
                                                                            weights,
                                                                            self.activationType,
                                                                            dropoutRate
                                                                            )
                                                                            
            self.networkLayers.append(myLiviaNet3DConvLayer)
            
            # Just for printing
            inputSampleToNextLayer_Train_Old = inputSampleToNextLayerShape_Train
            inputSampleToNextLayer_Test_Old  = inputSampleToNextLayerShape_Test
            
            # Update inputs for next layer
            inputSampleToNextLayer_Train = myLiviaNet3DConvLayer.outputTrain
            inputSampleToNextLayer_Test  = myLiviaNet3DConvLayer.outputTest    
    
            inputSampleToNextLayerShape_Train = myLiviaNet3DConvLayer.outputShapeTrain
            inputSampleToNextLayerShape_Test  = myLiviaNet3DConvLayer.outputShapeTest
            
            print(" ----- (Training) Input shape: {}  ---> Output shape: {}  ||  kernel shape {}".format(inputSampleToNextLayer_Train_Old,inputSampleToNextLayerShape_Train, currentLayerKernelShape))
            print(" ----- (Testing) Input shape: {}   ---> Output shape: {}".format(inputSampleToNextLayer_Test_Old,inputSampleToNextLayerShape_Test))

        ######### -------------- Create the intermediate (i.e. multi-scale) connections from conv layers to FCN ----------------- ##################
        featMapsInFullyCN = inputSampleToNextLayerShape_Train[1]

        [featMapsInFullyCN, 
        inputToFullyCN_Train,
        inputToFullyCN_Test] = self.connectIntermediateLayers(layersToConnect,
                                                              inputSampleToNextLayer_Train,
                                                              inputSampleToNextLayer_Test,
                                                              featMapsInFullyCN)
        
        
        ######### --------------  Generate the Fully Connected Layers  ----------------- ##################

        # Define inputs
        inputFullyCNShape_Train = [self.batch_Size, featMapsInFullyCN] + inputSampleToNextLayerShape_Train[2:5]
        inputFullyCNShape_Test = [self.batch_Size, featMapsInFullyCN] + inputSampleToNextLayerShape_Test[2:5]

        # Kamnitsas applied padding and mirroring to the images when kernels in FC layers were larger than 1x1x1.
        # For this current work, we employed kernels of this size (i.e. 1x1x1), so there is no need to apply padding or mirroring.
        # TODO. Check
        
        print(" --- Starting to create the fully connected layers....")
        for l_i in xrange(numberCNNLayers, numLayers) :
            numberOfKernels = cnnLayers[l_i]
            kernel_shape = [kernel_Shapes[l_i][0],kernel_Shapes[l_i][0],kernel_Shapes[l_i][0]]
            
            currentLayerKernelShape = [cnnLayers[l_i], inputFullyCNShape_Train[1]] +  kernel_shape
            
            # If weights are going to be initialized from other pre-trained network they should be loaded in this stage
            # Otherwise

            weights = []
            applyBatchNorm = True
            epochsToApplyBatchNorm = 60
            maxPoolingParameters = []
            dropoutRate = self.dropout_Rates[l_i-numberCNNLayers]
            
            myLiviaNet3DFullyConnectedLayer = LiviaNet3DConvLayer.LiviaNet3DConvLayer(rng,
                                                                            l_i,
                                                                            inputToFullyCN_Train,
                                                                            inputToFullyCN_Test,
                                                                            inputFullyCNShape_Train,
                                                                            inputFullyCNShape_Test,
                                                                            currentLayerKernelShape,
                                                                            self.applyBatchNorm,
                                                                            self.numberEpochToApplyBatchNorm,
                                                                            maxPoolingParameters,
                                                                            self.weight_Initialization_FCN,
                                                                            weights,
                                                                            self.activationType,
                                                                            dropoutRate
                                                                            )

            self.networkLayers.append(myLiviaNet3DFullyConnectedLayer)
            
            # Just for printing
            inputFullyCNShape_Train_Old = inputFullyCNShape_Train
            inputFullyCNShape_Test_Old  = inputFullyCNShape_Test
            
            # Update inputs for next layer
            inputToFullyCN_Train = myLiviaNet3DFullyConnectedLayer.outputTrain
            inputToFullyCN_Test = myLiviaNet3DFullyConnectedLayer.outputTest    

            inputFullyCNShape_Train = myLiviaNet3DFullyConnectedLayer.outputShapeTrain
            inputFullyCNShape_Test = myLiviaNet3DFullyConnectedLayer.outputShapeTest
 
            # Print
            print(" ----- (Training) Input shape: {}  ---> Output shape: {}  ||  kernel shape {}".format(inputFullyCNShape_Train_Old,inputFullyCNShape_Train, currentLayerKernelShape))
            print(" ----- (Testing) Input shape: {}   ---> Output shape: {}".format(inputFullyCNShape_Test_Old,inputFullyCNShape_Test))
        
        
        ######### -------------- Do Classification layer  ----------------- ##################

        # Define kernel shape for classification layer
        featMaps_LastLayer = self.cnnLayers[-1]
        filterShape_ClassificationLayer = [self.n_classes, featMaps_LastLayer, 1, 1, 1]

        # Define inputs and shapes for the classification layer
        inputImageClassificationLayer_Train = inputToFullyCN_Train
        inputImageClassificationLayer_Test = inputToFullyCN_Test

        inputImageClassificationLayerShape_Train = inputFullyCNShape_Train
        inputImageClassificationLayerShape_Test = inputFullyCNShape_Test
        
        print(" ----- (Classification layer) kernel shape {}".format(filterShape_ClassificationLayer))
        classification_layer_Index = l_i

        weights = []
        applyBatchNorm = True
        epochsToApplyBatchNorm = 60
        maxPoolingParameters = []
        dropoutRate = self.dropout_Rates[len(self.dropout_Rates)-1]
        softmaxTemperature = 1.0
                                                                              
        myLiviaNet_ClassificationLayer = LiviaSoftmax.LiviaSoftmax(rng,
                                                                   classification_layer_Index,
                                                                   inputImageClassificationLayer_Train,
                                                                   inputImageClassificationLayer_Test,
                                                                   inputImageClassificationLayerShape_Train,
                                                                   inputImageClassificationLayerShape_Test,
                                                                   filterShape_ClassificationLayer,
                                                                   self.applyBatchNorm,
                                                                   self.numberEpochToApplyBatchNorm,
                                                                   maxPoolingParameters,
                                                                   self.weight_Initialization_FCN,
                                                                   weights,
                                                                   0, #self.activationType,
                                                                   dropoutRate,
                                                                   softmaxTemperature
                                                                   )
                       
        self.networkLayers.append(myLiviaNet_ClassificationLayer)
        self.lastLayer = myLiviaNet_ClassificationLayer
        
        print(" ----- (Training) Input shape: {}  ---> Output shape: {}  ||  kernel shape {}".format(inputImageClassificationLayerShape_Train,myLiviaNet_ClassificationLayer.outputShapeTrain, filterShape_ClassificationLayer))
        print(" ----- (Testing) Input shape:  {}  ---> Output shape: {}".format(inputImageClassificationLayerShape_Test,myLiviaNet_ClassificationLayer.outputShapeTest))
        
# -------------------------------------------------------------------- END Function ------------------------------------------------------------------- #

                        
    def updateLayersMatricesBatchNorm(self):
        for l_i in xrange(0, len(self.networkLayers) ) :
            self.networkLayers[l_i].updateLayerMatricesBatchNorm()
# -------------------------------------------------------------------- END Function ------------------------------------------------------------------- #
   
    """ Function that connects intermediate layers to the input of the first fully connected layer 
        This is done for multi-scale features """    
    def connectIntermediateLayers(self,
                                  layersToConnect,
                                  inputSampleInFullyCN_Train,
                                  inputSampleInFullyCN_Test,
                                  featMapsInFullyCN):

        centralVoxelsTrain = self.centralVoxelsTrain
        centralVoxelsTest = self.centralVoxelsTest
    
        for l_i in layersToConnect :
            currentLayer = self.networkLayers[l_i]
            output_train = currentLayer.outputTrain
            output_trainShape = currentLayer.outputShapeTrain
            output_test = currentLayer.outputTest
            output_testShape = currentLayer.outputShapeTest

            # Get the middle part of feature maps at intermediate levels to make them of the same shape at the beginning of the
            # first fully connected layer
            featMapsCenter_Train = extractCenterFeatMaps(output_train, output_trainShape, centralVoxelsTrain)
            featMapsCenter_Test  = extractCenterFeatMaps(output_test, output_testShape, centralVoxelsTest)

            featMapsInFullyCN = featMapsInFullyCN + currentLayer._numberOfFeatureMaps
            inputSampleInFullyCN_Train = T.concatenate([inputSampleInFullyCN_Train, featMapsCenter_Train], axis=1)
            inputSampleInFullyCN_Test = T.concatenate([inputSampleInFullyCN_Test, featMapsCenter_Test], axis=1)

        return [featMapsInFullyCN, inputSampleInFullyCN_Train, inputSampleInFullyCN_Test]
    
    
    #############   Functions for OPTIMIZERS ################# 

    def getUpdatesOfTrainableParameters(self, cost, paramsTraining, numberParamsPerLayer) :
        # Optimizers
        def SGD():
            print (" --- Optimizer: Stochastic gradient descent (SGD)")
            updates = self.updateParams_SGD(cost, paramsTraining, numberParamsPerLayer)
            return updates
        def RMSProp():
            print (" --- Optimizer: RMS Prop")
            updates = self.updateParams_RMSProp(cost, paramsTraining, numberParamsPerLayer)
            return updates
       
        # TODO. Include more optimizers here
        optionsOptimizer = {0 : SGD,
                            1 : RMSProp}

        updates = optionsOptimizer[self.optimizerType]()
        
        return updates

    """ # Optimizers:
    # More optimizers in : https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py """
    # ========= Update the trainable parameters using Stocastic Gradient Descent ===============
    def updateParams_SGD(self, cost, paramsTraining, numberParamsPerLayer) :
        # Create a list of gradients for all model parameters
        grads = T.grad(cost, paramsTraining)

        # Get learning rates for each param
        #learning_rates = extendLearningRateToParams(numberParamsPerLayer,self.learning_rate)
        
        self.vel_Momentum = []
        updates = []
        
        constantForCurrentGradientUpdate = 1.0 - self.momentum*self.momentumNormalized 

        #for param, grad, lrate  in zip(paramsTraining, grads, learning_rates) :
        for param, grad  in zip(paramsTraining, grads) :
            v = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            self.vel_Momentum.append(v)

            stepToGradientDirection = constantForCurrentGradientUpdate*self.learning_rate*grad
            newVel = self.momentum * v - stepToGradientDirection
            
            if self.momentumType == 0 : 
                updateToParam = newVel
            else : 
                updateToParam = self.momentum*newVel - stepToGradientDirection
                
            updates.append((v, newVel)) 
            updates.append((param, param + updateToParam))
            
        return updates
        
    # ========= Update the trainable parameters using RMSProp ===============
    def updateParams_RMSProp(self, cost, paramsTraining, numberParamsPerLayer) : 
        # Original code: https://gist.github.com/Newmu/acb738767acb4788bac3
        # epsilon=1e-4 in paper.
        # Kamnitsas reported NaN values in cost function when employing this value.
        # Worked ok with epsilon=1e-6.

        grads = T.grad(cost, paramsTraining)

        # Get learning rates for each param
        #learning_rates = extendLearningRateToParams(numberParamsPerLayer,self.learning_rate)

        self.params_RmsProp = []
        self.vel_Momentum = []
        updates = []
        
        constantForCurrentGradientUpdate = 1.0 - self.momentum*self.momentumNormalized 
        
        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)

        for param, grad in zip(paramsTraining, grads):
            accu = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            self.params_RmsProp.append(accu) 
            
            v = theano.shared(param.get_value()*0., broadcastable=param.broadcastable) 
            
            self.vel_Momentum.append(v)

            accu_new = self.rho_RMSProp * accu + (one - self.rho_RMSProp) * T.sqr(grad)

            numGradStep = self.learning_rate * grad
            denGradStep = T.sqrt(accu_new + self.epsilon_RMSProp)
            
            stepToGradientDirection = constantForCurrentGradientUpdate*(numGradStep /denGradStep) 
            
            newVel = self.momentum * v - stepToGradientDirection
            
            if self.momentumType == 0 : 
                updateToParam = newVel
            else : 
                updateToParam = self.momentum*newVel - stepToGradientDirection
               
            updates.append((accu, accu_new))
            updates.append((v, newVel)) 
            updates.append((param, param + updateToParam))
            
        return updates
        
# -------------------------------------------------------------------- END Function ------------------------------------------------------------------- #
        
    """ ------ Get trainable parameters --------- """
    def getTrainable_Params(_self):
        trainable_Params = []
        numberTrain_ParamsLayer = [] 
        for l_i in xrange(0, len(_self.networkLayers) ) :
            trainable_Params = trainable_Params + _self.networkLayers[l_i].params
            numberTrain_ParamsLayer.append(_self.networkLayers[l_i].numberOfTrainableParams) # TODO: Get this directly as len(_self.networkLayers[l_i].params)
            
        return trainable_Params,numberTrain_ParamsLayer

# -------------------------------------------------------------------- END Function ------------------------------------------------------------------- #
    
    def initTrainingParameters(self,
                               costFunction,
                               L1_reg_C,
                               L2_reg_C,
                               learning_rate,
                               momentumType,
                               momentumValue,
                               momentumNormalized,
                               optimizerType,
                               rho_RMSProp,
                               epsilon_RMSProp
                               ) :

        print(" ------- Initializing network training parameters...........")
        self.numberOfEpochsTrained = 0

        self.L1_reg_C = L1_reg_C
        self.L2_reg_C = L2_reg_C

        # Set Learning rate and store the last epoch where it was modified
        self.initialLearningRate = learning_rate

        # TODO: Check the shared variables from learning rates
        self.learning_rate.set_value(self.initialLearningRate[0])          
        

        # Set momentum type and values
        self.momentumType = momentumType
        self.initialMomentumValue = momentumValue
        self.momentumNormalized = momentumNormalized
        self.momentum.set_value(self.initialMomentumValue)
        
        # Optimizers
        if (optimizerType == 2):
            optimizerType = 1
            
        def SGD():
            print (" --- Optimizer: Stochastic gradient descent (SGD)")
            self.optimizerType = optimizerType

        def RMSProp():
            print (" --- Optimizer: RMS Prop")
            self.optimizerType = optimizerType
            self.rho_RMSProp = rho_RMSProp
            self.epsilon_RMSProp = epsilon_RMSProp
       
        # TODO. Include more optimizers here
        optionsOptimizer = {0 : SGD,
                            1 : RMSProp}
        
        optionsOptimizer[optimizerType]()
                              
# -------------------------------------------------------------------- END Function ------------------------------------------------------------------- #
        
    def updateParams_BatchNorm(self) : 
        updatesForBnRollingAverage = []
        for l_i in xrange(0, len(self.networkLayers) ) :
            currentLayer = self.networkLayers[l_i]
            updatesForBnRollingAverage.extend( currentLayer.getUpdatesForBnRollingAverage() ) 
        return updatesForBnRollingAverage

    # ------------------------------------------------------------------------------------ #
    # ---------------------------     Compile the Theano functions     ------------------- #
    # ------------------------------------------------------------------------------------ #
    def compileTheanoFunctions(self):
        print(" ----------------- Starting compilation process ----------------- ")        
        
        # ------- Create and initialize sharedVariables needed to compile the training function ------ #
        # -------------------------------------------------------------------------------------------- #
        # For training 
        self.trainingData_x = theano.shared(np.zeros([1,1,1,1,1], dtype="float32"), borrow = True)
        self.trainingData_y = theano.shared(np.zeros([1,1,1,1], dtype="float32") , borrow = True)  
        
        # For testing 
        self.testingData_x = theano.shared(np.zeros([1,1,1,1,1], dtype="float32"), borrow = True)
        
        x_Train = self.inputNetwork_Train
        x_Test  = self.inputNetwork_Test
        y_Train = T.itensor4('y')
        
        # Allocate symbolic variables for the data
        index_Train = T.lscalar()
        index_Test  = T.lscalar()
        
        # ------- Needed to compile the training function ------ #
        # ------------------------------------------------------ #
        trainingData_y_CastedToInt   = T.cast( self.trainingData_y, 'int32') 
        
        # To accomodate the weights in the cost function to account for class imbalance
        weightsOfClassesInCostFunction = T.fvector()  
        weightPerClass = T.fvector() 
        
        # --------- Get trainable parameters (to be fit by gradient descent) ------- #
        # -------------------------------------------------------------------------- #
        
        [paramsTraining, numberParamsPerLayer] = self.getTrainable_Params()
        
        # ------------------ Define the cost function --------------------- #
        # ----------------------------------------------------------------- #
        def negLogLikelihood():
            print (" --- Cost function: negativeLogLikelihood")
            
            costInLastLayer = self.lastLayer.negativeLogLikelihoodWeighted(y_Train,weightPerClass)
            return costInLastLayer
            
        def NotDefined():
            print (" --- Cost function: Not defined!!!!!! WARNING!!!")

        optionsCostFunction = {0 : negLogLikelihood,
                               1 : NotDefined}

        costInLastLayer = optionsCostFunction[self.costFunction]()
        
        # --------------------------- Get costs --------------------------- #
        # ----------------------------------------------------------------- #
        # Get L1 and L2 weights regularization
        costL1 = 0
        costL2 = 0
        
        # Compute the costs
        for l_i in xrange(0, len(self.networkLayers)) :    
                costL1 += abs(self.networkLayers[l_i].W).sum()
                costL2 += (self.networkLayers[l_i].W ** 2).sum()
        
        # Add also the cost of the last layer                     
        cost = (costInLastLayer
                + self.L1_reg_C * costL1
                + self.L2_reg_C * costL2)

        # --------------------- Include all trainable parameters in updates (for optimization) ---------------------- #
        # ----------------------------------------------------------------------------------------------------------- #
        updates = self.getUpdatesOfTrainableParameters(cost, paramsTraining, numberParamsPerLayer)
        
        # --------------------- Include batch normalization params ---------------------- #
        # ------------------------------------------------------------------------------- #
        updates = updates + self.updateParams_BatchNorm()

        # For the testing function we need to get the Feature maps activations
        featMapsActivations = []
        lower_act = 0
        upper_act = 9999
        
        # TODO: Change to output_Test
        for l_i in xrange(0,len(self.networkLayers)):
            featMapsActivations.append(self.networkLayers[l_i].outputTest[:, lower_act : upper_act, :, :, :])

        # For the last layer get the predicted probabilities (p_y_given_x_test)
        featMapsActivations.append(self.lastLayer.p_y_given_x_test)

        # --------------------- Preparing data to compile the functions ---------------------- #
        # ------------------------------------------------------------------------------------ #
        
        givensDataSet_Train = { x_Train: self.trainingData_x[index_Train * self.batch_Size: (index_Train + 1) * self.batch_Size],
                                y_Train: trainingData_y_CastedToInt[index_Train * self.batch_Size: (index_Train + 1) * self.batch_Size],
                                weightPerClass: weightsOfClassesInCostFunction }

       
        givensDataSet_Test  = { x_Test: self.testingData_x[index_Test * self.batch_Size: (index_Test + 1) * self.batch_Size] }
        
        print(" ...Compiling the training function...")
        
        self.networkModel_Train = theano.function(
                                    [index_Train, weightsOfClassesInCostFunction],
                                    #[cost] + self.lastLayer.doEvaluation(y_Train),
                                    [cost],
                                    updates=updates,
                                    givens = givensDataSet_Train
                                    )
                          
        print(" ...The training function was compiled...")

        #self.getProbabilities = theano.function(
                         #[index],
                         #self.lastLayer.p_y_given_x_Train,
                         #givens={
                            #x: self.trainingData_x[index * _self.batch_size: (index + 1) * _self.batch_size]
                         #}
         #)
     

        print(" ...Compiling the testing function...")
        self.networkModel_Test = theano.function(
                                  [index_Test],
                                  featMapsActivations,
                                  givens = givensDataSet_Test
                                  )
        print(" ...The testing function was compiled...")
# -------------------------------------------------------------------- END Function ------------------------------------------------------------------- #

####### Function to generate the CNN #########

    def createNetwork(self,
                      networkName, 
                      folderName,
                      cnnLayers,
                      kernel_Shapes,
                      intermediate_ConnectedLayers,
                      n_classes,
                      sampleSize_Train,
                      sampleSize_Test,
                      batch_Size,
                      applyBatchNorm,
                      numberEpochToApplyBatchNorm,
                      activationType,
                      dropout_Rates,
                      pooling_Params,
                      weights_Initialization_CNN,
                      weights_Initialization_FCN,
                      weightsFolderName,
                      weightsTrainedIdx,
                      softmax_Temp
                      ):

        # ============= Model Parameters Passed as arguments ================
        # Assign parameters:
        self.networkName = networkName
        self.folderName = folderName
        self.cnnLayers = cnnLayers
        self.n_classes = n_classes
        self.kernel_Shapes = kernel_Shapes
        self.intermediate_ConnectedLayers = intermediate_ConnectedLayers
        self.pooling_scales = pooling_Params
        self.dropout_Rates = dropout_Rates
        self.activationType = activationType
        self.weight_Initialization_CNN = weights_Initialization_CNN
        self.weight_Initialization_FCN = weights_Initialization_FCN
        self.weightsFolderName = weightsFolderName
        self.weightsTrainedIdx = weightsTrainedIdx
        self.batch_Size = batch_Size
        self.sampleSize_Train = sampleSize_Train
        self.sampleSize_Test = sampleSize_Test
        self.applyBatchNorm = applyBatchNorm
        self.numberEpochToApplyBatchNorm = numberEpochToApplyBatchNorm
        self.softmax_Temp = softmax_Temp

        # Compute the CNN receptive field
        stride = 1;
        self.receptiveField = computeReceptiveField(self.kernel_Shapes, stride)

        # --- Size of Image samples ---
        self.sampleSize_Train = sampleSize_Train
        self.sampleSize_Test = sampleSize_Test
        
        ## --- Batch Size ---
        self.batch_Size = batch_Size

        # ======== Calculated Attributes =========
        self.centralVoxelsTrain = getCentralVoxels(self.sampleSize_Train, self.receptiveField) 
        self.centralVoxelsTest = getCentralVoxels(self.sampleSize_Test, self.receptiveField) 
        
        #==============================
        rng = numpy.random.RandomState(23455)

        # Transfer to LIVIA NET
        self.sampleSize_Train = sampleSize_Train
        self.sampleSize_Test = sampleSize_Test
        
        # --------- Now we build the model -------- #

        print("...[STATUS]: Building the Network model...")
        
        # Define the symbolic variables used as input of the CNN
        # start-snippet-1
        # Define tensor5
        tensor5 = T.TensorType(dtype='float32', broadcastable=(False, False, False, False, False))
        self.inputNetwork_Train = tensor5() 
        self.inputNetwork_Test = tensor5()

        # Define input shapes to the netwrok
        inputSampleShape_Train = (self.batch_Size, 1, self.sampleSize_Train[0], self.sampleSize_Train[1], self.sampleSize_Train[2])
        inputSampleShape_Test = (self.batch_Size, 1, self.sampleSize_Test[0], self.sampleSize_Test[1], self.sampleSize_Test[2])

        print (" - Shape of input subvolume (Training): {}".format(inputSampleShape_Train))
        print (" - Shape of input subvolume (Testing): {}".format(inputSampleShape_Test))

        inputSample_Train = self.inputNetwork_Train
        inputSample_Test = self.inputNetwork_Test

        # TODO change cnnLayers name by networkLayers
        self.generateNetworkLayers(cnnLayers,
                                   kernel_Shapes,
                                   self.pooling_scales,
                                   inputSampleShape_Train,
                                   inputSampleShape_Test,
                                   inputSample_Train,
                                   inputSample_Test,
                                   intermediate_ConnectedLayers)      

    # Release Data from GPU
    def releaseGPUData(self) :
        # GPU NOTE: Remove the input values to avoid copying data to the GPU
        
        # Image Data
        self.trainingData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))
        self.testingData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))

        # Labels
        self.trainingData_y.set_value(np.zeros([1,1,1,1], dtype="float32"))

