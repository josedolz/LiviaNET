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
 
import ConfigParser
import json
import os

# -------- Parse parameters to create the network model -------- #
class parserConfigIni(object):
   def __init__(_self):
      _self.networkName = []
      
   #@staticmethod
   def readConfigIniFile(_self,fileName,task):
      # Task: 0-> Generate model
      #       1-> Train model
      #       2-> Segmentation

      def createModel():
          print (" --- Creating model (Reading parameters...)")
          _self.readModelCreation_params(fileName)
      def trainModel():
          print (" --- Training model (Reading parameters...)")
          _self.readModelTraining_params(fileName)
      def testModel():
          print (" --- Testing model (Reading parameters...)")
          _self.readModelTesting_params(fileName)
       
        # TODO. Include more optimizers here
      optionsParser = {0 : createModel,
                       1 : trainModel,
                       2 : testModel}
      optionsParser[task]()

   # Read parameters to Generate model
   def readModelCreation_params(_self,fileName) :
      ConfigIni = ConfigParser.ConfigParser()
      ConfigIni.read(fileName)
      # ------- General --------- #
      _self.networkName = ConfigIni.get('General','networkName')
      _self.folderName  = ConfigIni.get('General','folderName')
      
      # ------- Network Architecture ------- #
      _self.n_classes                    = json.loads(ConfigIni.get('CNN_Architecture','n_classes')) # Number of (segmentation) classes
      _self.layers                       = json.loads(ConfigIni.get('CNN_Architecture','numkernelsperlayer'))  # Number of layers
      _self.kernels                      = json.loads(ConfigIni.get('CNN_Architecture','kernelshapes')) # Kernels shape
      _self.intermediate_ConnectedLayers = json.loads(ConfigIni.get('CNN_Architecture','intermediateConnectedLayers'))
      _self.pooling_scales               = json.loads(ConfigIni.get('CNN_Architecture','pooling_scales')) # Pooling
      _self.dropout_Rates                = json.loads(ConfigIni.get('CNN_Architecture','dropout_Rates')) # Dropout
      _self.activationType               = json.loads(ConfigIni.get('CNN_Architecture','activationType')) # Activation Typ
      _self.weight_Initialization_CNN    = json.loads(ConfigIni.get('CNN_Architecture','weight_Initialization_CNN')) # weight_Initialization (CNN)
      _self.weight_Initialization_FCN    = json.loads(ConfigIni.get('CNN_Architecture','weight_Initialization_FCN')) # weight_Initialization (FCN)
      _self.weightsFolderName            = ConfigIni.get('CNN_Architecture','weights folderName') # weights folder
      _self.weightsTrainedIdx            = json.loads(ConfigIni.get('CNN_Architecture','weights trained indexes')) # weights indexes to employ

      _self.batch_size                   = json.loads(ConfigIni.get('Training Parameters','batch_size')) # Batch size
      _self.sampleSize_Train             = json.loads(ConfigIni.get('Training Parameters','sampleSize_Train'))
      _self.sampleSize_Test              = json.loads(ConfigIni.get('Training Parameters','sampleSize_Test'))

      _self.costFunction       = json.loads(ConfigIni.get('Training Parameters','costFunction'))
      _self.L1_reg_C           = json.loads(ConfigIni.get('Training Parameters','L1 Regularization Constant'))
      _self.L2_reg_C           = json.loads(ConfigIni.get('Training Parameters','L2 Regularization Constant'))
      _self.learning_rate      = json.loads(ConfigIni.get('Training Parameters','Leraning Rate'))
      _self.momentumType       = json.loads(ConfigIni.get('Training Parameters','Momentum Type'))
      _self.momentumValue      = json.loads(ConfigIni.get('Training Parameters','Momentum Value'))
      _self.momentumNormalized = json.loads(ConfigIni.get('Training Parameters','momentumNormalized'))
      _self.optimizerType      = json.loads(ConfigIni.get('Training Parameters','Optimizer Type'))
      _self.rho_RMSProp        = json.loads(ConfigIni.get('Training Parameters','Rho RMSProp'))
      _self.epsilon_RMSProp    = json.loads(ConfigIni.get('Training Parameters','Epsilon RMSProp'))
      applyBatchNorm           = json.loads(ConfigIni.get('Training Parameters','applyBatchNormalization'))

      if applyBatchNorm == 1:
          _self.applyBatchNorm = True
      else:
          _self.applyBatchNorm = False
      
      _self.BatchNormEpochs   = json.loads(ConfigIni.get('Training Parameters','BatchNormEpochs'))
      _self.tempSoftMax       = json.loads(ConfigIni.get('Training Parameters','SoftMax temperature'))

      # TODO: Do some sanity checks

   # Read parameters to TRAIN model
   def readModelTraining_params(_self,fileName) :
      ConfigIni = ConfigParser.ConfigParser()
      ConfigIni.read(fileName)

      # Get training/validation image names
      # Paths
      _self.imagesFolder             = ConfigIni.get('Training Images','imagesFolder')
      _self.GroundTruthFolder        = ConfigIni.get('Training Images','GroundTruthFolder')
      _self.ROIFolder                = ConfigIni.get('Training Images','ROIFolder')
      _self.indexesForTraining       = json.loads(ConfigIni.get('Training Images','indexesForTraining'))
      _self.indexesForValidation     = json.loads(ConfigIni.get('Training Images','indexesForValidation'))
      _self.imageTypesTrain          = json.loads(ConfigIni.get('Training Images','imageTypes'))
      
      # training params
      _self.numberOfEpochs                    = json.loads(ConfigIni.get('Training Parameters','number of Epochs'))
      _self.numberOfSubEpochs                 = json.loads(ConfigIni.get('Training Parameters','number of SubEpochs'))
      _self.numberOfSamplesSupEpoch           = json.loads(ConfigIni.get('Training Parameters','number of samples at each SubEpoch Train'))
      _self.firstEpochChangeLR                = json.loads(ConfigIni.get('Training Parameters','First Epoch Change LR'))
      _self.frequencyChangeLR                 = json.loads(ConfigIni.get('Training Parameters','Frequency Change LR'))
      _self.applyPadding                      = json.loads(ConfigIni.get('Training Parameters','applyPadding'))  

   def readModelTesting_params(_self,fileName) :
      ConfigIni = ConfigParser.ConfigParser()
      ConfigIni.read(fileName)
 
      _self.imagesFolder      = ConfigIni.get('Segmentation Images','imagesFolder')
      _self.GroundTruthFolder = ConfigIni.get('Segmentation Images','GroundTruthFolder')
      _self.ROIFolder         = ConfigIni.get('Segmentation Images','ROIFolder')
     
      _self.imageTypes        = json.loads(ConfigIni.get('Segmentation Images','imageTypes'))
      _self.indexesToSegment  = json.loads(ConfigIni.get('Segmentation Images','indexesToSegment'))
      _self.applyPadding      = json.loads(ConfigIni.get('Segmentation Images','applyPadding'))
      

