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

from LiviaNet import LiviaNet3D
from Modules.General.Utils import dump_model_to_gzip_file
from Modules.General.Utils import makeFolder
from Modules.Parsers.parsersUtils import parserConfigIni


def generateNetwork(configIniName) :   

    myParserConfigIni = parserConfigIni()

    myParserConfigIni.readConfigIniFile(configIniName,0)
    print " **********************  Starting creation model **********************"
    print " ------------------------ General ------------------------ "
    print " - Network name: {}".format(myParserConfigIni.networkName)
    print " - Folder to save the outputs: {}".format(myParserConfigIni.folderName)
    print " ------------------------ CNN Architecture ------------------------  "
    print " - Number of classes: {}".format(myParserConfigIni.n_classes)
    print " - Layers: {}".format(myParserConfigIni.layers)
    print " - Kernel sizes: {}".format(myParserConfigIni.kernels)

    print " - Intermediate connected CNN layers: {}".format(myParserConfigIni.intermediate_ConnectedLayers)
   
    print " - Pooling: {}".format(myParserConfigIni.pooling_scales)
    print " - Dropout: {}".format(myParserConfigIni.dropout_Rates)
    
    def Linear():
        print " --- Activation function: Linear"
 
    def ReLU():
        print " --- Activation function: ReLU"
 
    def PReLU():
        print " --- Activation function: PReLU"

    def LeakyReLU():
        print " --- Activation function: Leaky ReLU"
                  
    printActivationFunction = {0 : Linear,
                               1 : ReLU,
                               2 : PReLU,
                               3 : LeakyReLU}

    printActivationFunction[myParserConfigIni.activationType]()
        
    def Random(layerType):
        print " --- Weights initialization (" +layerType+ " Layers): Random"
 
    def Delving(layerType):
        print " --- Weights initialization (" +layerType+ " Layers): Delving"
 
    def PreTrained(layerType):
        print " --- Weights initialization (" +layerType+ " Layers): PreTrained"
        
    printweight_Initialization_CNN = {0 : Random,
                                      1 : Delving,
                                      2 : PreTrained}
                               
    printweight_Initialization_CNN[myParserConfigIni.weight_Initialization_CNN]('CNN')
    printweight_Initialization_CNN[myParserConfigIni.weight_Initialization_FCN]('FCN')

    print " ------------------------ Training Parameters ------------------------  "
    if len(myParserConfigIni.learning_rate) == 1:
        print " - Learning rate: {}".format(myParserConfigIni.learning_rate)
    else:
        for i in xrange(len(myParserConfigIni.learning_rate)):
            print " - Learning rate at layer {} : {} ".format(str(i+1),myParserConfigIni.learning_rate[i])
    
    print " - Batch size: {}".format(myParserConfigIni.batch_size)

    if myParserConfigIni.applyBatchNorm == True:
        print " - Apply batch normalization in {} epochs".format(myParserConfigIni.BatchNormEpochs)
        
    print " ------------------------ Size of samples ------------------------  "
    print " - Training: {}".format(myParserConfigIni.sampleSize_Train)
    print " - Testing: {}".format(myParserConfigIni.sampleSize_Test)

    # --------------- Create my LiviaNet3D object  --------------- 
    myLiviaNet3D = LiviaNet3D()
    
    # --------------- Create the whole architecture (Conv layers + fully connected layers + classification layer)  --------------- 
    myLiviaNet3D.createNetwork(myParserConfigIni.networkName,
                               myParserConfigIni.folderName,
                               myParserConfigIni.layers,
                               myParserConfigIni.kernels,
                               myParserConfigIni.intermediate_ConnectedLayers,
                               myParserConfigIni.n_classes,
                               myParserConfigIni.sampleSize_Train,
                               myParserConfigIni.sampleSize_Test,
                               myParserConfigIni.batch_size,
                               myParserConfigIni.applyBatchNorm,
                               myParserConfigIni.BatchNormEpochs,
                               myParserConfigIni.activationType,
                               myParserConfigIni.dropout_Rates,
                               myParserConfigIni.pooling_scales,
                               myParserConfigIni.weight_Initialization_CNN,
                               myParserConfigIni.weight_Initialization_FCN,
                               myParserConfigIni.weightsFolderName,
                               myParserConfigIni.weightsTrainedIdx,
                               myParserConfigIni.tempSoftMax
                               )
                               # TODO: Specify also the weights if pre-trained
                               
                          
    #  ---------------  Initialize all the training parameters  --------------- 
    myLiviaNet3D.initTrainingParameters(myParserConfigIni.costFunction,
                                        myParserConfigIni.L1_reg_C,
                                        myParserConfigIni.L2_reg_C,
                                        myParserConfigIni.learning_rate,
                                        myParserConfigIni.momentumType,
                                        myParserConfigIni.momentumValue,
                                        myParserConfigIni.momentumNormalized,
                                        myParserConfigIni.optimizerType,
                                        myParserConfigIni.rho_RMSProp,
                                        myParserConfigIni.epsilon_RMSProp
                                        )
   
    # ---------------  Compile the functions (Training/Validation/Testing) --------------- 
    myLiviaNet3D.compileTheanoFunctions()

    #  --------------- Save the model --------------- 
    # Generate folders to store the model
    BASE_DIR  = os.getcwd()
    path_Temp = os.path.join(BASE_DIR,'outputFiles')
    # For the networks
    netFolderName  = os.path.join(path_Temp,myParserConfigIni.folderName)
    netFolderName  = os.path.join(netFolderName,'Networks')
   
    # For the predictions
    predlFolderName    = os.path.join(path_Temp,myParserConfigIni.folderName)
    predlFolderName    = os.path.join(predlFolderName,'Pred')
    predValFolderName  = os.path.join(predlFolderName,'Validation')
    predTestFolderName = os.path.join(predlFolderName,'Testing')
   
    makeFolder(netFolderName, "Networks")
    makeFolder(predValFolderName, "to store predictions (Validation)")
    makeFolder(predTestFolderName, "to store predictions (Testing)")

    modelFileName = netFolderName + "/" + myParserConfigIni.networkName + "_Epoch0"
    dump_model_to_gzip_file(myLiviaNet3D, modelFileName)
    
    strFinal =  " Network model saved in " + netFolderName + " as " + myParserConfigIni.networkName + "_Epoch0"
    print  strFinal
    
    return modelFileName
   
   
