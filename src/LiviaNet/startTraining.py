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

import sys
import time
import numpy as np
import random
import math
import os
                  
from Modules.General.Utils import getImagesSet
from Modules.General.Utils import dump_model_to_gzip_file
from Modules.General.Utils import load_model_from_gzip_file
from Modules.General.Evaluation import computeDice
from Modules.IO.sampling import getSamplesSubepoch
from Modules.Parsers.parsersUtils import parserConfigIni
from startTesting import segmentVolume
import pdb


def startTraining(networkModelName,configIniName):
    print " ************************************************  STARTING TRAINING **************************************************"
    print " **********************  Starting training model (Reading parameters) **********************"

    myParserConfigIni = parserConfigIni()
   
    myParserConfigIni.readConfigIniFile(configIniName,1)
    
    # Image type (0: Nifti, 1: Matlab)
    imageType = myParserConfigIni.imageTypesTrain

    print (" --- Do training in {} epochs with {} subEpochs each...".format(myParserConfigIni.numberOfEpochs, myParserConfigIni.numberOfSubEpochs))
    print "-------- Reading Images names used in training/validation -------------"

    # -- Get list of images used for training -- #
    (imageNames_Train, names_Train)          = getImagesSet(myParserConfigIni.imagesFolder,myParserConfigIni.indexesForTraining)  # Images
    (groundTruthNames_Train, gt_names_Train) = getImagesSet(myParserConfigIni.GroundTruthFolder,myParserConfigIni.indexesForTraining) # Ground truth
    (roiNames_Train, roi_names_Train)        = getImagesSet(myParserConfigIni.ROIFolder,myParserConfigIni.indexesForTraining) # ROI
    
    # -- Get list of images used for validation -- #
    (imageNames_Val, names_Val)          = getImagesSet(myParserConfigIni.imagesFolder,myParserConfigIni.indexesForValidation)  # Images
    (groundTruthNames_Val, gt_names_Val) = getImagesSet(myParserConfigIni.GroundTruthFolder,myParserConfigIni.indexesForValidation) # Ground truth
    (roiNames_Val, roi_names_Val)        = getImagesSet(myParserConfigIni.ROIFolder,myParserConfigIni.indexesForValidation) # ROI

    # Print names
    print " ================== Images for training ================"
    for i in range(0,len(names_Train)):
       if len(roi_names_Train) > 0:
            print(" Image({}): {}  |  GT: {}  |  ROI {} ".format(i,names_Train[i], gt_names_Train[i], roi_names_Train[i] ))
       else:
            print(" Image({}): {}  |  GT: {}  ".format(i,names_Train[i], gt_names_Train[i] ))
    print " ================== Images for validation ================"
    for i in range(0,len(names_Val)):
        if len(roi_names_Train) > 0:
            print(" Image({}): {}  |  GT: {}  |  ROI {} ".format(i,names_Val[i], gt_names_Val[i], roi_names_Val[i] ))
        else:
            print(" Image({}): {}  |  GT: {}  ".format(i,names_Val[i], gt_names_Val[i]))
    print " ==============================================================="
   
    # --------------- Load my LiviaNet3D object  --------------- 
    print (" ... Loading model from {}".format(networkModelName))
    myLiviaNet3D = load_model_from_gzip_file(networkModelName)
    print " ... Network architecture successfully loaded...."

    # Asign parameters to loaded Net
    myLiviaNet3D.numberOfEpochs = myParserConfigIni.numberOfEpochs
    myLiviaNet3D.numberOfSubEpochs = myParserConfigIni.numberOfSubEpochs
    myLiviaNet3D.numberOfSamplesSupEpoch  = myParserConfigIni.numberOfSamplesSupEpoch
    myLiviaNet3D.firstEpochChangeLR  = myParserConfigIni.firstEpochChangeLR
    myLiviaNet3D.frequencyChangeLR  = myParserConfigIni.frequencyChangeLR
    
    numberOfEpochs = myLiviaNet3D.numberOfEpochs
    numberOfSubEpochs = myLiviaNet3D.numberOfSubEpochs
    numberOfSamplesSupEpoch = myLiviaNet3D.numberOfSamplesSupEpoch
    
    # --------------- --------------  --------------- 
    # --------------- Start TRAINING  --------------- 
    # --------------- --------------  --------------- 
    # Get sample dimension values
    receptiveField = myLiviaNet3D.receptiveField
    sampleSize_Train = myLiviaNet3D.sampleSize_Train

    trainingCost = []

    if myParserConfigIni.applyPadding == 1:
        applyPadding = True
    else:
        applyPadding = False
    
    learningRateModifiedEpoch = 0
    
    # Run over all the (remaining) epochs and subepochs
    for e_i in xrange(numberOfEpochs):
        # Recover last trained epoch
        numberOfEpochsTrained = myLiviaNet3D.numberOfEpochsTrained
                                        
        print(" ============== EPOCH: {}/{} =================".format(numberOfEpochsTrained+1,numberOfEpochs))

        costsOfEpoch = []
        
        for subE_i in xrange(numberOfSubEpochs): 
            epoch_nr = subE_i+1
            print (" --- SubEPOCH: {}/{}".format(epoch_nr,myLiviaNet3D.numberOfSubEpochs))

            # Get all the samples that will be used in this sub-epoch
            [imagesSamplesAll,
            gt_samplesAll] = getSamplesSubepoch(numberOfSamplesSupEpoch,
                                                imageNames_Train,
                                                groundTruthNames_Train,
                                                roiNames_Train,
                                                imageType,
                                                sampleSize_Train,
                                                receptiveField,
                                                applyPadding
                                                )

            # Variable that will contain weights for the cost function
            # --- In its current implementation, all the classes have the same weight
            weightsCostFunction = np.ones(myLiviaNet3D.n_classes, dtype='float32')
               
            numberBatches = len(imagesSamplesAll) / myLiviaNet3D.batch_Size 
            
            myLiviaNet3D.trainingData_x.set_value(imagesSamplesAll, borrow=True)
            myLiviaNet3D.trainingData_y.set_value(gt_samplesAll, borrow=True)
                 
            costsOfBatches = []
            evalResultsSubepoch = np.zeros([ myLiviaNet3D.n_classes, 4 ], dtype="int32")
    
            for b_i in xrange(numberBatches):
                # TODO: Make a line that adds a point at each trained batch (Or percentage being updated)
                costErrors = myLiviaNet3D.networkModel_Train(b_i, weightsCostFunction)
                meanBatchCostError = costErrors[0]
                costsOfBatches.append(meanBatchCostError)
                myLiviaNet3D.updateLayersMatricesBatchNorm() 

            
            #======== Calculate and Report accuracy over subepoch
            meanCostOfSubepoch = sum(costsOfBatches) / float(numberBatches)
            print(" ---------- Cost of this subEpoch: {}".format(meanCostOfSubepoch))
            
            # Release data
            myLiviaNet3D.trainingData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))
            myLiviaNet3D.trainingData_y.set_value(np.zeros([1,1,1,1], dtype="float32"))

            # Get mean cost epoch
            costsOfEpoch.append(meanCostOfSubepoch)

        meanCostOfEpoch =  sum(costsOfEpoch) / float(numberOfSubEpochs)
        
        # Include the epoch cost to the main training cost and update current mean 
        trainingCost.append(meanCostOfEpoch)
        currentMeanCost = sum(trainingCost) / float(str( e_i + 1))
        
        print(" ---------- Training on Epoch #" + str(e_i) + " finished ----------" )
        print(" ---------- Cost of Epoch: {} / Mean training error {}".format(meanCostOfEpoch,currentMeanCost))
        print(" -------------------------------------------------------- " )
        
        # ------------- Update Learning Rate if required ----------------#

        if e_i >= myLiviaNet3D.firstEpochChangeLR :
            if learningRateModifiedEpoch == 0:
                currentLR = myLiviaNet3D.learning_rate.get_value()
                newLR = currentLR / 2.0
                myLiviaNet3D.learning_rate.set_value(newLR)
                print(" ... Learning rate has been changed from {} to {}".format(currentLR, newLR))
                learningRateModifiedEpoch = e_i
            else:
                if (e_i) == (learningRateModifiedEpoch + myLiviaNet3D.frequencyChangeLR):
                    currentLR = myLiviaNet3D.learning_rate.get_value()
                    newLR = currentLR / 2.0
                    myLiviaNet3D.learning_rate.set_value(newLR)
                    print(" ... Learning rate has been changed from {} to {}".format(currentLR, newLR))
                    learningRateModifiedEpoch = e_i
                
        # ---------------------- Start validation ---------------------- #
        
        numberImagesToSegment = len(imageNames_Val)
        print(" ********************** Starting validation **********************")

        # Run over the images to segment   
        for i_d in xrange(numberImagesToSegment) :
            print("-------------  Segmenting subject: {} ....total: {}/{}... -------------".format(names_Val[i_d],str(i_d+1),str(numberImagesToSegment)))
            strideValues = myLiviaNet3D.lastLayer.outputShapeTest[2:]
            
            segmentVolume(myLiviaNet3D,
                          i_d,
                          imageNames_Val,  # Full path
                          names_Val,       # Only image name
                          groundTruthNames_Val,
                          roiNames_Val,
                          imageType,
                          applyPadding,
                          receptiveField, 
                          sampleSize_Train,
                          strideValues,
                          myLiviaNet3D.batch_Size,
                          0 # Validation (0) or testing (1)
                          )
                         
       
        print(" ********************** Validation DONE ********************** ")

        # ------ In this point the training is done at Epoch n ---------#
        # Increase number of epochs trained
        myLiviaNet3D.numberOfEpochsTrained += 1

        #  --------------- Save the model --------------- 
        BASE_DIR = os.getcwd()
        path_Temp = os.path.join(BASE_DIR,'outputFiles')
        netFolderName = os.path.join(path_Temp,myLiviaNet3D.folderName)
        netFolderName  = os.path.join(netFolderName,'Networks')

        modelFileName = netFolderName + "/" + myLiviaNet3D.networkName + "_Epoch" + str (myLiviaNet3D.numberOfEpochsTrained)
        dump_model_to_gzip_file(myLiviaNet3D, modelFileName)
 
        strFinal =  " Network model saved in " + netFolderName + " as " + myLiviaNet3D.networkName + "_Epoch" + str (myLiviaNet3D.numberOfEpochsTrained)
        print  strFinal

    print("................ The whole Training is done.....")
    print(" ************************************************************************************ ")
