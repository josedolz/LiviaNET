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
import time
import os
import pdb

from Modules.General.Evaluation import computeDice
from Modules.General.Utils import getImagesSet
from Modules.General.Utils import load_model_from_gzip_file
from Modules.IO.ImgOperations.imgOp import applyUnpadding
from Modules.IO.loadData import load_imagesSinglePatient
from Modules.IO.saveData import saveImageAsNifti
from Modules.IO.saveData import saveImageAsMatlab
from Modules.IO.sampling import *
from Modules.Parsers.parsersUtils import parserConfigIni


def segmentVolume(myNetworkModel,
                  i_d,
                  imageNames_Test,
                  names_Test,
                  groundTruthNames_Test,
                  roiNames_Test,
                  imageType,
                  padInputImagesBool,
                  receptiveField, 
                  sampleSize_Test,
                  strideVal,
                  batch_Size,
                  task # Validation (0) or testing (1)
                  ):
        # Get info from the network model        
        networkName        = myNetworkModel.networkName
        folderName         = myNetworkModel.folderName
        n_classes          = myNetworkModel.n_classes
        sampleSize_Test    = myNetworkModel.sampleSize_Test
        receptiveField     = myNetworkModel.receptiveField  
        outputShape        = myNetworkModel.lastLayer.outputShapeTest[2:] 
        batch_Size         = myNetworkModel.batch_Size
        padInputImagesBool = True
    
        # Get half sample size
        sampleHalf = []
        for h_i in range(3):
            sampleHalf.append((receptiveField[h_i]-1)/2)
        
        # Load the images to segment
        [imgSubject,  
        gtLabelsImage, 
        roi, 
        paddingValues] = load_imagesSinglePatient(i_d,
                                                  imageNames_Test,
                                                  groundTruthNames_Test,
                                                  roiNames_Test,
                                                  padInputImagesBool,
                                                  receptiveField, 
                                                  sampleSize_Test,
                                                  imageType, 
                                                  )
                                                  
                                  
        # Get image dimensions                                                    
        imgDims = list(imgSubject.shape)
    
        [ sampleCoords ] = sampleWholeImage(imgSubject,
                                            roi,
                                            sampleSize_Test,
                                            strideVal,
                                            batch_Size
                                            )
        
        numberOfSamples = len(sampleCoords)
        sampleID = 0
        numberOfBatches = numberOfSamples/batch_Size

        #The probability-map that will be constructed by the predictions.
        probMaps = np.zeros([n_classes]+imgDims, dtype = "float32")
        
        # Run over all the batches 
        for b_i in xrange(numberOfBatches) :
                 
            # Get samples for batch b_i
            
            sampleCoords_b = sampleCoords[ b_i*batch_Size : (b_i+1)*batch_Size ]
            
            [imgSamples] = extractSamples(imgSubject,
                                          sampleCoords_b,
                                          sampleSize_Test,
                                          receptiveField)

            # Load the data of the batch on the GPU
            myNetworkModel.testingData_x.set_value(imgSamples, borrow=True)
           
            # Call the testing Theano function            
            predictions = myNetworkModel.networkModel_Test(0)
            
            predOutput = predictions[-1]
            
            # --- Now we can generate the probability maps from the predictions ----
            # Run over all the regions
            for r_i in xrange(batch_Size) :
 
                sampleCoords_i = sampleCoords[sampleID]
                coords = [ sampleCoords_i[0][0], sampleCoords_i[1][0], sampleCoords_i[2][0] ]

                # Get the min and max coords
                xMin = coords[0] + sampleHalf[0]
                xMax = coords[0] + sampleHalf[0] + strideVal[0]

                yMin = coords[1] + sampleHalf[1]
                yMax = coords[1] + sampleHalf[1] + strideVal[1]

                zMin = coords[2] + sampleHalf[2]
                zMax = coords[2] + sampleHalf[2] + strideVal[2]
                
                probMaps[:,xMin:xMax, yMin:yMax, zMin:zMax] = predOutput[r_i]

                sampleID += 1
            
        # Release data
        myNetworkModel.testingData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))

        # Segmentation has been done in this point.
        
        # Now: Save the data
        # Get the segmentation from the probability maps ---
        segmentationImage = np.argmax(probMaps, axis=0) 
        
        #Save Result:
        npDtypeForPredictedImage = np.dtype(np.int16)
        suffixToAdd = "_Segm"
 
        # Apply unpadding if specified
        if padInputImagesBool == True:
            segmentationRes = applyUnpadding(segmentationImage, paddingValues)
        else:
            segmentationRes = segmentationImage

        # Generate folders to store the model
        BASE_DIR = os.getcwd()
        path_Temp = os.path.join(BASE_DIR,'outputFiles')

        # For the predictions
        predlFolderName = os.path.join(path_Temp,myNetworkModel.folderName)
        predlFolderName = os.path.join(predlFolderName,'Pred')
        if task == 0:
            predTestFolderName = os.path.join(predlFolderName,'Validation')
        else:
            predTestFolderName = os.path.join(predlFolderName,'Testing')
        
        nameToSave = predTestFolderName + '/Segmentation_'+ names_Test[i_d]
        
        # Save Segmentation image
        
        print(" ... Saving segmentation result..."),
        if imageType == 0: # nifti
            imageTypeToSave = np.dtype(np.int16)
            saveImageAsNifti(segmentationRes,
                             nameToSave,
                             imageNames_Test[i_d],
                             imageTypeToSave)
        else: # Matlab
            # Cast to int8 for saving purposes
            saveImageAsMatlab(segmentationRes.astype('int8'),
                              nameToSave)


        # Save the prob maps for each class (except background)
        for c_i in xrange(1, n_classes) :
            
            
            nameToSave = predTestFolderName + '/ProbMap_class_'+ str(c_i) + '_' + names_Test[i_d] 

            probMapClass = probMaps[c_i,:,:,:]

            # Apply unpadding if specified
            if padInputImagesBool == True:
                probMapClassRes = applyUnpadding(probMapClass, paddingValues)
            else:
                probMapClassRes = probMapClass

            print(" ... Saving prob map for class {}...".format(str(c_i))),
            if imageType == 0: # nifti
                imageTypeToSave = np.dtype(np.float32)
                saveImageAsNifti(probMapClassRes,
                                 nameToSave,
                                 imageNames_Test[i_d],
                                 imageTypeToSave)
            else:
                # Cast to float32 for saving purposes
                saveImageAsMatlab(probMapClassRes.astype('float32'),
                                  nameToSave)

        # If segmentation done during evaluation, get dice
        if task == 0:
            print(" ... Computing Dice scores: ")
            DiceArray = computeDice(segmentationImage,gtLabelsImage)
            for d_i in xrange(len(DiceArray)):
                print(" -------------- DSC (Class {}) : {}".format(str(d_i+1),DiceArray[d_i]))

""" Main segmentation function """
def startTesting(networkModelName,
                 configIniName
                 ) :

    padInputImagesBool = True # from config ini
    print " ******************************************  STARTING SEGMENTATION ******************************************"

    print " **********************  Starting segmentation **********************"
    myParserConfigIni = parserConfigIni()
    myParserConfigIni.readConfigIniFile(configIniName,2)
    

    print " -------- Images to segment -------------"

    print " -------- Reading Images names for segmentation -------------"
    
    # -- Get list of images used for testing -- #
    (imageNames_Test, names_Test) = getImagesSet(myParserConfigIni.imagesFolder,myParserConfigIni.indexesToSegment)  # Images
    (groundTruthNames_Test, gt_names_Test) = getImagesSet(myParserConfigIni.GroundTruthFolder,myParserConfigIni.indexesToSegment) # Ground truth
    (roiNames_Test, roi_names_Test) = getImagesSet(myParserConfigIni.ROIFolder,myParserConfigIni.indexesToSegment) # ROI

    # --------------- Load my LiviaNet3D object  --------------- 
    print (" ... Loading model from {}".format(networkModelName))
    myLiviaNet3D = load_model_from_gzip_file(networkModelName)
    print " ... Network architecture successfully loaded...."

    # Get info from the network model        
    networkName        = myLiviaNet3D.networkName
    folderName         = myLiviaNet3D.folderName
    n_classes          = myLiviaNet3D.n_classes
    sampleSize_Test    = myLiviaNet3D.sampleSize_Test
    receptiveField     = myLiviaNet3D.receptiveField  
    outputShape        = myLiviaNet3D.lastLayer.outputShapeTest[2:] 
    batch_Size         = myLiviaNet3D.batch_Size
    padInputImagesBool = myParserConfigIni.applyPadding
    imageType          = myParserConfigIni.imageTypes
    numberImagesToSegment = len(imageNames_Test)
    
    strideValues = myLiviaNet3D.lastLayer.outputShapeTest[2:]

    # Run over the images to segment   
    for i_d in xrange(numberImagesToSegment) :
        print("**********************  Segmenting subject: {} ....total: {}/{}...**********************".format(names_Test[i_d],str(i_d+1),str(numberImagesToSegment)))
        
        segmentVolume(myLiviaNet3D,
                  i_d,
                  imageNames_Test,  # Full path
                  names_Test,       # Only image name
                  groundTruthNames_Test,
                  roiNames_Test,
                  imageType,
                  padInputImagesBool,
                  receptiveField, 
                  sampleSize_Test,
                  strideValues,
                  batch_Size,
                  1 # Validation (0) or testing (1)
                  )
                         
       
    print(" **************************************************************************************************** ")
