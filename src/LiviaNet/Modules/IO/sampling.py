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

from loadData import load_imagesSinglePatient
from loadData import getRandIndexes
import numpy as np
import math
import random


# **********************************  For Training *********************************
""" This function gets all the samples needed for training a sub-epoch """
def getSamplesSubepoch(numSamples,
                       imageNames,                                                                
                       groundTruthNames,
                       roiNames,
                       imageType,
                       sampleSizes,
                       receptiveField,
                       applyPadding
                       ):
    print (" ... Get samples for subEpoch...")
    
    numSubjects_Epoch = len(imageNames)
    randIdx = getRandIndexes(numSubjects_Epoch, numSubjects_Epoch)

    samplesPerSubject = numSamples/len(randIdx)
    print (" ... getting {} samples per subject...".format(samplesPerSubject)) 
    
    imagesSamplesAll = [] 
    gt_samplesAll = [] 
    
    numSubjectsSubEpoch = len(randIdx) 
    
    samplingDistribution = getSamplesDistribution(numSamples, numSubjectsSubEpoch)
    
    for i_d in xrange(0, numSubjectsSubEpoch) :
        # For displaying purposes
        perc = 100 * float(i_d+1)/numSubjectsSubEpoch
        print("...Processing subject: {}. {} % of the whole training set...".format(str(i_d + 1),perc))

        # -- Load images for a given patient --
        [imgSubject, 
         gtLabelsImage,
         roiMask,
         paddingValues] = load_imagesSinglePatient(randIdx[i_d],
                                                   imageNames,
                                                   groundTruthNames,
                                                   roiNames, 
                                                   applyPadding,
                                                   receptiveField,
                                                   sampleSizes,
                                                   imageType
                                                   )
                                                  

        # -- Get samples for that patient
        [imagesSamplesSinglePatient,
         gtSamplesSinglePatient] = getSamplesSubject(i_d,
                                                     imgSubject,
                                                     gtLabelsImage,
                                                     roiMask,
                                                     samplingDistribution,
                                                     sampleSizes,
                                                     receptiveField
                                                     )

        imagesSamplesAll = imagesSamplesAll + imagesSamplesSinglePatient
        gt_samplesAll = gt_samplesAll + gtSamplesSinglePatient
 
    # -- Permute the training samples so that in each batch both background and objects of interest are taken
    TrainingData = zip(imagesSamplesAll, gt_samplesAll)
    random.shuffle(TrainingData)
    rnd_imagesSamples = []
    rnd_gtSamples = []
    rnd_imagesSamples[:], rnd_gtSamples[:] = zip(*TrainingData)

    del imagesSamplesAll[:]
    del gt_samplesAll[:]

    return rnd_imagesSamples, rnd_gtSamples



def getSamplesSubject(imageIdx,
                      imgSubject,
                      gtLabelsImage,
                      roiMask,
                      samplingDistribution,
                      sampleSizes,
                      receptiveField
                      ):
    sampleSizes = sampleSizes
    imageSamplesSingleImage = []
    gt_samplesSingleImage = []
            
    imgDim = imgSubject.shape

    # Get weight maps for sampling
    weightMaps = getSamplingWeights(gtLabelsImage, roiMask)


    # We are extracting segments for foreground and background
    for c_i in xrange(2) :
        numOfSamplesToGet = samplingDistribution[c_i][imageIdx]
        weightMap = weightMaps[c_i]
        # Define variables to be used
        roiToApply = np.zeros(weightMap.shape, dtype="int32")
        halfSampleDim = np.zeros( (len(sampleSizes), 2) , dtype='int32')


        # Get the size of half patch (i.e sample)
        for i in xrange( len(sampleSizes) ) :
            if sampleSizes[i]%2 == 0: #even
                dimensionDividedByTwo = sampleSizes[i]/2
                halfSampleDim[i] = [dimensionDividedByTwo - 1, dimensionDividedByTwo] 
            else: #odd
                dimensionDividedByTwoFloor = math.floor(sampleSizes[i]/2) 
                halfSampleDim[i] = [dimensionDividedByTwoFloor, dimensionDividedByTwoFloor] 
    
        # --- Set to 1 those voxels in which we are interested in
        # - Define the limits
        roiMinx = halfSampleDim[0][0]
        roiMaxx = imgDim[0] - halfSampleDim[0][1]
        roiMiny = halfSampleDim[1][0]
        roiMaxy = imgDim[1] - halfSampleDim[1][1]
        roiMinz = halfSampleDim[2][0]
        roiMaxz = imgDim[2] - halfSampleDim[2][1]

        # Set
        roiToApply[roiMinx:roiMaxx,roiMiny:roiMaxy,roiMinz:roiMaxz] = 1
        
        maskCoords = weightMap * roiToApply
        
        # We do the following because np.random.choice 4th parameter needs the probabilities to sum 1
        maskCoords = maskCoords / (1.0* np.sum(maskCoords))
    
        maskCoordsFlattened = maskCoords.flatten()
  
        centralVoxelsIndexes = np.random.choice(maskCoords.size,
                                                size = numOfSamplesToGet,
                                                replace=True,
                                                p=maskCoordsFlattened)

        centralVoxelsCoord = np.asarray(np.unravel_index(centralVoxelsIndexes, maskCoords.shape))
        
        coordsToSampleArray = np.zeros(list(centralVoxelsCoord.shape) + [2], dtype="int32")
        coordsToSampleArray[:,:,0] = centralVoxelsCoord - halfSampleDim[ :, np.newaxis, 0 ] #np.newaxis broadcasts. To broadcast the -+.
        coordsToSampleArray[:,:,1] = centralVoxelsCoord + halfSampleDim[ :, np.newaxis, 1 ]

        
        # ----- Compute the coordinates that will be used to extract the samples ---- #
        numSamples = len(coordsToSampleArray[0])

        # Extract samples from computed coordinates
        for s_i in xrange(numSamples) :

            # Get one sample given a coordinate
            coordsToSample = coordsToSampleArray[:,s_i,:] 

            sampleSizes = sampleSizes
            imageSample = np.zeros((1, sampleSizes[0],sampleSizes[1],sampleSizes[2]), dtype = 'float32')

            xMin = coordsToSample[0][0]
            xMax = coordsToSample[0][1] + 1
            yMin = coordsToSample[1][0]
            yMax = coordsToSample[1][1] + 1
            zMin = coordsToSample[2][0]
            zMax = coordsToSample[2][1] + 1

            imageSample[:1] = imgSubject[ xMin:xMax,yMin:yMax,zMin:zMax]
            sample_gt_Orig = gtLabelsImage[xMin:xMax,yMin:yMax,zMin:zMax]

            roiLabelMin = np.zeros(3, dtype = "int8")
            roiLabelMax = np.zeros(3, dtype = "int8")

            for i_x in range(len(receptiveField)) :
                roiLabelMin[i_x] = (receptiveField[i_x] - 1)/2
                roiLabelMax[i_x] = sampleSizes[i_x] - roiLabelMin[i_x]
            
            gt_sample = sample_gt_Orig[roiLabelMin[0] : roiLabelMax[0],
                                       roiLabelMin[1] : roiLabelMax[1],
                                       roiLabelMin[2] : roiLabelMax[2]]
                                        
            imageSamplesSingleImage.append(imageSample)
            gt_samplesSingleImage.append(gt_sample)

    return imageSamplesSingleImage,gt_samplesSingleImage
   


def getSamplingWeights(gtLabelsImage,
                       roiMask
                       ) :

    foreMask = (gtLabelsImage>0).astype(int)
    backMask = (roiMask>0) * (foreMask==0)
    weightMaps = [ foreMask, backMask ] 
 
    return weightMaps
     


def getSamplesDistribution( numSamples,
                            numImagesToSample ) :
    # We have to sample foreground and background
    # Assuming that we extract the same number of samples per category: 50% each

    samplesPercentage = np.ones( 2, dtype="float32" ) * 0.5
    samplesPerClass = np.zeros( 2, dtype="int32" )
    samplesDistribution = np.zeros( [ 2, numImagesToSample ] , dtype="int32" )
    
    samplesAssigned = 0
    
    for c_i in xrange(2) :
        samplesAssignedClass = int(numSamples*samplesPercentage[c_i])
        samplesPerClass[c_i] += samplesAssignedClass
        samplesAssigned += samplesAssignedClass
 
    # Assign the samples that were not assigned due to the rounding error of integer division. 
    nonAssignedSamples = numSamples - samplesAssigned
    classesIDx= np.random.choice(2,
                                 nonAssignedSamples,
                                 True,
                                 p=samplesPercentage)

    for c_i in classesIDx : 
        samplesPerClass[c_i] += 1
        
    for c_i in xrange(2) :
        samplesAssignedClass = samplesPerClass[c_i] / numImagesToSample                
        samplesDistribution[c_i] += samplesAssignedClass
        samplesNonAssignedClass = samplesPerClass[c_i] % numImagesToSample
        for cU_i in xrange(samplesNonAssignedClass):
            samplesDistribution[c_i, random.randint(0, numImagesToSample-1)] += 1

    return samplesDistribution

# **********************************  For testing *********************************

def sampleWholeImage(imgSubject,
                     roi,
                     sampleSize,
                     strideVal,
                     batch_size
                     ):

    samplesCoords = []
 
    imgDims = list(imgSubject.shape)
    
    zMinNext=0
    zCentPredicted = False
    
    while not zCentPredicted :
        zMax = min(zMinNext+sampleSize[2], imgDims[2]) 
        zMin = zMax - sampleSize[2]
        zMinNext = zMinNext + strideVal[2]

        if zMax < imgDims[2]:
            zCentPredicted = False
        else:
            zCentPredicted = True 
        
        yMinNext=0
        yCentPredicted = False
        
        while not yCentPredicted :
            yMax = min(yMinNext+sampleSize[1], imgDims[1]) 
            yMin = yMax - sampleSize[1]
            yMinNext = yMinNext + strideVal[1]

            if yMax < imgDims[1]:
                yCentPredicted = False
            else:
                yCentPredicted = True
            
            xMinNext=0
            xCentPredicted = False
            
            while not xCentPredicted :
                xMax = min(xMinNext+sampleSize[0], imgDims[0])
                xMin = xMax - sampleSize[0]
                xMinNext = xMinNext + strideVal[0]

                if xMax < imgDims[0]:
                    xCentPredicted = False
                else:
                    xCentPredicted = True
                
                if isinstance(roi, (np.ndarray)) : 
                    if not np.any(roi[xMin:xMax, yMin:yMax, zMin:zMax ]) : 
                        continue
                    
                samplesCoords.append([ [xMin, xMax-1], [yMin, yMax-1], [zMin, zMax-1] ])
                
    # To Theano to not complain the number of samples have to exactly fit with the number of batches.
    sampledRegions = len(samplesCoords)

    if sampledRegions%batch_size <> 0:
        numberOfSamplesToAdd =  batch_size - sampledRegions%batch_size
    else:
      numberOfSamplesToAdd = 0
      
    for i in xrange(numberOfSamplesToAdd) :
        samplesCoords.append(samplesCoords[sampledRegions-1])

    return [samplesCoords]


def extractSamples(imgData,
                   sliceCoords,
                   imagePartDimensions,
                   patchDimensions
                   ) :
    numberOfSamples = len(sliceCoords)
    # Create the array that will contain the samples
    samplesArrayShape = [numberOfSamples, 1, imagePartDimensions[0], imagePartDimensions[1], imagePartDimensions[2]]
    samples = np.zeros(samplesArrayShape, dtype= "float32")
    
    for s_i in xrange(numberOfSamples) :
        cMin = []
        cMax = []
        for c_i in xrange(3):
            cMin.append(sliceCoords[s_i][c_i][0])
            cMax.append(sliceCoords[s_i][c_i][1] + 1)
            
        samples[s_i] = imgData[cMin[0]:cMax[0],
                               cMin[1]:cMax[1],
                               cMin[2]:cMax[2]]
                                                                    
    return [samples]
