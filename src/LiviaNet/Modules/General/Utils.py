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
import gzip
import cPickle
import sys
from os.path import isfile, join


# https://github.com/Theano/Theano/issues/689
sys.setrecursionlimit(50000)

# To set a learning rate at each layer
def extendLearningRateToParams(numberParamsPerLayer,learning_rate):
    if not isinstance(learning_rate, list):
        learnRates = np.ones(sum(numberParamsPerLayer), dtype = "float32") * learning_rate
    else:
        print("")
        learnRates = []
        for p_i in range(len(numberParamsPerLayer)) :
            for lr_i in range(numberParamsPerLayer[p_i]) :
                learnRates.append(learning_rate[p_i])
    return learnRates
# TODO: Check that length of learning rate (in config ini) actually corresponds to length of layers (CNNs + FCs + SoftMax)


def computeReceptiveField(kernelsCNN, stride) :
    # To-do. Verify receptive field with stride size other than 1
	if len(kernelsCNN) == 0:
		return 0
	
	# Check number of ConvLayers
	numberCNNLayers = []

	for l_i in range(1,len(kernelsCNN)):
		if len(kernelsCNN[l_i]) == 3:
			numberCNNLayers = l_i + 1
              
	kernelDim = len(kernelsCNN[0])
	receptiveField = [stride]*kernelDim

	for d_i in xrange(kernelDim) :
		for l_i in xrange(numberCNNLayers) :
			receptiveField[d_i] += kernelsCNN[l_i][d_i] - 1

	return receptiveField



###########################################################
######## Create bias and include them on feat maps ########
###########################################################

# TODO. Remove number of FeatMaps
def addBiasParametersOnFeatureMaps( bias, featMaps, numberOfFeatMaps ) :
    output = featMaps + bias.dimshuffle('x', 0, 'x', 'x', 'x')
    return (output)

###########################################################
########         Initialize CNN weights            ########
###########################################################
def initializeWeights(filter_shape, initializationMethodType, weights) :
    # filter_shape:[#FMs in this layer, #FMs in input, KernelDim_0, KernelDim_1, KernelDim_2]
    def Classic():
        print " --- Weights initialization type: Classic "
        rng = np.random.RandomState(24575)
        stdForInitialization = 0.01
        W = theano.shared(
            np.asarray(
                rng.normal(loc=0.0, scale=stdForInitialization, size=(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3],filter_shape[4])),
                dtype='float32'#theano.config.floatX
                ),
            borrow=True
        )
        return W
        
    def Delving():
       # https://arxiv.org/pdf/1502.01852.pdf
       print " --- Weights initialization type: Delving "
       rng = np.random.RandomState(24575)
       stdForInitialization = np.sqrt( 2.0 / (filter_shape[1] * filter_shape[2] * filter_shape[3] * filter_shape[4]) ) #Delving Into rectifiers suggestion.
       W = theano.shared(
           np.asarray(
               rng.normal(loc=0.0, scale=stdForInitialization, size=(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3],filter_shape[4])),
               dtype='float32'#theano.config.floatX
               ),
           borrow=True
       )
       return W

    # TODO: Add checks so that weights and kernel have the same shape
    def Load():
       print " --- Weights initialization type: Transfer learning... "
       W = theano.shared(
           np.asarray(
               weights,
               dtype=theano.config.floatX
               ),
           borrow=True
       )
       return W

    optionsInitWeightsType = {0 : Classic,
                              1 : Delving,
                              2 : Load}

    W = optionsInitWeightsType[initializationMethodType]()
        
    return W


def getCentralVoxels(sampleSize, receptiveField) :
    centralVoxels = []
    for d_i in xrange(0, len(sampleSize)) :
        centralVoxels.append(sampleSize[d_i] - receptiveField[d_i] + 1)
    return centralVoxels


 
def extractCenterFeatMaps(featMaps, featMaps_shape, centralVoxels) :

    centerValues = []
    minValues = []
    maxValues = []
    
    for i in xrange(3) :
        C_v = (featMaps_shape[i + 2] - 1) / 2
        min_v = C_v - (centralVoxels[i]-1)/2
        max_v = min_v + centralVoxels[i]
        centerValues.append(C_v)
        minValues.append(min_v)
        maxValues.append(max_v)
        
    return featMaps[:,
                    :,
                    minValues[0] : maxValues[0],
                    minValues[1] : maxValues[1],
                    minValues[2] : maxValues[2]]
                    

###########################################
############# Save/Load models ############
###########################################

def load_model_from_gzip_file(modelFileName) :
    f = gzip.open(modelFileName, 'rb')
    model_obj = cPickle.load(f)
    f.close()
    return model_obj

def dump_model_to_gzip_file(model, modelFileName) :
    # First release GPU memory
    model.releaseGPUData()
    
    f = gzip.open(modelFileName, 'wb')
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    return modelFileName

def makeFolder(folderName, display_Str) :
    if not os.path.exists(folderName) :
        os.makedirs(folderName)

    strToPrint = "..Folder " + display_Str + " created..."
    print strToPrint


    from os import listdir


""" Get a set of images from a folder given an array of indexes """
def getImagesSet(imagesFolder, imageIndexes) :
   imageNamesToGetWithFullPath = []
   imageNamesToGet = []
   
   if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]
       imageNames.sort()
   
       # Remove corrupted files (if any)
       if '.DS_Store' in imageNames: imageNames.remove('.DS_Store')

       imageNamesToGetWithFullPath = []
       imageNamesToGet = []
  
       if ( len(imageNames) > 0):  
           imageNamesToGetWithFullPath = [join(imagesFolder,imageNames[imageIndexes[i]]) for i in range(0,len(imageIndexes))]
           imageNamesToGet = [imageNames[imageIndexes[i]] for i in range(0,len(imageIndexes))]

   return (imageNamesToGetWithFullPath,imageNamesToGet)



"""" Get a set of weights from a folder given an array of indexes """
def getWeightsSet(weightsFolder, weightsIndexes) :
   weightNames = [f for f in os.listdir(weightsFolder) if isfile(join(weightsFolder, f))]
   weightNames.sort()
   
   # Remove corrupted files (if any)
   if '.DS_Store' in weightNames: weightNames.remove('.DS_Store')
 
   weightNamesToGetWithFullPath = [join(weightsFolder,weightNames[weightsIndexes[i]]) for i in range(0,len(weightsIndexes))]

   return (weightNamesToGetWithFullPath)
