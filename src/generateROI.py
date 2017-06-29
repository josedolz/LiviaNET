""" 
Copyright (c) 2017, Jose Dolz .All rights reserved.
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
Jose Dolz. Dec, 2017.
email: jose.dolz.upv@gmail.com
LIVIA Department, ETS, Montreal.
"""

import sys
import pdb
from os.path import isfile, join
import os
import numpy as np
import nibabel as nib
import scipy.io as sio

from LiviaNet.Modules.IO.loadData import load_nii
from LiviaNet.Modules.IO.loadData import load_matlab
from LiviaNet.Modules.IO.saveData import saveImageAsNifti
from LiviaNet.Modules.IO.saveData import saveImageAsMatlab

# NOTE: Only has been tried on nifti images. However, it should not give any error for Matlab images. 
""" To print function usage """
def printUsage(error_type):
    if error_type == 1:
        print(" ** ERROR!!: Few parameters used.")
    else:
        print(" ** ERROR!!: ...") # TODO
        
    print(" ******** USAGE ******** ")
    print(" --- argv 1: Folder containing mr images")
    print(" --- argv 2: Folder to save corrected label images")
    print(" --- argv 3: Image type")
    print(" ------------- 0: nifti format")
    print(" ------------- 1: matlab format")

def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames
    
def checkAnotatedLabels(argv):
    # Number of input arguments
    #    1: Folder containing label images
    #    2: Folder to save corrected label images
    #    3: Image type
    #             0: nifti format
    #             1: matlab format  
    # Do some sanity checks
    
    if len(argv) < 3:
        printUsage(1)
        sys.exit()
    
    imagesFolder = argv[0]
    imagesFolderdst  = argv[1]
    imageType = int(argv[2])
    
    imageNames = getImageImageList(imagesFolder)
    printFileNames = False
    
    for i_d in xrange(len(imageNames)) :
        if imageType == 0:
            imageFileName = imagesFolder + '/' + imageNames[i_d]
            [imageData,img_proxy] = load_nii(imageFileName, printFileNames)
        else:
            imageFileName = imagesFolder + '/' + imageNames[i_d]
            imageData = load_matlab(imageFileName, printFileNames)

        # Find voxels different to 0
        # NOTE: I assume voxels equal to 0 are outside my ROI (like in the skull stripped datasets)
        idx = np.where(imageData > 0 )

        # Create ROI and assign those indexes to 1
        roiImage = np.zeros(imageData.shape,dtype=np.int8)
        roiImage[idx] = 1
        
        print(" ... Saving roi...")
        nameToSave =  imagesFolderdst + '/ROI_' + imageNames[i_d]
        if imageType == 0: # nifti
            imageTypeToSave = np.dtype(np.int8)
            saveImageAsNifti(roiImage,
                             nameToSave,
                             imageFileName,
                             imageTypeToSave)
        else: # Matlab
            # Cast to int8 for saving purposes
            saveImageAsMatlab(labelCorrectedImage.astype('int8'),nameToSave)

            
    print " ******************************************  PROCESSING LABELS DONE  ******************************************"
  
   
if __name__ == '__main__':
   checkAnotatedLabels(sys.argv[1:])
