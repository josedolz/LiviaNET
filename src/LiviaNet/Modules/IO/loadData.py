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
import pdb
# If you are not using nifti files you can comment this line
import nibabel as nib
import scipy.io as sio

from ImgOperations.imgOp import applyPadding

# ----- Loader for nifti files ------ #
def load_nii (imageFileName, printFileNames) :
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()
    
    return (imageData,img_proxy)
    
def release_nii_proxy(img_proxy) :
    img_proxy.uncache()


# ----- Loader for matlab format ------- #
# Very important: All the volumes should have been saved as 'vol'.
# Otherwise, change its name here
def load_matlab (imageFileName, printFileNames) :
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))
    
    mat_contents = sio.loadmat(imageFileName)
    imageData = mat_contents['vol']
    
    return (imageData)
    
""" It loads the images (CT/MRI + Ground Truth + ROI) for the patient image Idx"""
def load_imagesSinglePatient(imageIdx, 
                             imageNames, 
                             groundTruthNames, 
                             roiNames,
                             applyPaddingBool,
                             receptiveField, 
                             sampleSizes,
                             imageType
                             ):
    
    if imageIdx >= len(imageNames) :
        print (" ERROR!!!!! : The image index specified is greater than images array size....)")
        exit(1)
    
    # --- Load image data (CT/MRI/...) ---
    printFileNames = False # Get this from config.ini

    imageFileName = imageNames[imageIdx]

    if imageType == 0:
        [imageData,img_proxy] = load_nii(imageFileName, printFileNames)
    else:
        imageData = load_matlab(imageFileName, printFileNames)
        
    if applyPaddingBool == True : 
        [imageData, paddingValues] = applyPadding(imageData, sampleSizes, receptiveField)
    else:
        paddingValues = ((0,0),(0,0),(0,0))


    if len(imageData.shape) > 3 :
         imageData = imageData[:,:,:,0]
    
    if imageType == 0:
        release_nii_proxy(img_proxy)
    
    # --- Load ground truth (i.e. labels) ---
    if len(groundTruthNames) > 0 : 
        GTFileName = groundTruthNames[imageIdx]
        
        if imageType == 0:
            [gtLabelsData, gt_proxy] = load_nii (GTFileName, printFileNames)
        else:
            gtLabelsData = load_matlab(GTFileName, printFileNames)
        
        # Convert ground truth to int type
        if np.issubdtype( gtLabelsData.dtype, np.int ) : 
            gtLabelsData = gtLabelsData 
        else: 
            np.rint(gtLabelsData).astype("int32")
        
        imageGtLabels = gtLabelsData
        
        if imageType == 0:
            # Release data
            release_nii_proxy(gt_proxy)
        
        if applyPaddingBool == True : 
            [imageGtLabels, paddingValues] = applyPadding(imageGtLabels,  sampleSizes, receptiveField)
        
    else : 
        imageGtLabels = np.empty(0)
        
    # --- Load roi ---
    if len(roiNames)> 0 :
        roiFileName = roiNames[imageIdx]
        
        if imageType == 0:
            [roiMaskData, roi_proxy] = load_nii (roiFileName, printFileNames)
        else:
            roiMaskData = load_matlab(roiFileName, printFileNames)
            
        roiMask = roiMaskData
        
        if imageType == 0:
            # Release data
            release_nii_proxy(roi_proxy)
        
        if applyPaddingBool == True : 
            [roiMask, paddingValues] = applyPadding(roiMask, sampleSizes, receptiveField)
    else :
        roiMask = np.ones(imageGtLabels.shape)

    return [imageData, imageGtLabels, roiMask, paddingValues]


# -------------------------------------------------------- #
def getRandIndexes(total, maxNumberIdx) :
    # Generate a shuffle array of a vector containing "total" elements
    idxs = range(total)
    np.random.shuffle(idxs)
    rand_idxs = idxs[0:maxNumberIdx]
    return rand_idxs

