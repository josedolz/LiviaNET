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
import numpy.lib as lib
import pdb
import math
import random  

# Get bounding box of a numpy array
def getBoundingBox(img):

    row = np.any(img, axis=(1, 2))
    col = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(row)[0][[0, -1]]
    cmin, cmax = np.where(col)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return (rmin, rmax, cmin, cmax, zmin, zmax)
    
    
# ---------------- Padding ------------------- #
def applyPadding(inputImg, sampleSizes, receptiveField) : 
    receptiveField_arr = np.asarray(receptiveField, dtype="int16")
    inputImg_arr = np.asarray(inputImg.shape,dtype="int16")
   
    receptiveField = np.array(receptiveField, dtype="int16")
    
    left_padding = (receptiveField - 1) / 2
    right_padding = receptiveField - 1 - left_padding
    
    extra_padding = np.maximum(0, np.asarray(sampleSizes,dtype="int16")-(inputImg_arr+left_padding+right_padding))
    right_padding += extra_padding  
    
    paddingValues = ( (left_padding[0],right_padding[0]),
                      (left_padding[1],right_padding[1]),
                      (left_padding[2],right_padding[2]))
                      
    paddedImage = lib.pad(inputImg, paddingValues, mode='reflect' )
    return [paddedImage, paddingValues]

# ----- Apply unpadding ---------
def applyUnpadding(inputImg, paddingValues) :
    unpaddedImg = inputImg[paddingValues[0][0]:, paddingValues[1][0]:, paddingValues[2][0]:]

    if paddingValues[0][1] > 0:
        unpaddedImg = unpaddedImg[:-paddingValues[0][1],:,:]
    
    if paddingValues[1][1] > 0:
        unpaddedImg = unpaddedImg[:,:-paddingValues[1][1],:]
        
    if paddingValues[2][1] > 0:
        unpaddedImg = unpaddedImg[:,:,:-paddingValues[2][1]]
        
    return unpaddedImg
