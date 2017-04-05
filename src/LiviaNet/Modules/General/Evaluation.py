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
import numpy as np

# ----- Dice Score -----
def computeDice(autoSeg, groundTruth):
    """ Returns
    -------
    DiceArray : floats array
          
          Dice coefficient as a float on range [0,1].
          Maximum similarity = 1
          No similarity = 0 """
          
    n_classes = int( np.max(groundTruth) + 1)
   
    DiceArray = []
    
    
    for c_i in xrange(1,n_classes):
        idx_Auto = np.where(autoSeg.flatten() == c_i)[0]
        idx_GT   = np.where(groundTruth.flatten() == c_i)[0]
        
        autoArray = np.zeros(autoSeg.size,dtype=np.bool)
        autoArray[idx_Auto] = 1
        
        gtArray = np.zeros(autoSeg.size,dtype=np.bool)
        gtArray[idx_GT] = 1
        
        dsc = dice(autoArray, gtArray)

        #dice = np.sum(autoSeg[groundTruth==c_i])*2.0 / (np.sum(autoSeg) + np.sum(groundTruth))
        DiceArray.append(dsc)
        
    return DiceArray


def dice(im1, im2):
    """
    Computes the Dice coefficient
    ----------
    im1 : boolean array
    im2 : boolean array
    
    If they are not boolean, they will be converted.
    
    -------
    It returns the Dice coefficient as a float on the range [0,1].
        1: Perfect overlapping 
        0: Not overlapping 
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.size != im2.size:
        raise ValueError("Size mismatch between input arrays!!!")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 1.0

    # Compute Dice 
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum
