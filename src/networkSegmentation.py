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


from LiviaNet.startTesting import startTesting

def printUsage(error_type):
    if error_type == 1:
        print(" ** ERROR!!: Few parameters used.")
    else:
        print(" ** ERROR!!: Asked to start with an already created network but its name is not specified.")
        
    print(" ******** USAGE ******** ")
    print(" --- argv 1: Name of the configIni file.")
    print(" --- argv 2: Network model name")


def networkSegmentation(argv):
    # Number of input arguments
    #    1: ConfigIniName (for segmentation)
    #    2: Network model name
   
    # Some sanity checks
    
    if len(argv) < 2:
        printUsage(1)
        sys.exit()

    configIniName = argv[0]
    networkModelName = argv[1]
    
    startTesting(networkModelName,configIniName)
    print(" ***************** SEGMENTATION DONE!!! ***************** ")
  
   
   
if __name__ == '__main__':
   networkSegmentation(sys.argv[1:])
