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
import pdb
import numpy

from LiviaNet.generateNetwork import generateNetwork
from LiviaNet.startTraining import startTraining

""" To print function usage """
def printUsage(error_type):
    if error_type == 1:
        print(" ** ERROR!!: Few parameters used.")
    else:
        print(" ** ERROR!!: Asked to start with an already created network but its name is not specified.")
        
    print(" ******** USAGE ******** ")
    print(" --- argv 1: Name of the configIni file.")
    print(" --- argv 2: Type of training:")
    print(" ------------- 0: Create a new model and start training")
    print(" ------------- 1: Use an existing model to keep on training (Requires an additional input with model name)")
    print(" --- argv 3: (Optional, but required if arg 2 is equal to 1) Network model name")


def networkTraining(argv):
    # Number of input arguments
    #    1: ConfigIniName
    #    2: TrainingType
    #             0: Create a new model and start training
    #             1: Use an existing model to keep on training (Requires an additional input with model name)
    #    3: (Optional, but required if arg 2 is equal to 1) Network model name
   
    # Do some sanity checks
    
    if len(argv) < 2:
        printUsage(1)
        sys.exit()
    
    configIniName = argv[0]
    trainingType  = argv[1]
    
    if trainingType == '1' and len(argv) == 2:
        printUsage(2)
        sys.exit()
        
    if len(argv)>2:
        networkModelName = argv[2]
   
    # Creating a new model 
    if trainingType == '0':
        print " ******************************************  CREATING NETWORK ******************************************"
        networkModelName = generateNetwork(configIniName)
        print " ******************************************  NETWORK CREATED ******************************************"

    # Training the network in model name
    print " ******************************************  STARTING NETWORK TRAINING ******************************************"
    startTraining(networkModelName,configIniName)
    print " ******************************************  DONE  ******************************************"
  
   
if __name__ == '__main__':
   networkTraining(sys.argv[1:])
