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
