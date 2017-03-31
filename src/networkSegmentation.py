import sys

from LiviaNet.startTesting import segmentVolume

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
   
    segmentVolume(networkModelName,configIniName)
    print(" ***************** SEGMENTATION DONE!!! ***************** ")
  
   
   
if __name__ == '__main__':
   networkSegmentation(sys.argv[1:])
