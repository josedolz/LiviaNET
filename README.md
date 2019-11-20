# LiviaNET. 3D fully Convolutional Neural Network for semantic image segmentation

- A new pytorch version has been implemented [here](https://github.com/josedolz/LiviaNet_pytorch)

This repository contains the code of LiviaNET, a 3D fully convolutional neural network that was employed in our work: [3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study](http://www.sciencedirect.com/science/article/pii/S1053811917303324) Accepted in Neuroimage, April,17th 2017.

## Requirements

- The code has been written in Python (2.7) and requires [Theano](http://deeplearning.net/software/theano/)
- You should also have installed [scipy](https://www.scipy.org/)
- (Optional) The code allows to load images in Matlab and Nifti formats. If you wish to use nifti formats you should install [nibabel](http://nipy.org/nibabel/) 

## Running the code

## Training

### How do I train my own architecture from scratch?

To start with your own architecture, you have to modify the file "LiviaNET_Config.ini" according to your requirements.

Then you simply have to write in the command line:

```
python ./networkTraining.py ./LiviaNET_Config.ini 0
```

This will save, after each epoch, the updated trained model.

If you use GPU, after nearly 5 minutes you will have your trained model from the example.

### Can I re-start the training from another epoch?

Imagine that after two days of training your model, and just before you have your new model ready to be evaluated, your computer breaks down. Do not panic!!! You will have only to re-start the training from the last epoch in which the model was saved (Let's say epoch 20) as follows:

```
python ./networkTraining.py ./LiviaNET_Config.ini 1 ./outputFiles/LiviaNet_Test/Networks/liviaTest_Epoch20
```

### Ok, cool. And what about employing pre-trained models?

Yes, you can also do that. Instead of loading a whole model, which limits somehow the usability of loading pre-trained models, this code allows to load weights for each layer independently. Therefore, weights for each layer have to be saved in an independent file. In its current version (v1.0) weights files must be in numpy format (.npy).

For that you will have to specify in the "LiviaNET_Config.ini" file the folder where the weights are saved ("weights folderName") and in which layers you want to use transfer learning ("weights trained indexes").

## Testing

### How can I use a trained model?

Once you are satisfied with your training, you can evaluate it by writing this in the command line:

```
python ./networkSegmentation.py ./LiviaNET_Segmentation.ini ./outputFiles/LiviaNet_Test/Networks/liviaTest_EpochX
```
where X denotes the last (or desired) epoch in which the model was saved.

### Versions
- Version 1.0. 
  * June,27th. 2017
    * Feature added:
      * Function to generate the ROI given the images to segment.
  * May,10th. 2017
    * Feature added:
      * Functionality to process your labels for training (See important notes).
  * April,2th. 2017.
    * Features:
      * Several activation functions supported.
      * Stochastic gradient descent and RmsProp optimizers.
      * Images in Matlab and Nifti format supported. 
      * Loading of pre-trained weights at different layers.
      * Connection of intermediate conv layers to the first fully connected layers (i.e. multi-scale fetures).
      * Frequency of changes on learning rate customizable.
      * Note. This version includes Batch Normalization, which was not used in the Neuroimage paper.


If you use this code for your research, please consider citing the original paper:

- Dolz J, Desrosiers C, Ben Ayed I. "[3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study."](http://www.sciencedirect.com/science/article/pii/S1053811917303324) NeuroImage (2017).

I strongly encourage to cite also the work of Kamnitsas :"Kamnitsas, Konstantinos, et al. ["Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation."](http://www.sciencedirect.com/science/article/pii/S1361841516301839) Medical Image Analysis 36 (2017): 61-78.", since this code is based on his previous work, DeepMedic.

### Important notes
* In order to correctly run the training, the convnet needs that training labels are provided in a consecutive manner. This means that the first class must be label 0, the second class label 1, and so on. To ease this process I have included a functionality that takes all the images contained in a given folder and automatically corrects labels to be 0,1,2,etc. To do this, you should proceed as follows:

```
python processLabels.py ~yourpath/Training/LabelsNonCorrected ~yourpath/Training/LabelsCorrected 9 0

```
where 9 is the number of expected classes and 0 is the format (nifti in this case).

## Some results from our paper

* Different views of a smoothed version of contours provided by our automatic segmentation system. In these images, the thalamus, caudate, putamen and pallidum are respectively depicted in yellow, cyan, red and green.

<br>
<img src="https://github.com/josedolz/LiviaNET/blob/master/Images/NeuroRes2.jpg" />
<br>

* Feature map activations in all convolutional layers of the FCNN (right), obtained for a given patch of the input MRI image (left). Each column corresponds to a different convolutional layer, from shallow to deeper, and each image in a row to a features map activation randomly selected in the layer.
<br>
<img src="https://github.com/josedolz/LiviaNET/blob/master/Images/NeuroFeatMaps.jpg" />
<br>



### Known problems
* In some computers I tried, when running in CPU, it complains about the type of some tensors. The work-around I have found is just to set some theano flags at the beginning of the scripts. Something like:

```
THEANO_FLAGS='floatX=float32' python ./networkTraining.py ./LiviaNET_Config.ini 0
```

You can contact me at: jose.dolz.upv@gmail.com


### Other implementations:
- * A version of the network in Keras has been implemented in: "joseabernal/iSEG2017"(https://github.com/joseabernal/iSeg2017-nic_vicorob)
- * Another Keras version of this network can be found in : "Deep Gray Matter (DGM) Segmentation using 3D Convolutional Neural Network: application to QSM"(https://github.com/zl376/segDGM_CNN)

