# LiviaNET. 3D fully Convolutional Neural Network for semantic image segmentation

This repository contains the code of LiviaNET, a 3D fully convolutional neural network that was employed in our work: [3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study](https://128.84.21.199/abs/1612.03925v1) (In current revision at NeuroImage).

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
- Version 1.0. April,2th. 2017.
  * Features:
    * Several activation functions supported.
    * Stochastic gradient descent and RmsProp optimizers.
    * Images in Matlab and Nifti format supported. (To use matlab files, every image should be saved as 'vol').
    * Loading of pre-trained weights at different layers.
    * Connection of intermediate conv layers to the first fully connected layers (i.e. multi-scale fetures).


If you use this code for your research, please consider citing the original paper:

- Dolz J, Desrosiers C, Ben Ayed I. "[3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study."](https://128.84.21.199/abs/1612.03925v1) arXiv preprint arXiv:1612.03925 (2016)

I strongly encourage to cite also the work of Kamnitsas :"Kamnitsas, Konstantinos, et al. ["Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation."](http://www.sciencedirect.com/science/article/pii/S1361841516301839) Medical Image Analysis 36 (2017): 61-78.", since this code is based on his previous work.

### Known problems
* In some computers I tried, when running in CPU, it complains about the type of some tensors. The work-around I have found is just to set some theano flags at the beginning of the scripts. Something like:

```
THEANO_FLAGS='floatX=float32' python ./networkTraining.py ./LiviaNET_Config.ini 0
```
