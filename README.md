# LiviaNET. 3D fully Convolutional Neural Network for semantic image segmentation

This repository contains the code of LiviaNET, a 3D fully convolutional neural network that was employed in our work: [3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study](https://128.84.21.199/abs/1612.03925v1) (In current revision at NeuroImage).

## Requirements

- The code has been written in Python (2.7) and requires [Theano](http://deeplearning.net/software/theano/)
- You should also have installed [scipy](https://www.scipy.org/)
- The code allows to load images in Matlab and Nifti formats. If you wish to use nifti formats you should install [nibabel](http://nipy.org/nibabel/) 

## Running the code

## Training

### How do I train my own architecture from scratch?

To start with your own architecture, you have to modify the file "LiviaNET_Config.ini" according to your requirements.

Then you simply have to write in the command line:

```
python ./networkTraining.py ./LiviaNet/LiviaNET_Config.ini 0
```
This will save, after each epoch, the updated trained model.

### Can I re-start the training from another epoch?

Imagine that after two days of training your model, and just before you have your new model ready to be evaluated, your computer breaks down. Do not panic!!! You will have only to re-start the training from the last epoch in which the model was saved (Let's say epoch 20) as follows:

```
python ./networkTraining.py ./LiviaNet/LiviaNET_Config.ini 1 ./outputFiles/LiviaNet_Test/Networks/liviaTest_Epoch0
```


Current status: Cleaning and commenting files.....

Expected date of release: 2017, April, 7th.


If you are reading this is because I did not still finish to upload all the files. Therefore, the code is incomplete!

### Versions
- April,2th. 2017.


If you use this code for your research, pleae cite the original paper:

- Dolz, J., C. Desrosiers, and I. Ben Ayed. "3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study." arXiv preprint arXiv:1612.03925 (2016).

I strongly encourage to cite the work of Kamnitsas :"Kamnitsas, Konstantinos, et al. "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation." Medical Image Analysis 36 (2017): 61-78.", since this code is based on his previous work.
