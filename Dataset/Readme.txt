
These few subjects used for showing how our CNN works come from the ABIDE repository: 

http://fcon_1000.projects.nitrc.org/indi/abide/

The best way to create your dataset to be loaded in this work, is as follows:
|
|____ MR/
|     |___ All your images (CT/MR/other modality)
|            |__ MR_0
|            |__ MR_1
|            |__ etc..
|
|____ Label/
|     |___ Corresponding labels (Be aware that they should be in the same order than images in previous folder)
