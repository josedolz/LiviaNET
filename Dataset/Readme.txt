
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
|            |__ GT_0
|            |__ GT_1
|            |__ etc...
|
|____ ROI/
|     |___ If you use masks to prone the region of interest, they should be here
|            |__ ROI_0
|            |__ ROI_1
|            |__ etc...
