# HookNet
## Multi-resolution convolutional neural networks for semantic segmentation in histopathology whole-slide images.

#### Training

##### dependecies
 - This code has been tested on Ubuntu 18.04, keras==2.0.8 and tensorflow-gpu==1.14
 
##### Examples
 - train.py in this repository will train HookNet on random values. Please adjust the script with your own batchgenerator or sampling function. 
 - For an elaborate explanantion how to use your own batchgenerator, sampling function or how to use our developed batchgenerator for sampling input patches from WSIs, please see train.ipynb in the notebook folder. This notebook also includes an eloborate explanantion about the possible settings for HookNet. 
 

#### Inference

##### dependecies
 - Inference depends on the python api multiresolutionimageinterface.py from ASAP (https://github.com/computationalpathologygroup/ASAP/releases).
 
 ##### Examples
  - apply.py in this repository will apply a trained hooknet on a WSI. 
  
### Additional Information
  
For more information, please check the code comments and the doc strings. If you happen to experience any problems, have questions, or would like to give feedback, feel free to open an issue.

A pretraind model on breast or colon can used on https://grand-challenge.org/. Please create an user account and request access to an algorith if you are interested.
You can try out a pretrained HookNet on breast tissue here:  
https://grand-challenge.org/algorithms/hooknet/  
You can try out a pretrained HookNet on colon tissue here:  
https://grand-challenge.org/algorithms/hooknet-colon/

