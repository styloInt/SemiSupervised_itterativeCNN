# SemiSupervised/itterativeCNN

## Install caffe : 
 https://github.com/BVLC/caffe/
 
 ## Train Your network with the labelled data
 
 You have to adapt the paths and the parameters in config_train.py for your application and start the training with :
 python train.py
 
 ## Test your network
  You have to adapt the paths and the parameters in config_test.py for your application and start the testing with :
  python test.py -f ../test.txt -r ../ -d ../predictions_test/ 
  With test.txt the file containing the list of test files in nifti format. and "predictions_test" the repertory where you want to save the results
 
 ## Use the unlabelled data to improve it
 Before starting you should finetune th graphcut parameters for you dataset. This step is important in order to improve the results givven by the CNN.
 
 
 
 ## Test the itterative CNN 
