# SemiSupervised/itterativeCNN

This work was done during the internship I did for my master thesis at Livia laboratory. You can find my report here : 

## Install caffe : 
 https://github.com/BVLC/caffe/
 
 Add the files in modif_caffe in the base of your caffe directory (it's the code for the weighted loss) and build them using the "make" command.
 
 ## Train Your network with the labelled data
 
 You have to adapt the paths and the parameters in config_train.py for your application. Also if you want to do a specific preprocessing, you can do it in preprocessing_im() and preprocessing_label() (in the file utils_dataRV) and start the training with :
 ```
 python train.py
 ```
 ## Test your network
  You have to adapt the paths and the parameters in config_test.py for your application and start the testing with :
  ```
  python test.py -f ../test.txt -r ../ -d ../predictions_test/ 
  ```
  
  With test.txt the file containing the list of test files in nifti format. and "predictions_test" the repertory where you want to save the results
 
 ## Use the unlabelled data to improve it
 Before starting you should finetune th graphcut parameters for you dataset. This step is important in order to improve the results givven by the CNN with the 3D graphcut.
 
 you have to change the paths and parameters in config.py
 
To start the itterative algorithm, you have to do :

```
python train_loop.py 8000 0.4 0.1
```

if you want to do 8000 cnn itterations and use a foreground loss of 0.4 and background loss of 0.1
