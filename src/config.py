import sys
import os
path_caffe = '/home/atemmar/caffe/';
sys.path.insert(0, path_caffe + '/python')
import caffe

k = int(sys.argv[1]) # Number of itteration
alpha_fg = float(sys.argv[2]) # Weight loss for the foreground predicted labels
alpha_bg = float(sys.argv[3]) # Weight loss for the background predicted labels


# The different paths we need
rep_dataset = "/home/atemmar/Documents/Data/RV_Segmentation_volume/"

orginal_net = rep_dataset + "/src/models/U-net/unet.prototxt" # The network you used for the training
original_solver_name = rep_dataset + "/src/models/U-net/solver_unet_softmax_iter.prototxt" # The solver you used for the training.

original_train_filenames = rep_dataset + "train.txt"
unlab_filename = rep_dataset + 'unlab.txt'
test_filename = rep_dataset + 'test.txt'

weight_file = rep_dataset + "/src/models_pretrained/U-net_noNorm/train_unet_rv_softmax_vp_iter_374000.caffemodel" # The pretraining model after the first training
net_deploy_name = rep_dataset + '/src/models/U-net/unet_deploy_vp.prototxt' # The deploy network

epoch_start = 0 # If your training crashed and you want to start from a specific epoch, change this parameter
number_epoch = 10 # Number of epochs you want to do
increment = 0 # if you to increment alpha_fg et alpha_bg after each itteration

# Graphcut parameter. Change depending on the dataset
postProc = "graphcut3D" # postProcessing you want to use
sigma = 0.04
lamda = 0.05
pWeightMax=0.95

suffixe = str(k) + "_" + str(alpha_fg) + "_" + str(alpha_bg) + "_" + postProc + "_" + str(increment)
if postProc == "graphcut3D" or postProc=="graphcut2D":
	suffixe = str(k) + "_" + str(alpha_fg) + "_" + str(alpha_bg) + "_" + postProc + "_" + str(sigma) + "_" + str(lamda) + "_" + str(pWeightMax) + "_" + str(increment)


prefix_snapshot = rep_dataset + "/src/models_pretrained/U-net_iter_vnoNorm/train_unet_spine_softmax_" + suffixe+"_iter" # SnapShot prefix to train the pretrained model
dice_saveUnlab_filename = rep_dataset + "diceResults/dicesResuls_unlab_" + suffixe + ".txt" # Save the dice result for the unlabelled dataset if you have the ground truth
dice_saveTest_filename = rep_dataset + "diceResults/dicesResuls_test_" + suffixe + ".txt" # Save the dice result for the test set.


# Levels of output caffe
# 0 - debug
# 1 - info (still a LOT of outputs)
# 2 - warnings
# 3 - errors
os.environ['GLOG_minloglevel'] = '2' 



