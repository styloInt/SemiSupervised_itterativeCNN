import sys
import os
# Levels of output caffe
# 0 - debug
# 1 - info (still a LOT of outputs)
# 2 - warnings
# 3 - errors
os.environ['GLOG_minloglevel'] = '3' 

path_caffe = '/home/atemmar/caffe/';
sys.path.insert(0, path_caffe + '/python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

rep_dataset = "/home/atemmar/Documents/Data/RV_Segmentation_volume/"
solver_name =  rep_dataset + "/src/models/U-net_v2/solver_unet_softmax.prototxt"

filenames_validation = rep_dataset + "/test.txt"
net_deploy_name = rep_dataset + '/src/models/U-net_v2/unet_deploy_vp.prototxt' # The deploy network

nb_itteration = 200000
nb_classes = 2

do_test = True


usePretrainedModel = 1 # 0 if not, 1 if you want to restart the training from a solverstate file, 2 if you want to use a pretrained model (caffemodel file)
pretrainedModel = rep_dataset + "/src/models_pretrained/U-net_noNorm_v2/train_unet_rv_softmax_vp_iter_160000.solverstate"
