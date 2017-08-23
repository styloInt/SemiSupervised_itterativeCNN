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

net_deploy_name = rep_dataset + '/src/models/U-net_v2/unet_deploy_vp.prototxt' # The deploy network

nb_classes = 2

do_test = True


# pretrainedModel = rep_dataset + "/src/models_pretrained/U-net_noNorm_v2/train_unet_rv_softmax_vp_iter_240000.caffemodel"
pretrainedModel = rep_dataset + "/src/models_pretrained/U-net_full/train_unet_rv_softmax_iter_158000.caffemodel"
