import sys
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import os.path
import argparse
from medpy.io import load, header, save
from utils_dataRV import *
from config_test import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
            action="store", dest="files",
            help="files we want to compute the segmentation")
parser.add_argument('-r', '--rep',
            action="store", dest="rep_dataset",
            help="repertory of the dataset")
parser.add_argument('-e', '--epoch',
            action="store", dest="epoch",
            help="after which epoch you want to compute")
parser.add_argument('-d', '--directory',
            action="store", dest="directorySave",
            help="directory where to save the results")

args = parser.parse_args()
filenames = args.files
directorySave = args.directorySave
rep_dataset = args.rep_dataset
num_epoch = int(args.epoch)


if num_epoch == 0:
	weight_file = weight_file
else:
	weight_file = prefix_snapshot + "_iter_" + str((num_epoch) * k) + ".caffemodel"

net_deploy = caffe.Net(net_deploy_name,      # defines the structure of the model
	                weight_file, caffe.TEST)

file_dataset = open(filenames)
for patientfile in file_dataset.read().splitlines():
	split_line = patientfile.split("\t")
	inputs_name = split_line[0]
	image_data, header = load(rep_dataset + "/" + inputs_name)
	nb_slices = image_data.shape[2]

	basename = os.path.basename(inputs_name)
	print ("Prediction for : ", basename)

	size_im = preprocessing_im(image_data[:,:,0]).shape

	mri3D = np.zeros((size_im[0], size_im[1], nb_slices))
	gt3D = np.zeros((size_im[0], size_im[1], nb_slices))
	hm3D = np.zeros((size_im[0], size_im[1], 2, nb_slices))

	for i in range(image_data.shape[2]):
		mri3D[:,:,i] = preprocessing_im(image_data[:,:,i])
		mri3D[:,:, i] -= np.min(mri3D[:,:, i])
		mri3D[:,:, i] = mri3D[:,:,i] / np.max(mri3D[:,:,i])
		hm3D[:,:,:,i] = get_heat_map(mri3D[:,:, i], net_deploy)

	output_file = directorySave + "/" + basename.replace(".nii", "_e{}_{}.nii".format(num_epoch, suffixe))
	createDirectoryPath(output_file)
	save(hm3D.argmax(2).astype(np.uint8), output_file,hdr=header, force=True)


file_dataset.close()
net_deploy=None


