import sys
from config_test import *

import numpy as np
import os
import re
import argparse
from medpy.io import load, header, save
from utils_dataRV import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
            action="store", dest="files",
            help="files we want to compute the segmentation")
parser.add_argument('-r', '--rep',
            action="store", dest="rep_dataset",
            help="repertory of the dataset")
parser.add_argument('-d', '--directory',
            action="store", dest="directorySave",
            help="directory where to save the results")

args = parser.parse_args()
filenames = args.files
directorySave = args.directorySave
rep_dataset = args.rep_dataset

mri3D, gt3D = load_dataset(filenames_test, rep_dataset)


print ("Begening of testing ...")
print ("Predictions will be save in {}".format(directorySave))

net_deploy = caffe.Net(net_deploy_name,      # defines the structure of the model
                pretrainedModel, caffe.TEST)

hm3D = {}
dice = []
file_dataset = open(filenames)
for patientfile in file_dataset.read().splitlines():
	split_line = patientfile.split("\t")
	inputs_name = split_line[0]
	image_data, header = load(rep_dataset + "/" + inputs_name)
	output_name = split_line[1]
	gt_data, header = load(rep_dataset + "/" + output_name)
	nb_slices = gt_data.shape[2]

	basename = os.path.basename(inputs_name)

	size_im = preprocessing_im(image_data[:,:,0]).shape

	mri3D = np.zeros((size_im[0], size_im[1], nb_slices))
	gt3D = np.zeros((size_im[0], size_im[1], nb_slices))
	hm3D = np.zeros((size_im[0], size_im[1], 2, nb_slices))

	for i in range(nb_slices):
		mri3D[:,:,i] = preprocessing_im(image_data[:,:,i])
		mri3D[:,:, i] -= np.min(mri3D[:,:, i])
		mri3D[:,:, i] = mri3D[:,:,i] / np.max(mri3D[:,:,i])
		hm3D[:,:,:,i] = get_heat_map(mri3D[:,:, i], net_deploy)
		gt3D[:,:,i] = preprocessing_label(gt_data[:,:,i])

	output_file = directorySave + "/" + basename.replace(".nii", "_pred.nii")
	createDirectoryPath(output_file)
	save(hm3D.argmax(2).astype(np.uint8), output_file,hdr=header, force=True)

	dice.append(np.mean(compute_dice_dataset(gt3D, hm3D.argmax(2), countBgImgs=True)))
	
	print("Dice for the file {} : {}".format(basename, dice[-1]))


net_deploy = None # free the memory
file_dataset.close()


print ("Mean Dice on the test set : {}".format(np.mean(dice)))
