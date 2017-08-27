import sys

from config_test_loop import *
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import os.path
import argparse
from medpy.io import load, header, save
from utils_dataSpine import *

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
parser.add_argument('-dirDice', '--dirDice',
            action="store", dest="dirDice",
            help="directory where to save the dice results")

args = parser.parse_args()
filenames = args.files
directorySave = args.directorySave
rep_dataset = args.rep_dataset
dirDice = args.dirDice

dice_saveTest_filename = dirDice + "/dicesResuls_test_" + suffixe + ".txt"

createDirectoryPath(dice_saveTest_filename)
dice_saveTest = open(dice_saveTest_filename, "a")

for num_epoch in range(epoch_start, number_epoch):
	if num_epoch == 0:
		weight_file = weight_file
	else:
		weight_file = prefix_snapshot + "_iter_" + str((num_epoch) * k) + ".caffemodel"

	net_deploy = caffe.Net(net_deploy_name,      # defines the structure of the model
		                weight_file, caffe.TEST)


	dices = {}
	file_dataset = open(filenames)
	print ("Beggening prediciton at epoch {}".format(num_epoch))
	for patientfile in file_dataset.read().splitlines():
		split_line = patientfile.split("\t")
		inputs_name = split_line[0]
		image_data, header = load(rep_dataset + "/" + inputs_name)
		output_name = split_line[1]
		gt_data, header = load(rep_dataset + "/" + output_name)
		nb_slices = gt_data.shape[2]

		basename = os.path.basename(inputs_name)
		patientNum = get_patient_num(basename)

		size_im = preprocessing_im(image_data[:,:,0]).shape

		mri3D = np.zeros((size_im[0], size_im[1], nb_slices))
		gt3D = np.zeros((size_im[0], size_im[1], nb_slices))
		hm3D = np.zeros((size_im[0], size_im[1], 2, nb_slices))

		for i in range(nb_slices):
			mri3D[:,:,i] = preprocessing_im(image_data[:,:,i])
			minimum = np.min(mri3D[:,:, i])
            		mri3D[:,:, i] = (mri3D[:,:,i] - minimum) / (np.max(mri3D[:,:,i]) - minimum)
			hm3D[:,:,:,i] = get_heat_map(mri3D[:,:, i], net_deploy)
			gt3D[:,:,i] = preprocessing_label(gt_data[:,:,i])


		dices[patientNum] = np.mean(compute_dice_dataset(gt3D, hm3D.argmax(2), countBgImgs=False))
		print ("\tDice patient {} : {}".format(patientNum, dices[patientNum]))

		output_file = directorySave + "/" + basename.replace(".nii", "_e{}_{}.nii".format(num_epoch, suffixe))
		createDirectoryPath(output_file)
		save(hm3D.argmax(2).astype(np.uint8), output_file,hdr=header, force=True)


	net_deploy=None

	print ("\tAt epoch {}, mean dice : {}".format(num_epoch, np.mean(dices.values())))
	dice_saveTest.write(str(num_epoch) + " : " + str(dices) + "\t" + str(dices) + "\n")


file_dataset.close()
dice_saveTest.close()
net_deploy=None


