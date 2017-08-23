import sys
from config_train import *

import numpy as np
import os
import re
from utils_dataRV import *

def getValueFromSolver(solver_name, param):
	solver_file = open(solver_name, "r")
	lines = solver_file.readlines()

	for line in lines:
		split = line.split(":")
		if param in split [0]: # if this line corresponds to the line give the network file
			solver_file.close()
			return split[1].replace("\"", "").replace(" ", "")[:-1] # we remove the "" and th \n

	solver_file.close()
	return None


nbItterationDone = 0
nbItterTest = int(getValueFromSolver(solver_name, "test_interval"))
snapshotPrefix = getValueFromSolver(solver_name, "snapshot_prefix")
createDirectoryPath(snapshotPrefix)

mri3D, gt3D = load_dataset(filenames_validation, rep_dataset)

solver = caffe.SGDSolver(solver_name)
if usePretrainedModel == 1:
	solver.restore(pretrainedModel)
	nbItterationDone = int(re.search('iter_([0-9]+).solverstate', pretrainedModel).group(1))
if usePretrainedModel == 2:
	solver.net.copy_from(pretrainedModel)


print ("Training for {} itterations...".format(nb_itteration))
print ("Dice will be computed on the validation set every {} itterations...".format(nbItterTest))
print ("Starting at itteration {}".format(nbItterationDone))
for i in range(int(nb_itteration/nbItterTest)):
	start_time = time.time()
	solver.step(nbItterTest)
	print ("\t {} itterations done in {} seconde".format(nbItterTest, (time.time() - start_time)))
	nbItterationDone += nbItterTest

	weight_file = snapshotPrefix + "_iter_" + str(nbItterationDone) + ".caffemodel"
	solver = None

	net_deploy = caffe.Net(net_deploy_name,      # defines the structure of the model
	                weight_file, caffe.TEST)

	hm3D = {}
	dice = {}
	for patientNum in mri3D.keys():
		hm3D[patientNum] = np.zeros((mri3D[patientNum].shape[0], mri3D[patientNum].shape[1], nb_classes, mri3D[patientNum].shape[2]))
		for i in range(mri3D[patientNum].shape[2]):
			hm3D[patientNum][:,:,:,i] = get_heat_map(mri3D[patientNum][:,:, i], net_deploy)

		dice[patientNum] = np.mean(compute_dice_dataset(gt3D[patientNum], hm3D[patientNum].argmax(2), countBgImgs=True))
	net_deploy = None # free the memory

	solver = caffe.SGDSolver(solver_name)
	solver.restore(weight_file.replace("caffemodel", "solverstate"))
	print ("After {} itterations, Dice on validation set : {}".format(nbItterationDone, np.mean(dice.values())))






