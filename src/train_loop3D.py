
# path_caffe = "/home/luffy/Documents/Stage/"

import sys
import re
from utils_dataRV import *
import numpy as np
from PIL import Image
from shutil import copyfile
from skimage import data, segmentation, color
from skimage.future import graph
from config import *
# import caffe
import shutil

# Temporary Files
repSavePrediction = rep_dataset
nameTmpTrainFiles = rep_dataset + "/train_tmp_" +suffixe +".txt"

# Remplace the label file in a network prototxt, in order to update the ground truth for the unlabelled data file
def replace_labelFiles(prototxt, old_labelName, newLabelFileName):

	# Step 1 : read file and save the content
	train_prototxt = open(prototxt, "r")
	train_prototxt_lines = train_prototxt.readlines()

	# Step 2: rewrite it and replace the labelFileName
	train_prototxt = open(prototxt, "w")

	for line in train_prototxt_lines:
		if old_labelName in line:
			line = line.replace(old_labelName, newLabelFileName)

		train_prototxt.write(line)

	train_prototxt.close()

def getPrototxtFromSolver(solver_name):
	solver_file = open(solver_name, "r")
	lines = solver_file.readlines()

	for line in lines:
		split = line.split(":")
		if 'net' in split [0]: # if this line corresponds to the line give the network file
			solver_file.close()
			return split[1].replace("\"", "").replace(" ", "")[:-1] # we remove the "" and th \n

	solver_file.close()
	return None

# This function modify a parameter of the solver 
# Certainly a better way to do that using protobuf, I didn't thought about that at the beggening :p
def modifyPrefixSolver(solver_name, lineToModify, newValue):
	solver_file = open(solver_name, "r")
	lines = solver_file.readlines()

	newLines = []
	for line in lines:
		split = line.split(":")
		if lineToModify + ":" in line.replace(" ", ""): # if this line corresponds to the line give the network file
			split[1] = str(newValue)
			newLines.append(split[0] + " : " + split[1] + '\n')
		else:
			newLines.append(line)
	solver_file.close()

	#we rewrite the file
	solver_file = open(solver_name, "w")
	for line in newLines:
		solver_file.write(line)
	solver_file.close()


# #################### INITIALISATION ########################### #
print ("Initialisation ...")
caffe.set_device(0)
caffe.set_mode_gpu()

 # LOAD unlabelled data
num_patient_file_unlab = open(unlab_filename)
num_patient_file_test = open(test_filename)

hm3D_unlab = {}
result3D_unlab = {}
unlabelled_inputs_name = {}
unlabelled_output_name = {}

mri3D_unlab, gt3D_unlab = load_dataset(unlab_filename, rep_dataset)
for patientfile in num_patient_file_unlab.read().splitlines():
	split_line = patientfile.split("\t")
	inputs_name = split_line[0]
	output_name = split_line[1]
	basename = os.path.basename(inputs_name)
	patientNum = re.search('P(.+?).nii', basename).group(1)
	unlabelled_inputs_name[patientNum] = inputs_name
	unlabelled_output_name[patientNum] = output_name
	hm3D_unlab[patientNum] = np.zeros((mri3D_unlab[patientNum].shape[0], mri3D_unlab[patientNum].shape[1], 2, mri3D_unlab[patientNum].shape[2]))
	result3D_unlab[patientNum] = np.zeros(mri3D_unlab[patientNum].shape)
num_patient_file_unlab.close()

hm3D_test = {}
result3D_test = {}
test_inputs_name = {}
test_output_name = {}
mri3D_test, gt3D_test = load_dataset(test_filename, rep_dataset)
for patientfile in num_patient_file_test.read().splitlines():
	split_line = patientfile.split("\t")
	inputs_name = split_line[0]
	output_name = split_line[1]
	basename = os.path.basename(inputs_name)
	patientNum = re.search('P(.+?).nii', basename).group(1)
	test_inputs_name[patientNum] = os.path.basename(inputs_name)
	test_output_name[patientNum] = os.path.basename(output_name)
	hm3D_test[patientNum] = np.zeros((mri3D_test[patientNum].shape[0], mri3D_test[patientNum].shape[1], 2, mri3D_test[patientNum].shape[2]))
	result3D_test[patientNum] = np.zeros(mri3D_test[patientNum].shape)
num_patient_file_test.close()


if epoch_start != 0:
	weight_file = prefix_snapshot + "_iter_" + str((epoch_start-1) * k) + ".solverstate"
else:
	copyfile(original_train_filenames, nameTmpTrainFiles)

createDirectoryPath(dice_saveUnlab_filename)
dice_saveUnlab_file = open(dice_saveUnlab_filename, "a")
dice_saveTest_file = open(dice_saveTest_filename, "a")

# Copy the original file in a tmp file
# network_prototxt = getPrototxtFromSolver(solver_name)
network_prototxt = orginal_net.replace(".prototxt", "_loop.prototxt")
#modify the snapshot prefix to fit the parameters
createDirectoryPath(prefix_snapshot)
modifyPrefixSolver(solver_name, "snapshot_prefix", prefix_snapshot)
modifyPrefixSolver(solver_name, "snapshot", k)
modifyPrefixSolver(solver_name, "net", network_prototxt)
copyfile(orginal_net, )
replace_labelFiles (network_prototxt, original_train_filenames, nameTmpTrainFiles)
print ("end of initialisation")


# #################### TRAINING ########################### # 
print ("training...")
for num_epoch in range(epoch_start, number_epoch):

	if num_epoch == 0:
		new_weight_file = weight_file
	else:
		new_weight_file = prefix_snapshot + "_iter_" + str((num_epoch) * k) + ".caffemodel"
	solver = None # free memory

	# #################### PREDICTION ON UNLABALLED  data ########################### # 
	# Use graph cut or trust region to improve the segmentation. save the results
	print ("Prediction for unlabelled dataset and test set...")
	net_deploy = caffe.Net(net_deploy_name,      # defines the structure of the model
	                new_weight_file, caffe.TEST)

	copyfile(original_train_filenames, nameTmpTrainFiles)
	train_tmp_file = open(nameTmpTrainFiles, "a")

	dice_patients_W_unlab = {}
	dice_patients_Wo_unlab = {}
	dice_patients_W_test = {}
	dice_patients_Wo_test = {}

	for patientNum in mri3D_unlab.keys():
		for i in range(mri3D_unlab[patientNum].shape[2]):
			hm3D_unlab[patientNum][:,:,:, i] = get_heat_map(mri3D_unlab[patientNum][:,:, i], net_deploy)
			if postProc == "graphcut2D":
				result3D_unlab[patientNum][:,:,i] = graphcut3D(mri3D_unlab[patientNum][:,:,i,np.newaxis], hm3D_unlab[patientNum][:,:,:,i,np.newaxis], lamda=lamda, sigma=sigma, pWeightMax=pWeightMax)[:,:,0]

		predictions = hm3D_unlab[patientNum].argmax(2)
		
		if postProc == "graphcut3D":
			result3D_unlab[patientNum] = graphcut3D(mri3D_unlab[patientNum], hm3D_unlab[patientNum], lamda=lamda, sigma=sigma, pWeightMax=pWeightMax)
		elif postProc=="noNorm":
			result3D_unlab[patientNum] = predictions.copy()

		dice_patients_W_unlab[patientNum] = np.mean(compute_dice_dataset(gt3D_unlab[patientNum], result3D_unlab[patientNum], countBgImgs=True))
		dice_patients_Wo_unlab[patientNum] = np.mean(compute_dice_dataset(gt3D_unlab[patientNum], predictions, countBgImgs=True))
		
		print dice_patients_Wo_unlab

		# Save the result to a temporary file and reinject it as input of the CNN
		nameFileInput = unlabelled_inputs_name[patientNum]
		output_name = unlabelled_output_name[patientNum]
		nameFileOutput = "./prediction_tmp/" + suffixe +  "/" +  output_name + ".npy"
		nameFileWeight = "./prediction_tmp/" + suffixe + "/" +  output_name + "_weight.npy"

		createDirectoryPath(repSavePrediction + nameFileOutput)
		np.save(repSavePrediction + nameFileOutput, result3D_unlab[patientNum])

		size_image = (result3D_unlab[patientNum][:,:,i].shape)
		weight = alpha_fg * result3D_unlab[patientNum]+ alpha_bg * (1-result3D_unlab[patientNum])

		np.save(repSavePrediction + nameFileWeight, weight)
		train_tmp_file.write(nameFileInput + "\t" + nameFileOutput + "\t" + nameFileWeight +"\n")

	# Prediction for test set
	for patientNum in mri3D_test.keys(): 
		for i in range(mri3D_test[patientNum].shape[2]):
			hm3D_test[patientNum][:,:,:, i] = get_heat_map(mri3D_test[patientNum][:,:, i], net_deploy)
			if postProc == "graphcut2D":
				result3D_test[patientNum][:,:,i] = graphcut3D(mri3D_test[patientNum][:,:,i,np.newaxis], hm3D_test[patientNum][:,:,:,i,np.newaxis], lamda=lamda, sigma=sigma, pWeightMax=pWeightMax)[:,:,0]	
		predictions_test = hm3D_test[patientNum].argmax(2)
		if postProc == "graphcut3D":
			result3D_test[patientNum] = graphcut3D(mri3D_test[patientNum], hm3D_test[patientNum], lamda=lamda, sigma=sigma, pWeightMax=pWeightMax)
		elif postProc == "norNorm":
			result3D_test[patientNum] = predictions_test.copy()

		dice_patients_W_test[patientNum] = np.mean(compute_dice_dataset(gt3D_test[patientNum], result3D_test[patientNum], countBgImgs=True))
		dice_patients_Wo_test[patientNum] = np.mean(compute_dice_dataset(gt3D_test[patientNum], predictions_test, countBgImgs=True))

	print ("End of prediction ")
	net_deploy = None
	train_tmp_file.close()

	print("Saving dice for the unlabel images in",dice_saveUnlab_filename,".... ")
	dice_saveUnlab_file.write(str(num_epoch) + " : " + str(dice_patients_W_unlab) + "\t" + str(dice_patients_Wo_unlab) + "\n")
	dice_saveUnlab_file.close()
	dice_saveUnlab_file = open(dice_saveUnlab_filename, "a")

	print("Saving dice for the test images in", dice_saveTest_filename,".... ")
	dice_saveTest_file.write(str(num_epoch) + " : " + str(dice_patients_W_test) + "\t" + str(dice_patients_Wo_test) + "\n")
	dice_saveTest_file.close()
	dice_saveTest_file = open(dice_saveTest_filename, "a")

	solver = caffe.SGDSolver(solver_name)
	if num_epoch == 0:
		solver.net.copy_from(weight_file)
	else:
		solver.restore(new_weight_file.replace("caffemodel", 'solverstate'))
		
	if num_epoch != (number_epoch - 1):
		print ("Beggening of ", str(k), " itterations" )
		solver.step(k)


	if increment == 1: 
		alpha_fg *= 1.1
		alpha_bg *= 1.1


# We remove temporary file
os.remove(nameTmpTrainFiles)
shutil.rmtree(repSavePrediction + "./prediction_tmp/", ignore_errors=True)

print ("end of training")

