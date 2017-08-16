import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage
# import cv2
import time
import scipy.misc
import caffe
import seaborn as sns
from IPython import display
import time 
from PIL import Image
import random
import cv2
import scipy.io
import skimage
import maxflow
import bisect
import medpy
import re
from medpy.io import load, header
from medpy.graphcut.generate import graph_from_labels, graph_from_voxels
from medpy.graphcut.energy_voxel import boundary_difference_exponential, boundary_maximum_exponential
from medpy.graphcut.energy_voxel import regional_probability_map



def regional_weight_map(graph, (background_weight, foreground_weight, alpha)):
    foreground_weight = scipy.asarray(foreground_weight)
    background_weight = scipy.asarray(background_weight)
    probabilities = np.vstack([(foreground_weight * alpha).flat,
                                  ((background_weight) * alpha).flat]).T
    graph.set_tweights_all(probabilities)


def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

def graphcut3D(im, hm, lamda, sigma, eps=1e-10, pWeightMax=None):
    pf = np.log(hm[:,:,1] + eps) - np.log(hm[:,:,0]+eps)
    pb = np.log(hm[:,:,0] + eps) - np.log(hm[:,:,1]+eps)
    # pb = np.zeros(im.shape)

    # Maximum weight for the pixels predicted foreground by the CNN
    pixelsFG_weight_max = hm.argmax(axis=2)
    pixelsBG_weight_max = np.zeros(im.shape)

    if pWeightMax:
        pixelsFG_weight_max[hm[:,:,1,:] < pWeightMax] = 0

    # for i in range(pixelsFG_weight_max.shape[2]):
    #     label_bin = pixelsFG_weight_max[:,:,i].copy()
    #     label_rp = skimage.measure.label(label_bin)
    #     rp = skimage.measure.regionprops(label_rp)

    #     rp = max(rp, key=lambda r: r.area)

    #     pixelsFG_weight_max[:,:,i] = np.zeros((im.shape[0], im.shape[1]))
    #     for coo in rp.coords:
    #         pixelsFG_weight_max[coo[0], coo[1],i] = 1



    pixelsFG_weight_max[pixelsBG_weight_max == 1] = 0
    im_boundary_term = im
    gcgraph = graph_from_voxels(    pixelsFG_weight_max,
                                    pixelsBG_weight_max,
                                    #regional_term = regional_weight_map,
                                    regional_term = regional_probability_map,
                                    regional_term_args = (hm[:,:,1,:], lamda),
                                    #regional_term_args = (pb,pf,alpha1),
                                    boundary_term = boundary_difference_exponential,
                                    boundary_term_args = (im_boundary_term, sigma, False))

    maxflow = gcgraph.maxflow()

    result_image_data = np.zeros(im.size, dtype=np.bool)
    for idx in range(len(result_image_data)):
        result_image_data[idx] = 0 if gcgraph.termtype.SINK == gcgraph.what_segment(idx) else 1

    result_image_data = result_image_data.reshape(im.shape)

    # We only keep the biggest connexe composante
    for i in range(result_image_data.shape[2]):
        label_bin = result_image_data[:,:,i].copy()
        label_rp = skimage.measure.label(label_bin)
        rp = skimage.measure.regionprops(label_rp)

        if rp == []: 
            result_image_data[:,:,i] = np.zeros((im.shape[:2]))
            continue

        rp = max(rp, key=lambda r: r.area)

        result_image_data[:,:,i] = np.zeros((im.shape[0], im.shape[1]))
        for coo in rp.coords:
            result_image_data[coo[0], coo[1],i] = 1
    return result_image_data

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def normalize_heatmap(heatmap):
    heat_map_normalize = np.zeros(heatmap.shape)
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heat_map_normalize[x,y,:] = softmax(heatmap[x,y,:])
    
    return heat_map_normalize

def do_training(solver, step_size, nb_step=0):
        solver.step(step_size)

        heat_map = solver.test_nets[0].blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
        heat_map_normalize = normalize_heatmap(heat_map)
#         heat_map_normalize = heat_map
        minimum = np.min(heat_map[:,:,0])

        nb_subplot = 4
        plt.figure(figsize=(10,10))
        image_test = solver.test_nets[0].blobs["data"].data[0,0,:,:]
        image_test_label = solver.test_nets[0].blobs["label"].data[0,0,:,:]
        plt.subplot(1,nb_subplot,1)
        plt.imshow(image_test)
        plt.title("image test")
        plt.subplot(1,nb_subplot,2)
        plt.imshow(image_test_label)
        plt.title("Label of the test image")
        plt.subplot(1,nb_subplot,3)
        plt.imshow(np.append(heat_map_normalize, np.zeros((heat_map_normalize.shape[0], heat_map_normalize.shape[1],1)), 2))
        plt.title("Heat map")
        # plt.subplot(1,nb_subplot,4)
        # plt.imshow(np.append(heat_map_normalize, np.zeros(heat_map_normalize.shape[0], heat_map_normalize.shape[1],1), 3))
        # plt.title("score")
        plt.subplot(1,nb_subplot,4)
        plt.imshow(solver.test_nets[0].blobs["score-final"].data[0,:,:,:].transpose(1,2,0).argmax(2), vmin=0, vmax=1)
        plt.title("After : " + str(nb_step+step_size) + " itterations")
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(1)
    
###
    # save_image : place where to save the image
###
def save_image(img, vmin=None, vmax=None, title='', save_image=None, save_asMat=False):
    # plt.imshow(img, vmin=vmin, vmax=vmax)
    # plt.axis('off')
    # plt.title(title)

    if not save_image is None: #if not nans
        path = os.path.dirname(save_image)
        if not os.path.exists(path):
            os.makedirs(path)

    scipy.misc.toimage(img, cmin=vmin, cmax=vmax).save(save_image)

    if save_asMat:
        scipy.io.savemat(save_image[:-4] + ".mat", mdict={os.path.basename(save_image)[:-4] : img})


def dice_metric(seg, gt):
    if (np.sum(seg) + np.sum(gt)) == 0:
        return 1

    dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice

def preprocessing_im(im):
    if len(im.shape) > 2:
        im = im[:,:,0]
        im = im.astype(np.float)
        im = im

    # We want the shape to be divisble by 8 for the network(because of pooling layer - upsampling), we add zero padding
    if im.shape[0] % 8 != 0:
        im = np.append(im, np.zeros((8 - im.shape[0] % 8,im.shape[1])), 0)

    if im.shape[1] % 8 != 0:
        im = np.append(im, np.zeros((im.shape[0], 8 - im.shape[1] % 8)), 1)

    return im

def preprocessing_label(label):
    if (len(label.shape) > 2):
        label = label[:,:,0]
    label[label > 0] = 1

    # We want the shape to be divisble by 8 for the network(because of pooling layer - upsampling), we add zero padding
    if label.shape[0] % 8 != 0:
        label = np.append(label, np.zeros((8 - label.shape[0] % 8,label.shape[1])), 0)

    if label.shape[1] % 8 != 0:
        label = np.append(label, np.zeros((label.shape[0], 8 - label.shape[1] % 8)), 1)

    return label



def compute_dice_dataset_net(dataset, gts, net_deploy):
    dices = []
    for num_image in range(dataset.shape[2]):
        img = dataset[:,:,num_image]
        img = preprocessing_im(img)

        label = gts[:,:, num_image]
        label = preprocessing_label(label)

        net_deploy.blobs['data'].data[...] = img
        net_deploy.forward()
        out = net_deploy.blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
        heat_map_normalize = normalize_heatmap(out)
        label_out = heat_map_normalize.argmax(2)
        dices.append(dice_metric(label_out, label))

    return dices

def compute_dice_dataset(gts, prediction, countBgImgs=True):
    dices = []
    for num_image in range(gts.shape[2]):
        if not countBgImgs and np.sum(gts[:,:,num_image]) == 0:
            continue
        dices.append(dice_metric(gt=gts[:,:,num_image], seg=prediction[:,:,num_image]))
    return dices


def get_heat_map(img, net_deploy, normalize=True):
    img = preprocessing_im(img)
    net_deploy.blobs['data'].reshape(1, 1, img.shape[0], img.shape[1])
    net_deploy.blobs['data'].data[...] = img
    net_deploy.forward()
    out = net_deploy.blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
    if normalize:
        return normalize_heatmap(out)
    return out

def get_prediction(img, net_deploy):
    heatmap = get_heat_map(img, net_deploy)
    return heatmap.argmax(axis=2)

def get_predictions(imgs, net_deploy):
    predictions = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    for num_image in range(imgs.shape[2]):
        predictions[:,:,num_image] = get_prediction(imgs[:,:,num_image], net_deploy)

    return predictions


def load_dataset(dataset_filename, rep_dataset, readGT=True):
    dataset_filename = open(dataset_filename)

    mri3D = {}
    gt3D = {}
    for patientfile in dataset_filename.read().splitlines():

        split_line = patientfile.split("\t")
        inputs_name = split_line[0]
        image_data, _ = load(rep_dataset + "/" + inputs_name)
        nb_slices = image_data.shape[2]
        if readGT:
            output_name = split_line[1]
            gt_data, _ = load(rep_dataset + "/" + output_name)
            nb_slices = gt_data.shape[2]

        basename = os.path.basename(inputs_name)
        patientNum = re.search('P(.+?).nii', basename).group(1)

        size_im = preprocessing_im(image_data[:,:,0]).shape

        mri3D[patientNum] = np.zeros((size_im[0], size_im[1], nb_slices))
        gt3D[patientNum] = np.zeros((size_im[0], size_im[1], nb_slices))

        for i in range(nb_slices):
            mri3D[patientNum][:,:,i] = preprocessing_im(image_data[:,:,i])
            mri3D[patientNum][:,:, i] -= np.min(mri3D[patientNum][:,:, i])
            mri3D[patientNum][:,:, i] = mri3D[patientNum][:,:,i] / np.max(mri3D[patientNum][:,:,i])
            if readGT:
                gt3D[patientNum][:,:, i] = preprocessing_label(gt_data[:,:,i])

    dataset_filename.close()

    if readGT:
        return mri3D, gt3D
    else:
        return mri3D

    

"""
    nbImageToSave : if the dataset is too big, you can choose one 
"""
def save_results(dataset,labels, net_deploy, rep_save_results = None, nbImageToSave = None, nameFiles=None):
    indices = list(range(dataset.shape[-1]))
    images_to_test = indices[:nbImageToSave]

    nbImageToDisplay = 4
    has_gt = True
    if labels is None :
        labels = np.zeros(dataset.shape)
        nbImageToDisplay = 3
        has_gt = False


    for num_image in images_to_test:
        img = dataset[:,:,num_image]
        img = preprocessing_im(img)

        label = labels[:,:, num_image]
        label = preprocessing_label(label)


        net_deploy.blobs['data'].data[...] = img
        net_deploy.forward()
        out = net_deploy.blobs["score-final"].data[0,:,:,:].transpose(1,2,0)
        heat_map_normalize = normalize_heatmap(out)
        label_out = heat_map_normalize.argmax(2)

        if rep_save_results is None:
            name_save_image = [None] * 4
        elif not nameFiles is None:
            name_save_image = [rep_save_results + nameFiles[num_image] + "_ori.jpg", rep_save_results + nameFiles[num_image] + "_gt.jpg", \
            rep_save_results + nameFiles[num_image] + "_hm.jpg", rep_save_results + nameFiles[num_image] + "_predict.jpg"]
        else:
            name_save_image = [rep_save_results + str(num_image) + "_ori.jpg", rep_save_results + str(num_image) + "_gt.jpg", \
            rep_save_results + str(num_image) + "_hm.jpg", rep_save_results + str(num_image) + "_predict.jpg"]

        index_plot = 1

        save_image(img, title="Orginal image", save_image=name_save_image[0])
        if has_gt:
            save_image(label, vmin=0, vmax=1, title="Ground truth segmentation", save_image=name_save_image[1])
        save_image(np.append(heat_map_normalize, np.zeros((heat_map_normalize.shape[0], heat_map_normalize.shape[1],1)), 2), vmin=0, vmax=1, title="Heat map", save_image=name_save_image[2])
        save_image(label_out, vmin=0, vmax=1, title="Segmentation predicted", save_image=name_save_image[3])


def createDirectoryPath(file_name):
    path = os.path.dirname(file_name)
    if not os.path.exists(path):
        os.makedirs(path)

