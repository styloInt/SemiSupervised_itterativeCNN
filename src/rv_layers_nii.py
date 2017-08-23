import caffe
import numpy as np
from PIL import Image
import scipy.io
import random
import cv2
import os
import sys
from skimage import img_as_uint
from medpy.io import load, header
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from utils_dataRV import *

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def data_augmentation(im, label):
    rotatation_angle = [-20, -10, 0, 10, 20]
    translate_x = [-15, -10, 0, 10, 15]
    translate_y = [-15, -10, 0, 10, 15]

    angle = random.choice(rotatation_angle)
    tx = random.choice(translate_x)
    ty = random.choice(translate_y)

    rows, cols = im.shape
    M_rotate = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    
    M_translate = np.float32([[1,0,tx],[0,1,ty]])
    im = cv2.warpAffine(im, M_translate,(cols,rows))
    label = cv2.warpAffine(label,M_translate,(cols,rows))
    
    im = cv2.warpAffine(im,M_rotate,(cols,rows))
    label = cv2.warpAffine(label, M_rotate,(cols,rows))

    return im, label




class RV_layer(caffe.Layer):
    """
    """
    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - rv_dir: path to RV dataset
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for RV semantic segmentation.

        """
        # config
        params = eval(self.param_str)
        self.listFiles = params['listFiles']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.rep = params['rep']
        self.elasticTransform = params.get('elasticTransform', False)

        # two tops: data and label
        if len(top) < 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        listFiles  = self.listFiles

        self.indices = open(listFiles, 'r').read().splitlines()
        self.idx = 0

         # LOAD the data. Here it's possible to do that because we have a small dataset which fit in the ram memory (I had 32 Gb ram availaible in my computer). 
         # If it doesnt fit , it's better to use LMDB file or load an image at each itteration

        self.mri3D = {}
        self.gt3D = {}
        self.weights = {}
        self.dataAugmenation = {}
        for ind, patientfile in enumerate(self.indices):
            split_line = patientfile.split("\t")
            inputs_name = split_line[0]
            output_name = split_line[1]
            weight_name = None
            if len(split_line) > 2: # weight file specified
                weight_name = split_line[2]

            image_data, _ = load(self.rep + "/" + inputs_name)
            if ".npy" in output_name:
                gt_data = np.load('{}/{}'.format(self.rep, output_name))
                gt_data = gt_data[:image_data.shape[0], :image_data.shape[1], :]
            else:
                gt_data, _ = load(self.rep + "/" + output_name)

            size_im = preprocessing_im(image_data[:,:,0]).shape

            self.mri3D[ind] = np.zeros((size_im[0], size_im[1], gt_data.shape[2]))
            self.gt3D[ind] = np.zeros((size_im[0], size_im[1], gt_data.shape[2]))

            if weight_name == None:
                self.weights[ind] = np.ones((size_im[0], size_im[1], gt_data.shape[2]))
                self.dataAugmenation[ind] = True
            else:
                self.weights[ind] = np.load('{}/{}'.format(self.rep, weight_name))
                self.dataAugmenation[ind] = False

            for i in range(gt_data.shape[2]):
                self.mri3D[ind][:,:,i] = preprocessing_im(image_data[:,:,i])
                self.mri3D[ind][:,:, i] -= np.min(self.mri3D[ind][:,:, i])
                self.mri3D[ind][:,:, i] = self.mri3D[ind][:,:,i] / np.max(self.mri3D[ind][:,:,i]) * 1
                self.gt3D[ind][:,:, i] = preprocessing_label(gt_data[:,:,i])

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = np.random.choice(self.mri3D.keys())
            self.indSlice =  np.random.randint(self.mri3D[self.idx].shape[2])


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.idx, self.indSlice)
        self.label = self.load_label(self.idx, self.indSlice)
        self.weight = self.weights[self.idx][:,:,self.indSlice]

        if self.elasticTransform and self.dataAugmenation[self.idx]:
            # im_merge = np.concatenate((self.data[...,None], self.label[...,None]), axis=2)
            # # Apply transformation on image
            # im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08, random_state=None)
            # # Split image and mask
            # self.data = im_merge_t[...,0]
            # self.label = im_merge_t[...,1]
            self.data, self.label = data_augmentation(self.data, self.label)

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1,1,*self.data.shape)
        top[1].reshape(1,1,*self.label.shape)
        top[2].reshape(1, 1, *self.weight.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.weight

        # pick next input
        if self.random:
            self.idx = np.random.choice(self.mri3D.keys())
            self.indSlice =  np.random.randint(self.mri3D[self.idx].shape[2])

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx, indSlice):
        """
        Load input image
        """
        # Read all the the slices in the repertory
        return self.mri3D[idx][:,:,indSlice]


    def load_label(self, idx, indSlice):
        """
        Load label image 
        """
        return self.gt3D[idx][:,:,indSlice]
