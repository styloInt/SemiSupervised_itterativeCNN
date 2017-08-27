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

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = np.random.choice(self.indices)
            split_line = self.idx.split("\t")
            self.mri3D , _ = load(self.rep + "/" + split_line[0])
            self.indSlice = 0


            if ".npy" in split_line[1]:
                self.gt_data = np.load('{}/{}'.format(self.rep, split_line[1]))
            else:
                self.gt_data, _ = load(self.rep + "/" + split_line[1])

            self.size_im = self.mri3D.shape
            weight_name = None
            if len(split_line) > 2: # weight file specified
                weight_name = split_line[2]

            if weight_name == None:
                self.weights = np.ones((self.size_im[0], self.size_im[1], self.size_im[2]))
                self.dataAugmenation = True
            else:
                self.weights = np.load('{}/{}'.format(self.rep, weight_name))
                self.dataAugmenation = False


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.label = self.load_label(self.idx)
        self.weight = self.load_weight(self.idx)

        if self.elasticTransform and self.dataAugmenation:
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
            self.indSlice += 1
            if self.indSlice >= self.mri3D.shape[2]:
                self.idx = np.random.choice(self.indices)
                split_line = self.idx.split("\t")
                self.mri3D , _ = load(self.rep + "/" + split_line[0])

                if ".npy" in split_line[1]:
                    self.gt_data = np.load('{}/{}'.format(self.rep, split_line[1]))
                else:
                    self.gt_data, _ = load(self.rep + "/" + split_line[1])

                weight_name = None
                if len(split_line) > 2: # weight file specified
                    weight_name = split_line[2]

                self.size_im = self.mri3D.shape
                if weight_name == None:
                    self.weights = np.ones((self.size_im[0], self.size_im[1], self.size_im[2]))
                    self.dataAugmenation = True
                else:
                    self.weights = np.load('{}/{}'.format(self.rep, weight_name))
                    self.dataAugmenation = False

                self.indSlice = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image
        """
        # Read all the the slices in the repertory


        im = preprocessing_im(self.mri3D[:,:,self.indSlice])
        minimum = np.min(im)
        im = (im - minimum) / (np.max(im)-minimum) * 1

        return im


    def load_label(self, idx):
        """
        Load label image 
        """

        return preprocessing_label(self.gt_data[:,:,self.indSlice])

    def load_weight(self, idx):

        return self.weights[:,:,self.indSlice]
