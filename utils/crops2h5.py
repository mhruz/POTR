"""Provides importer classes for importing data from different datasets.
DepthImporter provides interface for loading the data from a dataset, esp depth images.

ICVLImporter, NYUImporter, HANDSImporter are
Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>
This files are part of DeepPrior++.
DeepPrior++ is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""


import os
import io
import time
import copy
from os.path import isfile
import sys

from PIL import Image
from multiprocessing import Pool
import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mayavi.mlab as mlab

import cv2
import json
import glob
import h5py

from handdetector import HandDetector


##### visualisation #####
def visualisation(V, color=(0, 1, 0), scale=0.75, show=True):
    yy, xx, zz = np.where(V == True)
    mlab.points3d(xx, yy, zz,
                         mode="cube",
                         color=color,
                         scale_factor=scale)
    if show:
        mlab.show()

def plotlabels(labels, gray=False, gray2=False, datasource='hands2017'):
    if gray:
        colors = [(0.6, 0.6, 0.6), (0.6, 0.6, 0.6), (0.6, 0.6, 0.6), (0.6, 0.6, 0.6), (0.6, 0.6, 0.6)]
    else:
        if gray2:
            colors = [(0.4, 0.4, 0.4), (0.4, 0.4, 0.4), (0.4, 0.4, 0.4), (0.4, 0.4, 0.4), (0.4, 0.4, 0.4)]
        else:
            colors = [(1, 0, 1),(0, 0, 1),(0, 1, 0),(1, 1, 0),(1, 0, 0)]

    if datasource=='hands2017':
        mlab.plot3d(np.hstack((labels[0, 0], labels[1, 0], labels[6:9, 0])),
                    np.hstack((labels[0, 1], labels[1, 1], labels[6:9, 1])),
                    np.hstack((labels[0, 2], labels[1, 2], labels[6:9, 2])), color=colors[0], representation="wireframe",
                    line_width=5.0)
        mlab.plot3d(np.hstack((labels[0, 0], labels[2, 0], labels[9:12, 0])),
                    np.hstack((labels[0, 1], labels[2, 1], labels[9:12, 1])),
                    np.hstack((labels[0, 2], labels[2, 2], labels[9:12, 2])), color=colors[1],
                    representation="wireframe", line_width=5.0)
        mlab.plot3d(np.hstack((labels[0, 0], labels[3, 0], labels[12:15, 0])),
                    np.hstack((labels[0, 1], labels[3, 1], labels[12:15, 1])),
                    np.hstack((labels[0, 2], labels[3, 2], labels[12:15, 2])), color=colors[2],
                    representation="wireframe", line_width=5.0)
        mlab.plot3d(np.hstack((labels[0, 0], labels[4, 0], labels[15:18, 0])),
                    np.hstack((labels[0, 1], labels[4, 1], labels[15:18, 1])),
                    np.hstack((labels[0, 2], labels[4, 2], labels[15:18, 2])), color=colors[3],
                    representation="wireframe", line_width=5.0)
        mlab.plot3d(np.hstack((labels[0, 0], labels[5, 0], labels[18:21, 0])),
                    np.hstack((labels[0, 1], labels[5, 1], labels[18:21, 1])),
                    np.hstack((labels[0, 2], labels[5, 2], labels[18:21, 2])), color=colors[4],
                    representation="wireframe", line_width=5.0)

    if datasource=='nyu':
        mlab.plot3d(np.hstack((labels[13, 1], labels[1::-1, 1])),
                    np.hstack((labels[13, 0], labels[1::-1, 0])),
                    np.hstack((labels[13, 2], labels[1::-1, 2])), color=colors[0], representation="wireframe",
                    line_width=5.0)
        mlab.plot3d(np.hstack((labels[13, 1], labels[3:1:-1, 1])),
                    np.hstack((labels[13, 0], labels[3:1:-1, 0])),
                    np.hstack((labels[13, 2], labels[3:1:-1, 2])), color=colors[1],
                    representation="wireframe", line_width=5.0)
        mlab.plot3d(np.hstack((labels[13, 1], labels[5:3:-1, 1])),
                    np.hstack((labels[13, 0], labels[5:3:-1, 0])),
                    np.hstack((labels[13, 2], labels[5:3:-1, 2])), color=colors[2],
                    representation="wireframe", line_width=5.0)
        mlab.plot3d(np.hstack((labels[13, 1], labels[7:5:-1, 1])),
                    np.hstack((labels[13, 0], labels[7:5:-1, 0])),
                    np.hstack((labels[13, 2], labels[7:5:-1, 2])), color=colors[3],
                    representation="wireframe", line_width=5.0)
        mlab.plot3d(np.hstack((labels[13, 1], labels[10:7:-1, 1])),
                    np.hstack((labels[13, 0], labels[10:7:-1, 0])),
                    np.hstack((labels[13, 2], labels[10:7:-1, 2])), color=colors[4],
                    representation="wireframe", line_width=5.0)
        mlab.plot3d(np.hstack((labels[13, 1], labels[11, 1])),
                    np.hstack((labels[13, 0], labels[11, 0])),
                    np.hstack((labels[13, 2], labels[11, 2])), color=colors[3],
                    representation="wireframe", line_width=5.0)
        mlab.plot3d(np.hstack((labels[13, 1], labels[12, 1])),
                    np.hstack((labels[13, 0], labels[12, 0])),
                    np.hstack((labels[13, 2], labels[12, 2])), color=colors[4],
                    representation="wireframe", line_width=5.0)

def plotlabels_plt(ax, labels, gray=False, gray2=False, datasource='hands2017'):

    if datasource=='hands2017':
        ax.plot3D(np.hstack((labels[0, 0], labels[1, 0], labels[6:9, 0])),
                    np.hstack((labels[0, 1], labels[1, 1], labels[6:9, 1])),
                    np.hstack((labels[0, 2], labels[1, 2], labels[6:9, 2])))
        ax.plot3D(np.hstack((labels[0, 0], labels[2, 0], labels[9:12, 0])),
                    np.hstack((labels[0, 1], labels[2, 1], labels[9:12, 1])),
                    np.hstack((labels[0, 2], labels[2, 2], labels[9:12, 2])))
        ax.plot3D(np.hstack((labels[0, 0], labels[3, 0], labels[12:15, 0])),
                    np.hstack((labels[0, 1], labels[3, 1], labels[12:15, 1])),
                    np.hstack((labels[0, 2], labels[3, 2], labels[12:15, 2])))
        ax.plot3D(np.hstack((labels[0, 0], labels[4, 0], labels[15:18, 0])),
                    np.hstack((labels[0, 1], labels[4, 1], labels[15:18, 1])),
                    np.hstack((labels[0, 2], labels[4, 2], labels[15:18, 2])))
        ax.plot3D(np.hstack((labels[0, 0], labels[5, 0], labels[18:21, 0])),
                    np.hstack((labels[0, 1], labels[5, 1], labels[18:21, 1])),
                    np.hstack((labels[0, 2], labels[5, 2], labels[18:21, 2])))

    if datasource=='nyu':
        ax.plot3D(np.hstack((labels[13, 1], labels[1::-1, 1])),
                    np.hstack((labels[13, 0], labels[1::-1, 0])),
                    np.hstack((labels[13, 2], labels[1::-1, 2])))
        ax.plot3D(np.hstack((labels[13, 1], labels[3:1:-1, 1])),
                    np.hstack((labels[13, 0], labels[3:1:-1, 0])),
                    np.hstack((labels[13, 2], labels[3:1:-1, 2])))
        ax.plot3D(np.hstack((labels[13, 1], labels[5:3:-1, 1])),
                    np.hstack((labels[13, 0], labels[5:3:-1, 0])),
                    np.hstack((labels[13, 2], labels[5:3:-1, 2])))
        ax.plot3D(np.hstack((labels[13, 1], labels[7:5:-1, 1])),
                    np.hstack((labels[13, 0], labels[7:5:-1, 0])),
                    np.hstack((labels[13, 2], labels[7:5:-1, 2])))
        ax.plot3D(np.hstack((labels[13, 1], labels[10:7:-1, 1])),
                    np.hstack((labels[13, 0], labels[10:7:-1, 0])),
                    np.hstack((labels[13, 2], labels[10:7:-1, 2])))
        ax.plot3D(np.hstack((labels[13, 1], labels[11, 1])),
                    np.hstack((labels[13, 0], labels[11, 0])),
                    np.hstack((labels[13, 2], labels[11, 2])))
        ax.plot3D(np.hstack((labels[13, 1], labels[12, 1])),
                    np.hstack((labels[13, 0], labels[12, 0])),
                    np.hstack((labels[13, 2], labels[12, 2])))

def plot_skeleton(joints, style, linewidth = 2.0):
    finger = joints[[13, 10, 9, 8]]
    plt.plot(finger[:, 0], finger[:, 1], "m" + style, linewidth = linewidth)
    finger = joints[[13, 7, 6]]
    plt.plot(finger[:, 0], finger[:, 1], "b" + style, linewidth = linewidth)
    finger = joints[[13, 5, 4]]
    plt.plot(finger[:, 0], finger[:, 1], "g" + style, linewidth = linewidth)
    finger = joints[[13, 3, 2]]
    plt.plot(finger[:, 0], finger[:, 1], "y" + style, linewidth = linewidth)
    finger = joints[[13, 1, 0]]
    plt.plot(finger[:, 0], finger[:, 1], "r" + style, linewidth = linewidth)
    finger = joints[[11, 13, 12]]
    plt.plot(finger[:, 0], finger[:, 1], "k" + style, linewidth = linewidth)

def plot2Dskeleton_ICVL(joints, style='-', linewidth = 2.0):
    finger = joints[[0, 1, 2, 3]]
    plt.plot(finger[:, 0], finger[:, 1], "m" + style, linewidth = linewidth)
    finger = joints[[0, 4, 5, 6]]
    plt.plot(finger[:, 0], finger[:, 1], "b" + style, linewidth = linewidth)
    finger = joints[[0, 7, 8, 9]]
    plt.plot(finger[:, 0], finger[:, 1], "g" + style, linewidth = linewidth)
    finger = joints[[0, 10, 11, 12]]
    plt.plot(finger[:, 0], finger[:, 1], "y" + style, linewidth = linewidth)
    finger = joints[[0, 13, 14, 15]]
    plt.plot(finger[:, 0], finger[:, 1], "r" + style, linewidth = linewidth)


class DepthImporterHANDS2017(object):
    """
    provide basic functionality to load depth data
    """

    def __init__(self):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = 475.065948
        self.fy = 475.065857
        self.ux = 315.944855
        self.uy = 245.287079
        self.depth_map_size = (640, 480)
        self.crop_joint_idx = 0

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret
    
    

    def loadDepthMap(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)  # open image
        #print(img.mode)
        #exit()
        assert len(img.getbands()) == 1  # ensure depth image
        imgdata = np.asarray(img, np.float32)

        return imgdata


class DepthImporterHANDS2019(object):
    """
    provide basic functionality to load depth data
    """

    def __init__(self):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = 475.065948
        self.fy = 475.065857
        self.ux = 315.944855
        self.uy = 245.287079
        self.depth_map_size = (640, 480)
        self.crop_joint_idx = 0

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret



    def loadDepthMap(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)  # open image
        #print(img.mode)
        #exit()
        assert len(img.getbands()) == 1  # ensure depth image
        imgdata = np.asarray(img, np.float32)

        return imgdata


class DepthImporterNYU(object):
    """
    provide basic functionality to load depth data
    """

    def __init__(self):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = 588.03
        self.fy = 587.07
        self.ux = 320.
        self.uy = 240.
        self.depth_map_size = (640, 480)
        self.crop_joint_idx = 0

    #slow original
    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])

        return ret

    #quick new
    def jointsImgTo3D_quick(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        sample[:,0] = (sample[:,0] - self.ux) * sample[:,2] / self.fx
        sample[:,1] = (self.uy - sample[:,1]) * sample[:,2] / self.fy

        return sample

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret


    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3, ), np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret
    

    def loadDepthMap(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        dpt = np.bitwise_or(np.left_shift(g, 8), b)
        imgdata = np.asarray(dpt, np.float32)

        return imgdata

class DepthImporterICVL(object):
    """
    provide basic functionality to load depth data
    """

    def __init__(self):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = 241.42
        self.fy = 241.42
        self.ux = 160.0
        self.uy = 120.0
        self.depth_map_size = (320, 240)
        #self.refineNet = None
        self.crop_joint_idx = 0
        #self.hand = hand

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointsImgTo3D_quick(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        sample[:,0] = (sample[:,0] - self.ux) * sample[:,2] / self.fx
        sample[:,1] = (sample[:,1] - self.uy) * sample[:,2] / self.fy

        return sample

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret

    def loadDepthMap(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)  # open image
        assert len(img.getbands()) == 1  # ensure depth image
        imgdata = np.asarray(img, np.float32)

        return imgdata


def loadCropDepthMap(filename):
        """
        Read croped depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)  # open image
        assert len(img.getbands()) == 1  # ensure depth image
        imgdata = np.asarray(img, np.float32)

        return imgdata

def jsonReadWrite(jsonfilename, key=None, data=None):
    
    if isfile(jsonfilename):
        with open(jsonfilename) as fr:
            config = json.load(fr)
    else:
        config = {}
            
    if data is None:
        if key in config:
            return config[key]
    else:
        config[key] = data
        with open(jsonfilename, 'w') as fw:
            json.dump(config, fw, indent=4, sort_keys=True)
    
    return None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def points2range(points3D, com3D, cube=None, dsize=None):
    
    points3D_centered = points3D - com3D
    
    ind2D = points3D_centered[:,:2].copy()
    ind2D = ind2D[:,::-1] #change x,y <--> uy ux
    
    ind2D[:,0] += cube[1] / 2.
    ind2D[:,1] += cube[0] / 2.
    ind2D[:,0] /= cube[1]
    ind2D[:,1] /= cube[0]
    ind2D[:,0] *= dsize[1]
    ind2D[:,1] *= dsize[0]
    
    ind2D = np.asarray(np.rint(ind2D), dtype=np.int32) #round coordinated
    
    #filter out of range data
    idx_x = np.logical_and(ind2D[:,0] >= 0, ind2D[:,0] < dsize[1])
    idx_y = np.logical_and(ind2D[:,1] >= 0, ind2D[:,1] < dsize[0])
    idx = np.logical_and(idx_x, idx_y)
    ind2D = ind2D[idx]
    
    im = np.zeros(dsize, np.float32)
    val = points3D[:,2] #original depth pixel

    im[ind2D.T.tolist()] = val[idx]
    
    return im

def points2range_ver2(points3D, com3D, cube=None, dsize=None):

    points3D_centered = points3D - com3D

    ind2D = points3D_centered[:,:2].copy()
    ind2D = ind2D[:,::-1]

    ind2D[:,0] += cube[1] / 2.
    ind2D[:,1] += cube[0] / 2.
    ind2D[:,0] /= cube[1]
    ind2D[:,1] /= cube[0]
    ind2D[:,0] *= dsize[1]
    ind2D[:,1] *= dsize[0]

    ind2D = np.asarray(np.rint(ind2D), dtype=np.int32)
    idx_x = np.logical_and(ind2D[:,0] >= 0, ind2D[:,0] < dsize[1])
    idx_y = np.logical_and(ind2D[:,1] >= 0, ind2D[:,1] < dsize[0])
    idx = np.logical_and(idx_x, idx_y)
    ind2D = ind2D[idx]

    im = np.zeros(dsize, np.float32)
    val = points3D[:,2]
    val = val[idx]

    for coord, value in zip(ind2D, val):
        value_in = im[coord[0], coord[1]]
        if (value_in == 0.0) or (value < value_in):
            im[coord[0], coord[1]] = value

    return im

def saveCrops(argv, masksize=(200,200)):
    """
    Main algorithm to save croped images
    :param argv: algorithm parameters
    """

    objdir_source = argv['objdir_source'] #source pngs
    objdir_crop = argv['objdir_crop'] #out crop pngs
    objdir_json = argv['objdir_json'] #out jsons
    labelsfilelist = argv['labelsfilelist']
    bbcomfilelist = argv['bbcomfilelist']
    di = argv['di']
    docom = argv['docom']
    islabels = argv['islabels']
    islabels3D = argv['islabels3D'] #if labels are in 3D
    isbb = argv['isbb']
    iscom = argv['iscom']
    iscom3D = argv['iscom3D']
    numJoints = argv['numJoints']
    cube = argv['cube']
    dsize= argv['dsize'] #(224, 224)
    isOPENGLcoord = argv['isOPENGLcoord']
    H5labelformat = argv['H5labelformat']
    
    start = time.time()
    for i, (labelline, bbline) in enumerate(zip(labelsfilelist, bbcomfilelist)):
        
        if (i % 100) == 0:

            duration = time.time() - start
            start = time.time()
            print(i, 'duration:', duration)

            
        labelpart = labelline.split()
        bbcompart = bbline.split()
        
        #if labelpart[0] != bbpart[0]:
            #print('LABELS {} NOT CORESPOND TO BOUNDING BOX {}'.format(labelpart[0], bbpart[0]))
            #continue
        
        filename = labelpart[0]

        dptFileName = '{}/{}'.format(objdir_source, filename)

        if not os.path.isfile(dptFileName):
            print("File {} does not exist!".format(dptFileName))
            continue

        dpt = di.loadDepthMap(dptFileName)

        if islabels:
            # joints in image coordinates
            gt3Dorig = np.zeros((numJoints, 3), np.float32)
            for joint in range(numJoints):
                for xyz in range(0, 3):
                    gt3Dorig[joint, xyz] = labelpart[joint * 3 + xyz + 1]
             
            if not islabels3D:
                # normalized joints in 3D coordinates ... kvuli ICVL ktere ma labels ve 2D
                gt3Dorig = di.jointsImgTo3D(gt3Dorig)

            if isbb:
                # take COM from GT, we learn it
                com3D = np.mean(gt3Dorig, axis=0)
                ## reproject com from [x y z] to [ux uy z]
                com = di.joint3DToImg(com3D)
        
        
        #is bounding box
        if isbb:
            if docom:
                com = None
            
            # nacti bounding box
            gtBoundingBox = np.zeros((4,), np.float32)
            for point in range(4):
                gtBoundingBox[point] = bbcompart[point + 1]

            center = ((gtBoundingBox[0] + gtBoundingBox[2]) / 2., (gtBoundingBox[1] + gtBoundingBox[3]) / 2.)
        
            #if not iscom:
            # add calculate boundaries
            xstart = max(0, int(center[0] - masksize[0]/2.))
            xend  = min(dpt.shape[1], int(center[0] + masksize[0]/2.))
            ystart = max(0, int(center[1] - masksize[1]/2.))
            yend = min(dpt.shape[1], int(center[1] + masksize[1]/2.))
            
            # zeros out of boundingbox
            dpt[:, 0:xstart] = 0.0
            dpt[:, xend:] = 0.0
            dpt[0:ystart, :] = 0.0
            dpt[yend:, :] = 0.0

        if iscom: #we don't bb, but we have com
            com = np.zeros((3,), np.float32)
            try:
                for point in range(3):
                    com[point] = float(bbcompart[point])
            except ValueError:
               print("File {} does not have com!".format(dptFileName))
               continue
            
            
            if np.sum(com) == 0:
                print("File {} have zero com!".format(dptFileName))
                continue #neni vypredikovany
            
            if iscom3D:
                com = di.joint3DToImg(com)

        hd = HandDetector(dpt, di.fx, di.fy, refineNet=None, importer=None)

        try:
            dpt, M, com = hd.cropArea3D(com=com, size=cube, dsize=dsize, docom=docom)
        except UserWarning:
            print("Skipping image {}, no hand detected".format(dptFileName))
            continue

        com3D = di.jointImgTo3D(com)

        #for 3Dprojection need create new depth map
        if H5labelformat == '3Dprojection':

            #if (i % 50) == 0:
                #plt.imshow(dpt)
                #plt.show()

            ind = np.where(dpt > 0.0)

            points = np.column_stack((ind[1], ind[0], dpt[ind]))  # nyu ux,uy,z (jakoby obrazkove x y z)
            points = np.asarray(points, 'float32')

            #promitnu 2D pixely v cropu do 2D pixelu obrazku
            M_inv = invertM(M)

            points2D = transformPoints2D_quick(points, M_inv)

            points3D = di.jointsImgTo3D_quick(points2D)

            dpt = points2range_ver2(points3D, com3D, cube=cube, dsize=dsize)

            #filter holes
            median = cv2.medianBlur(dpt, 5)
            dpt[dpt==0] = median[dpt==0]

        #else for 2Dproj and Obeweger nothing to do
            
        if isOPENGLcoord: #uprava pravotociveho na levotocivy
            dpt = np.flip(dpt,axis=0)


        ###########
        # save
        file_dir = os.path.dirname(filename)
        if file_dir != '':
            json_file_dir = '{}/{}/'.format(objdir_json, file_dir)
            crop_file_dir = '{}/{}/'.format(objdir_crop, file_dir)
            ensure_dir(json_file_dir)
            ensure_dir(crop_file_dir)

        jsonfilename = '{}/{}.json'.format(objdir_json, filename)
        
        if islabels:
            gt3Dcrop = gt3Dorig - com3D  # normalize to COM

            jsonReadWrite(jsonfilename, key='gt3Dcrop', data=gt3Dcrop.tolist())
        
            
        jsonReadWrite(jsonfilename, key='filename', data=filename)
        jsonReadWrite(jsonfilename, key='com3D', data=com3D.tolist())
        jsonReadWrite(jsonfilename, key='M', data=M.tolist())
        jsonReadWrite(jsonfilename, key='cube', data=list(cube))
        
        dptFileNameCrop = '{}/{}'.format(objdir_crop, filename)
        result = Image.fromarray(dpt.astype(np.uint32), mode="I") #stejny mod jako vstupni data
        result.save(dptFileNameCrop)
        


def transformPoints2D_quick(pts, M):
    points2D = pts.copy()
    points2D[:,2] = 1.
    points2D = np.transpose(np.dot(M, np.transpose(points2D)))
    points2D[:, 0] /= points2D[:, 2]
    points2D[:, 1] /= points2D[:, 2]
    points2D[:,2] = pts[:,2]
    return points2D



def saveH5(argv):
    """
    Algorithm to make H5 from croped images
    :param argv: algorithm parameters
    """

    objdir_crop = argv['objdir_crop']
    objdir_json = argv['objdir_json']
    
    jsonfilelist = sorted(glob.glob(objdir_json +'/**/*.json', recursive=True))
    
    N = len(jsonfilelist)
    print('N=', N)
    
    h5_file_name = argv['h5file']
    h5_file = h5py.File(h5_file_name, 'w')
    h5_data_group = h5_file.create_group(argv['H5datagroup'])
    h5_file_name_group = h5_file.create_group("file_name")
    
    di = argv['di']
    
    isOPENGLcoord = argv['isOPENGLcoord']
    
    dsize= np.array(argv['dsize'], np.int32) #(224, 224) traning image resolution
    
    numJoints = argv['numJoints'] #kde budu brat json
    alllabels = np.zeros((N, numJoints, 3), np.float32)
    allgt3Dcrop = np.zeros((N, numJoints, 3), np.float32)
    allcoms = np.zeros((N, 3), np.float32)
    allcubes = np.zeros((N, 3), np.float32)
    allM = np.zeros((N, 3, 3), np.float32)
    i = 0
    for json in jsonfilelist:
        
        if (i % 100) == 0:
            print(i)
            
        filename = jsonReadWrite(json, key='filename')
      
        dptFileName = '{}/{}'.format(objdir_crop, filename)

        if not os.path.isfile(dptFileName):
            print("File {} does not exist!".format(dptFileName))
            continue

        dpt = loadCropDepthMap(dptFileName)

        com3D = np.array(jsonReadWrite(json, key='com3D'), np.float32)
        cube = np.array(jsonReadWrite(json, key='cube'), np.float32)
        gt3Dcrop = np.array(jsonReadWrite(json, key='gt3Dcrop'), np.float32)
        M = np.array(jsonReadWrite(json, key='M'), np.float32)

        # data normalization for H5
        
        imgD = np.asarray(dpt.copy(), 'float32')
        labels = gt3Dcrop
        
            
        if argv['H5labelformat'] == 'Oberweger':
            # from imgStackDepthOnly()... Oberweger
            imgD[imgD == 0] = com3D[2] + (cube[2] / 2.)
            imgD -= com3D[2]
            imgD /= (cube[2] / 2.)
            labels = gt3Dcrop / (cube[2] / 2.)
            
            if isOPENGLcoord:
                labels[:,1] *= -1
        
        
        if argv['H5labelformat'] == '2Dprojection':
            imgD[imgD == 0] = com3D[2] + (cube[2] / 2.)
            imgD -= com3D[2]
            imgD /= (cube[2] / 2.)
            
            gt3Dorig = gt3Dcrop + com3D
            gtorig = di.joints3DToImg(gt3Dorig)
            gtcrop = transformPoints2D_quick(gtorig, M)
            labels = gtcrop
            labels[:,:2] /= (dsize[1] / 2.)
            labels[:,:2] -= 1.0
            labels[:,2] -= com3D[2]
            labels[:,2] /= (cube[2] / 2.)
            
            
        if argv['H5labelformat'] == '3Dprojection':
            #Oberweger
            imgD[imgD == 0] = com3D[2] + (cube[2] / 2.)
            imgD -= com3D[2]
            imgD /= (cube[2] / 2.)

            labels = gt3Dcrop / (cube[2] / 2.)
            if isOPENGLcoord:
               labels[:,1] *= -1
               
            #instead of image data, we save voxels
            if (argv['H5datagroup'] == 'real_voxels'):
                ind = list(np.where(imgD < 0.99))
                ind.append(np.asarray(np.rint((imgD[ind] + 1.0) * ((dsize[0]/2.0) - 1.0)), dtype='int'))
                imgD = np.zeros((dsize[0], dsize[0], dsize[0]), dtype='bool')
                imgD[ind] = True
                #visualisation(imgD)
            
        
        ##############
        # write
        _file = io.BytesIO()
        np.savez_compressed(_file, imgD)
        data = _file.getvalue()
        _file.close()
        h5_data_group.create_dataset(name=str(i), shape=(1,), dtype=(np.void, len(data)))
        h5_file[argv['H5datagroup']][str(i)][:] = np.void(data)
     
        #file name
        dptFileName_as_bytes = str.encode(dptFileName)
        h5_file_name_group.create_dataset(name=str(i), shape=(1,), dtype=(np.void, len(dptFileName_as_bytes)))
        h5_file["file_name"][str(i)][:] = np.void(dptFileName_as_bytes)

        alllabels[i] = labels
        allcoms[i] = com3D
        allgt3Dcrop[i] = gt3Dcrop
        allcubes[i] = cube
        allM[i] = M
        
        i += 1

    N = i
    h5_file.create_dataset(name="com3D", shape=(N, 3), dtype=np.float32)
    h5_file['com3D'][:] = allcoms[:N]
    h5_file.create_dataset(name="cube", shape=(N, 3), dtype=np.float32)
    h5_file['cube'][:] = allcubes[:N]
    h5_file.create_dataset(name="gt3Dcrop", shape=(N, numJoints, 3), dtype=np.float32)
    h5_file['gt3Dcrop'][:] = allgt3Dcrop[:N]
    h5_file.create_dataset(name="labels", shape=(N, numJoints, 3), dtype=np.float32)
    h5_file['labels'][:] = alllabels[:N]
    h5_file.create_dataset(name="M", shape=(N, 3, 3), dtype=np.float32)
    h5_file['M'][:] = allM[:N]
    
    h5_file.close()
   

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def makeCrops(argv, chunk_size=10000, Num_processes=6):

    ensure_dir(argv['objdir_crop'])
    ensure_dir(argv['objdir_json'])
    
    N1 = 0
    N1chunk = 0
    if isfile(argv['trainlabelsfilename']):
        f = open(argv['trainlabelsfilename'])
        f.seek(0)
        trainlabelslist =  f.readlines()
        N1 = len(trainlabelslist)
        trainlabelslistchunked = chunks(trainlabelslist, chunk_size)
        N1chunk= len(trainlabelslistchunked)

    N2 = 0
    N2chunk = 0
    if isfile(argv['bbcomsfilename']):
        f = open(argv['bbcomsfilename'])
        f.seek(0)
        bbcomlist =  f.readlines()
        N2 = len(bbcomlist)
        bbcomlistchunked = chunks(bbcomlist, chunk_size)
        N2chunk = len(bbcomlistchunked)
    
    print(N1, N2)
    
    if (N1 != 0) and (N2 != 0):
        assert N1 == N2
    
    argvs = []
    Nchunk = max(N1chunk, N2chunk)
    for i in range(Nchunk):
        argvchunk = argv.copy()
        
        if N1 > 0:
            argvchunk['labelsfilelist'] = trainlabelslistchunked[i]
        else:
            argvchunk['labelsfilelist'] = []
        if N2 > 0:
            argvchunk['bbcomfilelist'] = bbcomlistchunked[i]
        else:
            argvchunk['bbcomfilelist'] = []
        
        argvs.append(argvchunk)

    try:
        pool = Pool(processes=Num_processes, maxtasksperchild=1)
        pool.map(saveCrops, argvs)

    finally:
        pool.close()
        pool.join()


def invertM(M):

    M_inv = M.copy()
    M_inv[0, 0] = 1./M_inv[0, 0]
    M_inv[1, 1] = 1./M_inv[1, 1]
    M_inv[0, 2] = - M_inv[0, 0] * M_inv[0, 2]
    M_inv[1, 2] = - M_inv[1, 1] * M_inv[1, 2]
    return M_inv


def rmse(a, b):
    return np.mean(np.sqrt(np.square(a - b).sum(axis=1)))


def showH5(argv):
    
    di = argv['di']
    dsize= argv['dsize']
    cube= argv['cube']
    
    h5_file_name = argv['h5file']
    h5_file = h5py.File(h5_file_name, 'r')

    for key in h5_file[argv['H5datagroup']].keys():
        print(i)
        pdata = h5_file[argv['H5datagroup']][str(i)][:].tostring()
        _file = io.BytesIO(pdata)
        imgD = np.load(_file)['arr_0']
        #gt3Dcrop = h5_file['gt3Dcrop'][i]
        com3D = h5_file['com3D'][i]
        labels = h5_file['labels'][i]

        #vizualization
        if argv['H5datagroup'] == 'images':
            if argv['islabels']:
                #plt.plot(112*labels[:,0]+112, 112*labels[:,1]+112, 'r')
                labels[:,:2] = 112*labels[:,:2]+112
                plot_skeleton(labels, '-o')
                #plot2Dskeleton_ICVL(labels)
            #plt.plot(gtcrop[:,0], gtcrop[:,1])
            plt.imshow(imgD)
            plt.show()
        
        if argv['H5datagroup'] == 'voxels':
        
            visualisation(imgD, show=False)
            plotlabels((labels + 1.0) * 44)
            mlab.show()



if __name__ == '__main__':
    
    jsonfilename = sys.argv[1]
    with open(jsonfilename) as fr:
        argv = json.load(fr)
    argv['di'] = getattr(sys.modules[__name__], argv['di'])()


    makeCrops(argv)
    saveH5(argv)
    #showH5(argv)
    
        
        
    
    
