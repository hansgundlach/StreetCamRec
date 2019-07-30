#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:55:06 2019

@author: hansgundlach
"""

#test of countapi
# Imports
import tensorflow as tf
import os

# Object detection imports
from utils import backbone
from api import object_counting_api


from utils import label_map_util



input_video =  "./input_images_and_videos/streetvid1.mp4" #  "./input_images_and_videos/The Dancing Traffic Light Manikin by smart.mp4" #"./input_images_and_videos/videoPasto2.mov" #"./input_images_and_videos/pedestrian_survaillance.mp4" #"./input_images_and_videos/videoPasto2.mov"

#if you want to use defaukt model 
#detection_graph, category_index = backbone.set_model('saved_model')

#Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
#IMAGE_NAME = 'frame31.jpg'

# Grab path to current working directory
CWD_PATH = "/Users/hansgundlach/Downloads/TensorStart/models/research/object_detection/StreetCamRec"#os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')


# Number of classes the object detector can identify ie bik, motorcycle, etc 
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#real time video object recognition

#dimension of streetvid1
is_color_recognition_enabled = 1 # set it to 1 for enabling the color prediction for the detected objects
fps = 23 # change it with your input video fps
width =640 #854#3840 #626#3840 # change it with your input video width
height =480 #480# 2160 #360#2160 # change it with your input vide height    
targeted_objects = ['car','bicycle','person','bus','truck']
#object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
object_counting_api.custom_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width,height)