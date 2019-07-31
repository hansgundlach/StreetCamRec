# StreetCamRec
## Purpose

System Built to Recognize bikes,motorbikes and trucks in Pasto,Columbia.

## Model Design

## Data Collection 
Data was collected from video cameras from several streets in Pasto. Approximatley one hour of total footage was sampled into 
500 frames at random. The frames were than marked using LabelImg for bikes, motorbikes,taxis,and buses. 
## Training
Transfer learning was used to speed the training process. 
Model was trained using tensorflow ontop of faster_rcnn_incneption_v2_coco_2018_01_28. 

## Functionality 
The main method built to recognize vehicules is custom counting. Example implementation:
```python
is_color_recognition_enabled = 1 # set it to 1 for enabling the color prediction for the detected objects
fps = 23 # change it with your input video fps
width =640  # change it with your input video width
height =480 # change it with your input vide height    
targeted_objects = ['car','bicycle','person','bus','truck']
#input_video, detection_graph, category_index, is_color_recognition_enabled,targeted_objects, fps, width, height
object_counting_api.custom_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects,fps, width) 
```

## Results 

## Extension
