# StreetCamRec
## Purpose

System Built to Recognize bikes, motorbikes and trucks in Pasto, Columbia.

## Model Design

## Data Collection 
Data was collected from video cameras from several streets in Pasto. Approximatley one hour of total footage was sampled into 
500 frames at random. The frames were than manualy marked using LabelImg for bikes, motorbikes, taxis, and buses. 

## Training
Transfer learning was used to speed the training process. 
Model was trained using tensorflow ontop of faster_rcnn_incneption_v2_coco_2018_01_28. Faster_RCNN was found to be slower but significantly more accurate than the default ssd_mobilenet_v1_coco_2017_11_17 model. The model was trained for approximatley 10,000 epochs using the default faster_rcnn setting ie adam, etc . Below is an example training image from the training set. 

![alt text](https://github.com/hansgundlach/StreetCamRec/blob/master/test_images/image9.jpg)

## Functionality 
The model was than put into the obje_counting api as part of the tensorflow object_detection api. 
The main method built to recognize vehicules is custom counting. Custom counting sample a frame every minute which is fed into the object recognition network where objects are identifed and counted. The results are then written to a csv. 

Example implementation:
```python
is_color_recognition_enabled = 1 # set it to 1 for enabling the color prediction for the detected objects
fps = 23 # change it with your input video fps
width =640  # change it with your input video width
height =480 # change it with your input vide height    
targeted_objects = ['car','bicycle','person','bus','truck'] #objects which the network will count
object_counting_api.custom_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects,fps, width) 


output 

```
## Results 

## Extension
Faster algorithms are availble for object detection. If you would like real-time object_detection consider using YOLO based algorithms. Further, object recognition parameters were set to default feel free to change the custom parameters in api/object_counting.py. For example, boxes are only show with a confidence score over 50%. Score can be changed to increase or decrease the number of boxes albeit with coresponding decreases and increases in accuracy (connfidence score). 
