# StreetCamRec
## Purpose

System Built to Recognize bikes, motorbikes,  in Pasto, Columbia.
## Data Collection 
Data was collected from video cameras from several streets in Pasto. Approximatley one hour of total footage was sampled into 
500 frames at random. The frames were than manualy marked using LabelImg for bikes, motorbikes, taxis, and buses. 

## Training
Transfer learning was used to speed the training process. 
Model was trained using tensorflow ontop of faster_rcnn_incneption_v2_coco_2018_01_28. Faster_RCNN was found to be slower but significantly more accurate than the default ssd_mobilenet_v1_coco_2017_11_17 model. The model was trained for approximatley 10,000 epochs using the default faster_rcnn setting ie adam, etc. Below is an example training image from the training set. 

![alt text](https://github.com/hansgundlach/StreetCamRec/blob/master/test_images/image9.jpg)


## Functionality 
The model was than put into the obje_counting api as part of the tensorflow object_detection api. 
The main method built to recognize vehicules is custom counting. Custom counting sample a frame every minute which is fed into the object recognition network where objects are identifed and counted. The results are then written to a csv. 

## Personal Installation and Use 
To run StreetCamRec. Install Tensorflow Object Detection Library (https://github.com/tensorflow/models/tree/master/research/object_detection). After Installation of Tensorflow Object Detection Libarary place StreetCamRec inside the object_detectionn folder. 

Example implementation:

view countexample.py for the full example file to run 
```python
input_video =  "./input_images_and_videos/lowcut.mp4"

#Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_kitti7000'
#IMAGE_NAME = 'frame31.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')

# Number of classes the object detector can identify ie bik, motorcycle, etc 
NUM_CLASSES = 4

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
is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
fps = 23 # change it with your input video fps
width =640 #854#3840 #626#3840 # change it with your input video width
height =480 #480# 2160 #360#2160 # change it with your input vide height    
targeted_objects = ['bicycle','motorcycle','redbus','taxi']
roi = 300
deviation = 5
#object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
object_counting_api.custom_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width,height,roi,deviation)

output 
# the csv output is in this format 
time, bicycle ,motorcycle, 
2019-08-09 13:32 1,0,0,0
```
## Extension
Faster algorithms are availble for object detection. If you would like real-time object_detection consider using YOLO based algorithms. Further, object recognition parameters were set to default feel free to change the custom parameters in api/object_counting.py. For example, boxes are only show with a confidence score over 50%. Score can be changed to increase or decrease the number of boxes albeit with coresponding decreases and increases in accuracy (connfidence score). 
