#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pcms.openvino_models import *

def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        

dnn_yolo = Yolov8("bagv5",device_name="GPU")
dnn_yolo.classes = ["obj"]

if __name__=='__main__':
    rospy.init_node("node_test")
    _image = None
    _topic_image = "/cam1/color/image_raw"
    rospy.Subscriber(_topic_image, Image, callback_image)
    rospy.wait_for_message(_topic_image, Image)
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        image = _image.copy()
        results = dnn_yolo.forward(image)
        for boxes in results:
            for i, (x1, y1, x2, y2, score, cls) in enumerate(results[0]["det"]):
                x1, y1, x2, y2, cls = map(int, (x1, y1, x2, y2, cls))      
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0),1)
                cv2.putText(image, str(cls), (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 1)  
        cv2.imshow('detection',image)
        if cv2.waitKey(1) == 27: break
        
