#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *

def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        
dnn_pose = HumanPoseEstimation(device_name="GPU")

if __name__=='__main__':
    rospy.init_node("node_test")
    _image = None
    _topic_image = "/cam1/color/image_raw"
    rospy.Subscriber(_topic_image, Image, callback_image)
    rospy.wait_for_message(_topic_image, Image)
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        image = _image.copy()
        poses = dnn_pose.forward(image)
        for pose in poses:
            x1 = 10000
            y1 = 10000
            x2 = -1
            y2 = -1
            for cx, cy, score in pose:
                if cx!=0 and cy!=0:
                    x1 = int(min(x1, cx))
                    y1 = int(min(y1, cy))
                    x2 = int(max(x2, cx))
                    y2 = int(max(y2, cy))
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0),1)
        cv2.imshow('detection',image)
        if cv2.waitKey(1) == 27: break
