#!/usr/bin/env python

# https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/

import rospy
import rospkg
import cv2 as cv
import numpy as np
import time
import os
from cv_bridge import CvBridge, CvBridgeError
from sauvc_common.msg import Object 
from sensor_msgs.msg import Image

class gate_detector:
    def __init__(self, camera_sub_topic, gate_pub_topic, dnn_pub_topic, device, confidence):
        rospy.loginfo("gate_detector node initializing")
        # init cv_bridge
        self.bridge = CvBridge()
        # init publishers and subscribers
        self.image_sub = rospy.Subscriber(camera_sub_topic, Image, self.callback, queue_size=1)
        self.gate_pub = rospy.Publisher(gate_pub_topic, Object, queue_size=1)
        self.dnn_image_pub = rospy.Publisher(dnn_pub_topic, Image, queue_size=1)
        self.gate_message = Object()

        # get ros parameters
        self.device = device
        self.confidence = confidence
        # get package path
        rospack = rospkg.RosPack()
        rospack.list() 
        path = rospack.get_path('sauvc_vision')
        # other constants
        self.labels = path + "/net/labels.txt"
        self.colors = path + "/net/colors.txt"
        self.weights = path + "/net/frozen_inference_graph.pb"
        self.config = path + "/net/opencv_graph.pbtxt"
        # load our NET from disk
        rospy.loginfo("Loading NET from disk...")
        self.cvNet = cv.dnn.readNetFromTensorflow(self.weights, self.config)
            
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            is_exist, x_start, y_start, x_end, y_end, x_center, y_center, dnn_cv_image = self.detector(cv_image)
            #publish gate coordinates and existance
            self.gate_message.is_exist = is_exist
            self.gate_message.x_start = x_start
            self.gate_message.y_start = y_start
            self.gate_message.x_end = x_end
            self.gate_message.y_end = y_end
            self.gate_message.x_center = x_center
            self.gate_message.y_center = y_center
            self.gate_pub.publish(self.gate_message)
            #convert cv image into ros format
            ros_image = self.bridge.cv2_to_imgmsg(dnn_cv_image, "bgr8")
            #publish image after dnn
            self.dnn_image_pub.publish(ros_image)
        except CvBridgeError as e:
            print(e)

    def detector(self, image):
        # construct a blob from the input image and then perform a
        # forward pass, giving us the bounding box
        # coordinates of the objects in the image
        self.cvNet.setInput(cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
        start = time.time()
        cvOut = self.cvNet.forward()
        end = time.time()
        # show timing information and volume information on NET
        rospy.loginfo("Took {:.6f} seconds".format(end - start))
        # find gate
        gate_confidence = 0
        gate_confidence_index = 0
        is_exist = False
        x_start = 0
        y_start = 0
        x_end = 0
        y_end = 0
        x_center = 0
        y_center = 0
        for i in range(0, cvOut.shape[2]):
            # check confidence
            confidence = cvOut[0, 0, i, 2]
            # check object ID
            if (int(cvOut[0, 0, i, 1]) == 1) and (confidence >= self.confidence) and (confidence >= gate_confidence):
                gate_confidence = confidence
                gate_confidence_index = i
                is_exist = True
        # scale the bounding box coordinates back relative to the
        # size of the image and then compute the width and the height
        # of the bounding box
        if is_exist:
            rows = image.shape[0]
            cols = image.shape[1]
            box = cvOut[0, 0, gate_confidence_index, 3:7] * np.array([cols, rows, cols, rows])
            (x_start, y_start, x_end, y_end) = box.astype("int")
            x_center = int(x_end - x_start + x_start)
            y_center = int(y_end - y_start + y_start)
            #draw bounding box on image
            boxW = x_end - x_start
            boxH = y_end - y_start 
            cv.rectangle(image, (x_start, y_start), (x_end, y_end), (173,255,47), 4)  
            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = "gate: {:.4f}".format(self.confidence)
            cv.putText(image, text, (x_start, y_start + 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (173,255,47), 2)

        return is_exist, x_start, y_start, x_end, y_end, x_center, y_center, image

if __name__ == '__main__':
    rospy.init_node('gate_detector')
    # parameters
    camera_sub_topic = rospy.get_param('~camera_sub_topic')
    gate_pub_topic = rospy.get_param('~gate_pub_topic')
    dnn_pub_topic = rospy.get_param('~dnn_pub_topic')
    video_device = rospy.get_param('~video_device')
    dnn_confidence = int(rospy.get_param('~dnn_confidence'))
    try:
        gt = gate_detector(camera_sub_topic, gate_pub_topic, dnn_pub_topic, video_device, dnn_confidence)
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")
