#!/usr/bin/env python

# https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/

import rospy
import tensorflow as tf
import rospkg
import cv2 as cv
import numpy as np
import time
import os
from cv_bridge import CvBridge, CvBridgeError
from sauvc_common.msg import Object 
from sensor_msgs.msg import Image

class object_detector:
    def __init__(self, front_camera_sub_topic, bottom_camera_sub_topic, confidence):
        node_name = rospy.get_name()
        rospy.loginfo(node_name + "node initializing...")
        # constants 
        rospack = rospkg.RosPack()
        path = rospack.get_path('sauvc_vision')
        labels_path = path + "/net/labels.txt"
        colors_path = path + "/net/colors.txt"
        weights_path = path + "/net/frozen_inference_graph.pb"
        config_path = path + "/net/opencv_graph.pbtxt"
        self.confidence = confidence
        self.labels = open(labels_path).read().strip().split("\n")
        self.colors = open(colors_path).read().strip().split("\n")
        # ROS Topic names
        gate_topic = node_name + "/gate"
        red_flare_topic = node_name + "/red_flare"
        yellow_flare_1_topic = node_name + "/yellow_flare_1"
        yellow_flare_2_topic = node_name + "/yellow_flare_2"
        mat_topic = node_name + "/mat_topic"
        red_bowl_1_topic = node_name + "/red_bowl_1_topic"
        red_bowl_2_topic = node_name + "/red_bowl_2_topic"
        red_bowl_3_topic = node_name + "/red_bowl_3_topic"
        blue_bowl_topic = node_name + "/blue_bowl_topic"
        dnn_image_topic = "/dnn/image"
        # init msg
        self.gate_message = Object()
        self.red_flare_message = Object()
        self.yellow_flare_1_message = Object()
        self.yellow_flare_2_message = Object()
        self.mat_message = Object()
        self.red_bowl_1_message = Object()
        self.red_bowl_2_message = Object()
        self.red_bowl_3_message = Object()
        self.blue_bowl_message = Object()
        self.messages = dict()
        self.messages['gate'] = self.gate_message
        self.messages['red_flare'] = self.red_flare_message
        self.messages['yellow_flare_1'] = self.yellow_flare_1_message
        self.messages['yellow_flare_2'] = self.yellow_flare_2_message
        self.messages['mat'] = self.mat_message
        self.messages['red_bowl_1'] = self.red_bowl_1_message
        self.messages['red_bowl_2'] = self.red_bowl_2_message
        self.messages['red_bowl_3'] = self.red_bowl_3_message
        self.messages['blue_bowl'] = self.blue_bowl_message
        # subscribers
        self.image_sub = rospy.Subscriber(front_camera_sub_topic, Image, self.callback, queue_size=1)
        # publishers
        self.gate_pub = rospy.Publisher(gate_topic, Object, queue_size=1)
        self.red_flare_pub = rospy.Publisher(red_flare_topic, Object, queue_size=1)
        self.yellow_flare_1_pub = rospy.Publisher(yellow_flare_1_topic, Object, queue_size=1)
        self.yellow_flare_2_pub = rospy.Publisher(yellow_flare_2_topic, Object, queue_size=1)
        self.mat_pub = rospy.Publisher(mat_topic, Object, queue_size=1)
        self.red_bowl_1_pub = rospy.Publisher(red_bowl_1_topic, Object, queue_size=1)
        self.red_bowl_2_pub = rospy.Publisher(red_bowl_2_topic, Object, queue_size=1)
        self.red_bowl_3_pub = rospy.Publisher(red_bowl_3_topic, Object, queue_size=1)
        self.blue_bowl_pub = rospy.Publisher(blue_bowl_topic, Object, queue_size=1)
        self.image_pub = rospy.Publisher(dnn_image_topic, Image, queue_size=1)
        self.publishers = dict()
        self.publishers['gate'] = self.gate_pub
        self.publishers['red_flare'] = self.red_flare_pub
        self.publishers['yellow_flare_1'] = self.yellow_flare_1_pub
        self.publishers['yellow_flare_2'] = self.yellow_flare_2_pub
        self.publishers['mat'] = self.mat_pub
        self.publishers['red_bowl_1'] = self.red_bowl_1_pub
        self.publishers['red_bowl_2'] = self.red_bowl_2_pub
        self.publishers['red_bowl_3'] = self.red_bowl_3_pub
        self.publishers['blue_bowl'] = self.blue_bowl_pub
        # init cv_bridge
        self.bridge = CvBridge()
        # load our NET from disk
        rospy.loginfo("Loading NET from disk...")
        self.cvNet = cv.dnn.readNetFromTensorflow(weights_path, config_path)
            
    def callback(self,data):
        try:
            # convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # detect our objects
            objects = self.detector(cv_image)
            # publish object coordinates and existance
            for key in objects.keys():
                obj = objects.get(key)
                self.messages[key].name = obj.get('name')
                self.messages[key].is_exist = obj.get('is_exist')
                self.messages[key].x_start = obj.get('x_start')
                self.messages[key].y_start = obj.get('y_start')
                self.messages[key].x_end = obj.get('x_end')
                self.messages[key].y_end = obj.get('y_end')
                self.messages[key].x_center = obj.get('x_center')
                self.messages[key].y_center = obj.get('y_center')
                self.publishers[key].publish(self.messages[key])
            # draw bounding boxes
            dnn_cv_image = self.draw(cv_image, objects)
            # convert cv image into ros format
            ros_image = self.bridge.cv2_to_imgmsg(dnn_cv_image, "bgr8")
            # publish image after dnn
            self.image_pub.publish(ros_image)
        except CvBridgeError as e:
            print(e)

    def detector(self, img):
        # construct a blob from the input image and then perform a
        # forward pass, giving us the bounding box
        # coordinates of the objects in the image
        self.cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
        start = time.time()
        cvOut = self.cvNet.forward()
        end = time.time()
        # show timing information and volume information on NET
        #rospy.loginfo("Took {:.6f} seconds".format(end - start))
        # dictionary with objects: obj[object_id, object_label_id, confidence]
        objects = dict()
        # go through all detected objects
        for i in range(0, cvOut.shape[2]):
            # check confidence
            object_confidence = cvOut[0, 0, i, 2]
            object_id = i
            object_label_id = int(cvOut[0, 0, i, 1])
            if object_confidence >= self.confidence:
                key = self.labels[object_label_id]
                if objects.get(key):
                    if  object_confidence > objects.get(key)[2]:
                        objects[key] = object_id, object_label_id, object_confidence
                    else:
                        continue
                else:
                    objects[key] = object_id, object_label_id, object_confidence
        for key in objects.keys():
            id, label_id, confidence = objects.pop(key)
            rows = img.shape[0]
            cols = img.shape[1]
            box = cvOut[0, 0, id, 3:7] * np.array([cols, rows, cols, rows])
            (x_start, y_start, x_end, y_end) = box.astype("int")
            x_center = int((x_end + x_start) / 2.0)
            y_center = int((y_end + y_start) / 2.0)
            objects[key] = {'name': self.labels[id],
                            'is_exist': True,
                            'confidence': confidence,
                            'x_start': x_start,
                            'y_start': y_start,
                            'x_end': x_end,
                            'y_end': y_end,
                            'x_center': x_center,
                            'y_center': y_center}
        
        return objects

    def draw(self, img, objects):
        # scale the bounding box coordinates back relative to the
        # size of the image and then compute the width and the height
        # of the bounding box
        for key in objects.keys():
            obj = objects.get(key)
            #draw bounding box on image
            cv.rectangle(img, (obj['x_start'], obj['y_start']), (obj['x_end'], obj['y_end']), (173,255,47), 4)  
            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = obj['name'] + str(obj['confidence'])
            cv.putText(img, text, (obj['x_start'], obj['y_start'] + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (173,255,47), 2)
        
        return img

if __name__ == '__main__':
    rospy.init_node('object_detector')
    # parameters
    front_camera_sub_topic = rospy.get_param('~front_camera_sub_topic')
    bottom_camera_sub_topic = rospy.get_param('~bottom_camera_sub_topic')
    dnn_confidence = int(rospy.get_param('~dnn_confidence'))
    try:
        gt = object_detector(front_camera_sub_topic, bottom_camera_sub_topic, dnn_confidence)
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")
    cv.destroyAllWindows()
