#!/usr/bin/env python

import cv2 as cv
import json
import os
import numpy as np
import time
import os
from random import randint
from itertools import groupby

import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sauvc_common.msg import Object
from sauvc_common.msg import ObjectsArray
from sensor_msgs.msg import Image


class object_detector:
    def __init__(self, input_image_topic, confidence):
        # get node name
        node_name = rospy.get_name()
        rospy.loginfo("{} node initializing".format(node_name))
        # get paths
        rospack = rospkg.RosPack()
        path = rospack.get_path('sauvc_vision')
        weights_path = os.path.sep.join(
            [path, "net", "frozen_inference_graph.pb"])
        labels_path = os.path.sep.join([path, "net", "labels.json"])
        config_path = os.path.sep.join([path, "net", "opencv_graph.pbtxt"])
        self.confidence = confidence
        # read labels
        with open("labels.json") as json_file:
            self.labels = json.loads(json_file.read())["labels"]

        # ROS Topic names
        objects_array_topic = "/{}/objects".format(node_name)
        output_image_topic = "/{}/image".format(node_name)

        # init msg
        self.object_msg = Object()
        self.objects_array_msg = ObjectsArray()

        # subscribers
        self.image_sub = rospy.Subscriber(
            input_image_topic, Image, self.callback, queue_size=1)

        # publishers
        self.objects_array_pub = rospy.Publisher(
            objects_array_topic, ObjectsArray, queue_size=10)
        self.image_pub = rospy.Publisher(
            output_image_topic, Image, queue_size=1)

        # init cv_bridge
        self.bridge = CvBridge()
        # load our NET from disk
        rospy.loginfo("Loading NET from disk...")
        self.cvNet = cv.dnn.readNetFromTensorflow(weights_path, config_path)

    def callback(self, data):
        try:
            # convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # detect our objects
            dnn_objects = self.detector(cv_image)
            # publish objects
            for dnn_object in dnn_objects:
                self.object_msg.name = dnn_object["name"]
                self.object_msg.confidence = dnn_object["confidence"]
                x_start = dnn_object['box'][0]
                self.object_msg.x_start = x_start
                y_start = dnn_object['box'][1]
                self.object_msg.y_start = y_start
                x_end = dnn_object['box'][2]
                self.object_msg.x_end = x_end
                y_end = dnn_object['box'][3]
                self.object_msg.y_end = y_end
                x_center = x_start + (x_end - x_start)/2
                self.object_msg.x_center = x_center
                y_center = y_center + (y_end - y_center)/2
                self.object_msg.y_center = y_center
                self.objects_array_msg.push_back(object_msg)
            self.objects_array_pub(self.objects_array_msg)
            # draw bounding boxes
            dnn_cv_image = self.draw(cv_image, dnn_objects)
            # convert cv image into ros format
            ros_image = self.bridge.cv2_to_imgmsg(dnn_cv_image, "bgr8")
            # publish output image
            self.image_pub.publish(ros_image)
        except CvBridgeError as e:
            print(e)

    def detector(self, img):
        # construct a blob from the input image and then perform a
        # forward pass, giving us the bounding box
        # coordinates of the objects in the image
        self.cvNet.setInput(cv.dnn.blobFromImage(
            img, size=(300, 300), swapRB=True, crop=False))
        start = time.time()
        cvOut = self.cvNet.forward()
        end = time.time()
        # show timing information and volume information on NET
        #rospy.loginfo("Took {:.6f} seconds".format(end - start))
        # dictionary with objects: obj[object_id, object_label_id, confidence]
        objects = []
        rows = img.shape[0]
        cols = img.shape[1]
        # go through all detected objects
        for i in range(0, cvOut.shape[2]):
            # check confidence
            classID = int(cvOut[0, 0, i, 1])
            confidence = cvOut[0, 0, i, 2]
            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimum probability
            if confidence > self.confidence:
                object_ = {}
                object_["name"] = self.labels[classID-1]
                object_["confidence"] = confidence
                # clone our original image so we can draw on it
                clone = img.copy()
                # scale the bounding box coordinates back relative to the
                # size of the image and then compute the width and the height
                # of the bounding box
                box = cvOut[0, 0, i, 3:7] * np.array([cols, rows, cols, rows])
                object_["box"] = box.astype("int")
                objects.append(object_)
        # filter founded objects
        # group by name and sort
        groups = [(group_name, list(group)) for group_name, group in groupby(
            sorted(objects, key=lambda label: label['name']), lambda label: label['name'])]
        objects = []
        # go through groups and pop unnecessary items
        for group in groups:
            object_count_from_json = list(
                filter(lambda label: label['name'] == group[0], labels))[0]["count"]
            del group[1][object_count_from_json:]
            objects += group[1]
        return objects

    def deleteMultipleObjects(self, objects):
        pass

    def draw(self, img, objects):
        for dnn_object in objects:
            # get box coordinates
            x_start = dnn_object['box'][0]
            y_start = dnn_object['box'][1]
            x_end = dnn_object['box'][2]
            y_end = dnn_object['box'][3]
            x_center = int(x_start + (x_end - x_start)/2)
            y_center = int(y_center + (y_end - y_center)/2)
            # random color
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            # draw rectangle and center point
            cv.rectangle(img, (x_start, y_start),
                         (x_end, y_end), color, thickness=1)
            cv.line(img, (x_center - 5, y_center - 5),
                    (x_center + 5, y_center + 5), color, thickness=1)
            cv.line(img, (x_center + 5, y_center - 5),
                    (x_center - 5, y_center + 5), color, thickness=1)
            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            if (y_start < 15):
                text_name = "{}".format(dnn_object["name"])
                cv.putText(img, text_name, (x_start, y_start - 5),
                           cv.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                text_confidence = "{:.2f}".format(confidence)
                cv.putText(img, text_confidence, (x_start, y_start + 15),
                           cv.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
        return img


if __name__ == '__main__':
    rospy.init_node('object_detector')
    # parameters
    input_image_topic = rospy.get_param('~input_image_topic')
    dnn_confidence = int(rospy.get_param('~dnn_confidence'))
    try:
        gt = object_detector(input_image_topic, dnn_confidence)
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down {} node".format(rospy.get_name()))
    cv.destroyAllWindows()
