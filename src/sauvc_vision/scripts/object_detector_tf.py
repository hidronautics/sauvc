#!/usr/bin/env python

import tensorflow as tf

import rospy
import rospkg
import cv2 as cv
import numpy as np
import time
import os
from cv_bridge import CvBridge, CvBridgeError
from sauvc_common.msg import Objects
from sensor_msgs.msg import Image


class object_detector:
    def __init__(self, front_camera_sub_topic, bottom_camera_sub_topic, confidence):
        node_name = rospy.get_name()
        rospy.loginfo("%s node initializing", node_name)
        # constants
        rospack = rospkg.RosPack()
        path = rospack.get_path('sauvc_vision')
        net_path = path + "/net/"
        labels_path = net_path + "label_map.pbtxt"
        self.confidence = confidence
        self.labels = open(labels_path).read().strip().split("\n")
        # ROS Topic names
        objects_topic = node_name + "/objects"
        dnn_image_topic = "/dnn/image"
        # init msg
        self.objects_message = Objects()
        # subscribers
        self.image_sub = rospy.Subscriber(
            front_camera_sub_topic, Image, self.callback, queue_size=1)
        # publishers
        self.objects_pub = rospy.Publisher(objects_topic, Objects, queue_size=1)
        self.image_pub = rospy.Publisher(dnn_image_topic, Image, queue_size=1)
        # init cv_bridge
        self.bridge = CvBridge()
        # load our NET from disk
        rospy.loginfo("Loading NET from disk...")
        self.detection_model = load_model(net_path)

    def load_model(self, model_dir):
        model = tf.saved_model.load(model_dir)
        return model

    def inference(self, model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        model_fn = model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))

        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}

        # detection_classes should be ints.
        detection_boxes = output_dict['detection_boxes']
        detection_classes = output_dict['detection_classes'].astype(int)
        detection_scores = output_dict['detection_scores']

        true_detection_classes = []
        true_detection_box_x_start = []
        true_detection_box_y_start = []
        true_detection_box_x_end = []
        true_detection_box_y_end = []
        true_detection_scores = []
        for index, confidence in enumerate(detection_scores):
            if confidence > 0.5:
                true_detection_classes.append(detection_classes[index])
                true_detection_box_x_start.append(
                    int(detection_boxes[index][0] * 640))
                true_detection_box_y_start.append(
                    int(detection_boxes[index][1] * 480))
                true_detection_box_x_end.append(
                    int(detection_boxes[index][2] * 640))
                true_detection_box_y_end.append(
                    int(detection_boxes[index][3] * 480))
                true_detection_scores.append(detection_scores[index])
        return true_detection_classes, true_detection_box_x_start, true_detection_box_y_start, true_detection_box_x_end, true_detection_box_y_end, true_detection_scores

    def callback(self, data):
        try:
            # convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # detect our objects
            raw_objects = self.inference(self.detection_model, cv_image)

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
            rospy.logwarn("%s | Can't load image", node_name)

    def draw(self, img, objects):
        # scale the bounding box coordinates back relative to the
        # size of the image and then compute the width and the height
        # of the bounding box
        for key in objects.keys():
            obj = objects.get(key)
            # draw bounding box on image
            cv.rectangle(img, (obj['x_start'], obj['y_start']),
                         (obj['x_end'], obj['y_end']), (173, 255, 47), 4)
            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = obj['name'] + str(obj['confidence'])
            cv.putText(img, text, (obj['x_start'], obj['y_start'] + 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (173, 255, 47), 2)

        return img


if __name__ == '__main__':
    rospy.init_node('object_detector')
    # parameters
    front_camera_sub_topic = rospy.get_param('~front_camera_sub_topic')
    bottom_camera_sub_topic = rospy.get_param('~bottom_camera_sub_topic')
    dnn_confidence = int(rospy.get_param('~dnn_confidence'))
    try:
        gt = object_detector(front_camera_sub_topic,
                             bottom_camera_sub_topic, dnn_confidence)
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")
    cv.destroyAllWindows()
