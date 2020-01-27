#!/usr/bin/env python
"""
check zed odom and obstacle translation and rotation againts vicon
"""

import rospy
import numpy as np
import rospkg
import cv2
import ros_numpy
import message_filters

import pdb
import tf
import time
import matplotlib.pyplot as plt

from nav_msgs.msg import Odometry

class ErrCheck():

    def __init__(self):
        rospy.init_node("error_check")
        self.first_run = True
        self.t_zed = []
        self.R_zed = []
        self.t_vicon = []
        self.R_vicon = []
        self.R_origin = []
        self.T_origin = []

    def getROSdata(self):
        odom_zed = message_filters.Subscriber("/zed/zed_node/odom", Odometry)
        odom_vicon = message_filters.Subscriber("/vicon/chair/odom", Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([odom_zed, odom_vicon], 10, 0.1)
        rospy.loginfo('messages synchronized')
        ts.registerCallback(self.errorCallback)
        rospy.spin()


    def errorCallback(self,zed_odom, vicon_odom):



        T_zed_odom = [zed_odom.pose.pose.position.x, zed_odom.pose.pose.position.y, zed_odom.pose.pose.position.z]
        euler_zed = tf.transformations.euler_from_quaternion([zed_odom.pose.pose.orientation.x,
                                                              zed_odom.pose.pose.orientation.y,
                                                              zed_odom.pose.pose.orientation.z,
                                                              zed_odom.pose.pose.orientation.w])

        if self.first_run == True:

            self.first_run = False
            self.T_origin = [vicon_odom.pose.pose.position.x, vicon_odom.pose.pose.position.y, vicon_odom.pose.pose.position.z]
            R_origin = tf.transformations.quaternion_matrix([vicon_odom.pose.pose.orientation.x,
                                                                    vicon_odom.pose.pose.orientation.y,
                                                                    vicon_odom.pose.pose.orientation.z,
                                                                    vicon_odom.pose.pose.orientation.w])
            self.R_origin = np.linalg.inv(R_origin)


        R_vicon = tf.transformations.quaternion_matrix([vicon_odom.pose.pose.orientation.x,
                                                                vicon_odom.pose.pose.orientation.y,
                                                                vicon_odom.pose.pose.orientation.z,
                                                                vicon_odom.pose.pose.orientation.w])
        R_true = np.matmul(self.R_origin,R_vicon)


        T_vicon_odom = [vicon_odom.pose.pose.position.x, vicon_odom.pose.pose.position.y, vicon_odom.pose.pose.position.z]
        T_true = np.matmul(R_true[0:3,0:3], T_vicon_odom) - self.T_origin
        #T_true = np.asarray(T_vicon_odom) - np.asarray(self.T_origin)
        # pdb.set_trace()




        print("zed  ", T_zed_odom)
        print("vicon", T_true)
        return




if __name__=="__main__":
    O = ErrCheck()
    O.getROSdata()


## plot
