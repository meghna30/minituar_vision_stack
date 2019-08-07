#!/usr/bin/env python

import rospy
import numpy as np
import rospkg
import cv2
import ros_numpy
import message_filters
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo, PointCloud2


class PlaneFiltering():
    def __init__(self):
        rospy.init_node("Segmentation")
        self.P = [] # list of points
        self.R = [] # list of normals
        self.C = [] # list of convex polygons
        self.O = [] # list of outliers
        self.delta_plane =  50 # neighbourhood for gobal samples(in pixels)
        self.k_max = 50 # maximum number of neighbourhoods to sample
        self.S = 1.5 # plane size in world space for local samples
        self.error = 0.1 # min plane offset error for inliers
        self.alpha = 0.5 # min fraction of inliers to accept local sample
        self.l = 80 # number of local samples
        #self.K = rospy.wait_for_message("/zed/depth/camera_info", CameraInfo)
        #print(self.K)
        self.f_h = 45#self.K[0,0]
        self.f_v = 45#self.K[1,1]
        self.bridge = CvBridge()


    def extract_plane(self):
        I = rospy.Subscriber("/zed/depth/depth_registered", Image, self.doPlaneFiltering)
        rospy.spin()

    def doPlaneFiltering(self, I):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(I)
        except CvBridgeError as e:
            print(e)
        h,w = np.shape(cv_image) # size of the image
        n = 0
        k = 0
        self.P = [] # list of points
        self.R = [] # list of normals
        self.C = [] # list of convex polygons
        self.O = [] # list of outliers\
        self.n_max = h*w

        while n < self.n_max and k < self.k_max:
            k = k+1
            d0 = [np.random.randint(0,h), np.random.randint(0,w)]
            d1 = d0 + [np.random.randint(-self.delta_plane, self.delta_plane),np  .random.randint(-self.delta_plane, self.delta_plane)]
            d2 = d0 + [np.random.randint(-self.delta_plane, self.delta_plane),np  .random.randint(-self.delta_plane, self.delta_plane)]
            p0 = self.PixeltoWorld(cv_image, d0, h, w)
            p1 = self.PixeltoWorld(cv_image, d1, h, w)
            p2 = self.PixeltoWorld(cv_image, d2, h, w)
            # normal to the plane
            r = np.cross((p1-p0),(p2-p0))/np.linalg.norm(np.cross((p1-p0),(p2-p0)))
            z_mean = (p0[2] + p1[2] + p2[2])/3
            w_window = w*(self.S/z_mean)*np.tan(self.f_h)
            h_window = h*(self.S/z_mean)*np.tan(self.f_v)
            num_inliers = 0
            P_ = []
            R_ = []
            c_ = []

            for j in range(3,self.l):
                dj = d0 + [np.random.randint(-h_window/2, h_window/2) , np.random.randint(-w_window/2, w_window/2)]
                pj = self.PixeltoWorld(cv_image, dj, h, w)
                err = np.abs(np.dot(r,(pj - p0)))
                if err < self.error:
                    P_.append(pj)
                    R_.append(r)
                    num_inliers +=1

            if num_inliers > self.alpha*self.l:
                self.P.append(P_)
                self.R.append(R_)
                ## construct convex polygon c from P_
                # c = []
                # self.C.append(c)
                n = n + num_inliers
            else:
                self.O.append(P_)





    def PixeltoWorld(self, image, d,w,h):
        px = image[d[0],d[1]]*(d[1]/(w-1)-0.5)*np.tan(self.f_h/2)
        py = image[d[0],d[1]]*(d[0]/(h-1)-0.5)*np.tan(self.f_v/2)
        pz = image[d[0],d[1]]
        p = [px,py,pz]
        p = np.asarray(p)
        return p




if __name__=='__main__':
    P = PlaneFiltering()
    P.extract_plane()
